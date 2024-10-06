import polars as pl
import fasttext

def is_english(text):
    text = text.replace('\n', ' ')
    lang, *_ = lang_model.predict(text, k=1)
    return lang[0] == '__label__en'

def filter_values(high_value_clusters, labels_df, filter_english=False):
    df = high_value_clusters.filter(pl.col("Classification").is_in(["Acceptable", "**Acceptable**"]))
    high_value_cluster_list = df["Cluster"].to_list()
    
    filtered_labels = labels_df.filter(pl.col("cluster_label").is_in(high_value_cluster_list))
    
    if filter_english:
        filtered_labels = filtered_labels.with_columns(
            pl.col("data_points").map_elements(is_english, return_dtype=pl.Boolean).alias("is_english")
        )
        filtered_labels = filtered_labels.filter(pl.col("is_english") == True)
    
    return filtered_labels.select(["uid", "data_points", "cluster_label"])

def filter_clusters(input_classifications_file, input_labels_file, output_file, filter_english=False):
    if filter_english:
        # Suppress warnings from fasttext
        fasttext.FastText.eprint = lambda x: None
        
        global lang_model
        lang_model = fasttext.load_model('./models/lid.176.bin')
    
    good_clusters_df = pl.read_csv(input_classifications_file)
    labels_df = pl.read_csv(input_labels_file)
    
    filtered_df = filter_values(good_clusters_df, labels_df, filter_english)
    filtered_df.write_csv(output_file)
    print(f"Filtered high value data saved to: {output_file}")

if __name__ == "__main__":
    input_classifications_file = "./non_normalized_data/classifications_minipile_style.csv"
    input_labels_file = "./non_normalized_data/cluster_labels.csv"
    output_file = "./non_normalized_data/filtered_high_value_data_minipile_style.csv"
    filter_clusters(input_classifications_file, input_labels_file, output_file, filter_english=False)
