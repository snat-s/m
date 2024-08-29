import fasttext
import polars as pl

def is_english(text):
    text = text.replace('\n', ' ')
    lang, *_ = lang_model.predict(text, k=1)
    return lang[0] == '__label__en'

def filter_values(high_value_clusters, labels_df):
    df = high_value_clusters.filter(pl.col("Classification") == "High Value")
    high_value_cluster_list = df["Cluster"].to_list()
    
    filtered_labels = labels_df.filter(pl.col("cluster_label").is_in(high_value_cluster_list))
    
    filtered_labels = filtered_labels.with_columns(
        pl.col("data_points").map_elements(is_english, return_dtype=pl.Boolean).alias("is_english")
    )
    print(filtered_labels) 
    # Keep only English text
    filtered_labels = filtered_labels.filter(pl.col("is_english") == True)
    print(filtered_labels) 
    return filtered_labels.select(["uid", "data_points", "cluster_label"])

fasttext.FastText.eprint = lambda x: None  # Suppress warnings
lang_model = fasttext.load_model('./models/lid.176.bin')

good_clusters_df = pl.read_csv("./data/classification_of_clusters.csv")
labels_df = pl.read_csv("./data/cluster_labels.csv")

filtered_df = filter_values(good_clusters_df, labels_df)
filtered_df.write_csv("./data/filtered_high_value_english_data.csv")
