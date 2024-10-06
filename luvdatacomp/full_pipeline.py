"""
You should only run this for the full pipeline.
"""
import os
from cluster import cluster
from classify import classify
from parse_classifications import parse_classifications
from filter_clusters import filter_clusters

def output_path(filename):
    return os.path.join(DEFAULT_OUTPUT_DIR, filename)

def main(metadata_path, cluster_samples_path, prompt_file, classifications_file, 
         parsed_classifications_file, cluster_labels_file, filtered_output_file, filter_english=False):
    #cluster(metadata_path, cluster_labels_file, cluster_samples_path)
    #classify(cluster_samples_path, prompt_file, classifications_file)
    parse_classifications(classifications_file, parsed_classifications_file)
    filter_clusters(parsed_classifications_file, cluster_labels_file, filtered_output_file, filter_english)
    print(f"Finished filtering the clusters! Find the results at {filtered_output_file}")

if __name__ == "__main__":
    # output dir
    DEFAULT_OUTPUT_DIR = "minipile_style_txt"
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    # configs
    METADATA_PATH = "/mnt/data/small_datacomp/metadata/" # place where you have .parquet and .npz files
    CLUSTER_SAMPLES_PATH = output_path("cluster_samples.json")
    PROMPT_FILE = "prompt.txt"
    CLASSIFICATIONS_FILE = output_path("cluster_classifications_on_minipile_similar_prompt.json")
    PARSED_CLASSIFICATIONS_FILE = output_path("classifications_minipile_style.csv")
    CLUSTER_LABELS_FILE = output_path("cluster_labels.csv")
    FILTERED_OUTPUT_FILE = output_path("filtered_high_value_data_minipile_style.csv")
    FILTER_ENGLISH = False 

    main(METADATA_PATH, CLUSTER_SAMPLES_PATH, PROMPT_FILE, CLASSIFICATIONS_FILE, 
         PARSED_CLASSIFICATIONS_FILE, CLUSTER_LABELS_FILE, FILTERED_OUTPUT_FILE, FILTER_ENGLISH)
