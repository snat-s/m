# Originally from Datacomp
# Found here: https://github.com/mlfoundations/datacomp/blob/main/aggregate_scores.py
# but modified so that i can get an entire table in markdown for a folder of evaled models.

import argparse
import os
import numpy as np
import pandas as pd

DATASET_GROUPS = {
    "ImageNet dist. shifts": {
        "ImageNet Sketch",
        "ImageNet v2",
        "ImageNet-A",
        "ImageNet-O",
        "ImageNet-R",
        "ObjectNet",
    },
    "VTAB": {
        "Caltech-101",
        "CIFAR-100",
        "CLEVR Counts",
        "CLEVR Distance",
        "Describable Textures",
        "EuroSAT",
        "KITTI Vehicle Distance",
        "Oxford Flowers-102",
        "Oxford-IIIT Pet",
        "PatchCamelyon",
        "RESISC45",
        "SVHN",
        "SUN397",
    },
    "Retrieval": {"Flickr", "MSCOCO", "WinoGAViL"},
}

def get_aggregate_scores(results_file):
    """Returns a dictionary with aggregated scores from a results file."""
    df = pd.read_json(results_file, lines=True)
    df = pd.concat(
        [df.drop(["metrics"], axis=1), df["metrics"].apply(pd.Series)], axis=1
    )
    df = df.dropna(subset=["main_metric"])
    assert len(df) == 38, f"Results file has unexpected size, {len(df)}"
    results = dict(zip(df.dataset, df.main_metric))
    aggregate_results = {"ImageNet": results["ImageNet 1k"]}
    for group, datasets in DATASET_GROUPS.items():
        score = np.mean([results[dataset] for dataset in datasets])
        aggregate_results[group] = score
    aggregate_results["Average"] = np.mean(list(results.values()))
    return aggregate_results

def generate_markdown_table(folder_path):
    """Generates a markdown table for all results files in the given folder."""
    all_scores = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.jsonl'):
            model_name = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)
            scores = get_aggregate_scores(file_path)
            all_scores[model_name] = scores
    
    # Create the markdown table
    headers = ["Model"] + list(next(iter(all_scores.values())).keys())
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "| " + " | ".join(["---" for _ in headers]) + " |\n"
    
    for model, scores in all_scores.items():
        row = [model] + [f"{score:.3f}" for score in scores.values()]
        markdown_table += "| " + " | ".join(row) + " |\n"
    
    return markdown_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the folder containing results files."
    )
    args = parser.parse_args()
    
    markdown_table = generate_markdown_table(args.input)
    print(markdown_table)   
