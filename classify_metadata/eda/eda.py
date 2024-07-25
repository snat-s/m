import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import glob
from tqdm import tqdm
import os
import pyarrow.parquet as pq
import pandas as pd
import umap
import random
import colorsys

def load_first_n_samples(directory, n_samples=4_000_000):
    all_embeddings = []
    for file in tqdm(glob.glob(f"{directory}/*_embeddings.npz")):
        with np.load(file) as data:
            embeddings = data['embeddings']
            all_embeddings.append(embeddings)
        if sum(len(emb) for emb in all_embeddings) >= n_samples:
            break
    return np.concatenate(all_embeddings, axis=0)[:n_samples]

def load_first_n_predictions(parquet_directory, n_samples=4_000_000):
    predictions = []
    for file in glob.glob(f"{parquet_directory}/output_batch_*.parquet"):
        table = pq.read_table(file)
        df = table.to_pandas()
        predictions.extend(df['prediction'].tolist())
        if len(predictions) >= n_samples:
            break
    return predictions[:n_samples]

def perform_umap(embeddings, n_components=3):
    reducer = umap.UMAP()
    return reducer.fit_transform(embeddings)

def perform_pca(embeddings, n_components=3):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embeddings)

def generate_distinct_colors(n):
    hue_partition = 1.0 / (n + 1)
    colors = []
    for i in range(n):
        hue = i * hue_partition
        saturation = 0.8 + random.uniform(0, 0.2)  # High saturation with some variation
        lightness = 0.5 + random.uniform(0, 0.2)   # Medium to high lightness
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb)
    random.shuffle(colors)  # Shuffle to avoid adjacent similar colors
    return colors

def visualize_pca_2d_per_class(pca_result, predictions, output_directory, manifold_decomposition):
    valid_mask = np.array(predictions) != 'NA' 
    print(valid_mask.sum())
    pca_result = pca_result[valid_mask]
    predictions = np.array(predictions)[valid_mask]

    # Get unique classes and assign colors
    unique_classes = list(set(predictions))
    colors = generate_distinct_colors(len(unique_classes)) #plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
    color_dict = dict(zip(unique_classes, colors))

    # Create a plot for each class
    for class_name in unique_classes:
        plt.figure(figsize=(10, 8))
        mask = np.array(predictions) == class_name
        plt.scatter(
            pca_result[mask, 0], 
            pca_result[mask, 1], 
            c=[color_dict[class_name]],
            s=0.1,  # Slightly larger points for individual plots
            alpha=0.3,
            label=class_name
        )

        # Add labels and title
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'2D PCA of Embeddings - Class: {class_name}')
        
        # Add a legend
        plt.legend(markerscale=20)  # Increase markerscale for visibility

        # Improve the layout
        plt.tight_layout()

        # Save the plot
        save_path = os.path.join(output_directory, f"{manifold_decomposition}_visualization_{class_name}_no_na.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free up memory
        print(f"PCA visualization for class {class_name} saved to {save_path}")

def visualize_2d(pca_result, predictions, save_path=None):
    # Get unique classes and assign colors
    unique_classes = list(set(predictions))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
    color_dict = dict(zip(unique_classes, colors))

    # Create a 2D scatter plot
    plt.figure(figsize=(12, 10))
    for class_name in unique_classes:
        mask = np.array(predictions) == class_name
        plt.scatter(
            pca_result[mask, 0], 
            pca_result[mask, 1], 
            c=[color_dict[class_name]],
            s=0.1,  # Keep the actual plot points small
            alpha=0.1,
            label=class_name
        )

    # Add labels and title
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'2D PCA of Embeddings (First {len(pca_result)} samples)')
    
    # Create a custom legend with larger points
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  label=class_name, 
                                  markerfacecolor=color_dict[class_name], 
                                  markersize=10)  # Increase this value for larger legend points
                       for class_name in unique_classes]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    # Improve the layout
    plt.tight_layout()

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA visualization saved to {save_path}")

def main(embedding_directory, parquet_directory, output_directory, manifold_decomposition, visualize_per_class, n_samples=4_000_000):
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    output_path = os.path.join(output_directory, f"{manifold_decomposition}_output.npy")

    if os.path.exists(output_path):
        print(f"Loading existing {manifold_decomposition} output...")
        result = np.load(output_path)
    else: 
        print(f"Loading first {n_samples} embeddings...")
        all_embeddings = load_first_n_samples(embedding_directory, n_samples)
        print(f"Performing {manifold_decomposition} on {len(all_embeddings)} embeddings...")

        if manifold_decomposition == 'umap':
            result = perform_umap(all_embeddings)
        elif manifold_decomposition == 'pca':
            result = perform_pca(all_embeddings)

        np.save(output_path, result)
        print(f"{manifold_decomposition} output saved to {output_path}")

    print(f"Loading first {n_samples} predictions...")
    predictions = load_first_n_predictions(parquet_directory, n_samples)

    assert len(result) == len(predictions), f"Mismatch between {manifold_decomposition} results and predictions count"

    save_path = os.path.join(output_directory, f"{manifold_decomposition}_visualization_{len(result)}_samples.png")
    visualize_2d(result, predictions, save_path)

    if visualize_per_class:
        visualize_pca_2d_per_class(result, predictions, output_directory, manifold_decomposition)

if __name__ == "__main__":
    embedding_directory = "/mnt/sets/embeddings-pdf-corpus-urls/"
    parquet_directory = "/mnt/sets/parquet_pdf/"
    output_directory = "./classes"
    n_samples = 6_500_000
    visualize_per_class = True
    manifold_decomposition = 'umap' # 'pca'
    main(embedding_directory, parquet_directory, output_directory, manifold_decomposition, visualize_per_class, n_samples)
