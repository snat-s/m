import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import glob
from tqdm import tqdm
import os
import pyarrow.parquet as pq
from collections import Counter
import pandas as pd
import umap
import random
import colorsys

colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
]

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

def visualize_2d_individual_classes(pca_result, predictions, output_directory, manifold_decomposition):
    valid_mask = np.array(predictions) != 'NA' 
    pca_result = pca_result[valid_mask]
    predictions = np.array(predictions)[valid_mask]
    
    # Get unique classes and assign colors
    unique_classes = sorted(list(set(predictions)))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
    color_dict = dict(zip(unique_classes, colors))
    
    # Create a subdirectory for individual class plots
    individual_plots_dir = os.path.join(output_directory, f"{manifold_decomposition}_individual_plots")
    os.makedirs(individual_plots_dir, exist_ok=True)
    
    for class_name in unique_classes:
        plt.figure(figsize=(10, 8))
        mask = np.array(predictions) == class_name
        plt.scatter(
            pca_result[mask, 0], 
            pca_result[mask, 1], 
            c=[color_dict[class_name]],
            s=0.01,
            alpha=0.01,
            label=class_name
        )
        plt.title(f'2D {manifold_decomposition} of Embeddings - Class: {class_name}')
        plt.legend(markerscale=20, loc='upper right')
        plt.xticks([])
        plt.yticks([])
        
        # Improve the layout
        plt.tight_layout()
        
        # Save the individual plot
        save_path = os.path.join(individual_plots_dir, f"{manifold_decomposition}_visualization_{class_name}_no_na.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{manifold_decomposition} visualization for class {class_name} saved to {save_path}")

def visualize_2d_per_class(pca_result, predictions, output_directory, manifold_decomposition):
    valid_mask = np.array(predictions) != 'NA' 
    pca_result = pca_result[valid_mask]
    predictions = np.array(predictions)[valid_mask]
    
    # Get unique classes and assign colors
    unique_classes = sorted(list(set(predictions)))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
    color_dict = dict(zip(unique_classes, colors))
    
    # Calculate grid dimensions
    n_classes = len(unique_classes)
    n_cols = int(np.ceil(np.sqrt(n_classes)))
    n_rows = int(np.ceil(n_classes / n_cols))
    
    # Create a single figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    fig.suptitle(f'2D {manifold_decomposition} of Embeddings by Class', fontsize=16)
    
    # Flatten axes array for easier indexing
    axes = axes.flatten()
    
    for i, class_name in enumerate(unique_classes):
        ax = axes[i]
        mask = np.array(predictions) == class_name
        ax.scatter(
            pca_result[mask, 0], 
            pca_result[mask, 1], 
            c=[color_dict[class_name]],
            s=0.01,
            alpha=0.01,
            label=class_name
        )
        ax.set_title(f'Class: {class_name}')
        ax.legend(markerscale=20, loc='upper right')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Remove any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    # Improve the layout
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(output_directory, f"{manifold_decomposition}_visualization_all_classes_no_na.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{manifold_decomposition} visualization for all classes saved to {save_path}")
    visualize_2d_individual_classes(pca_result, predictions, output_directory, manifold_decomposition)

def plot_pie(predictions, save_path):
    # Count the occurrences of each prediction using Counter
    prediction_counts = Counter(predictions)
    
    # Since we know there are exactly 17 classes, we can use all of them
    labels = list(prediction_counts.keys())
    sizes = list(prediction_counts.values())
    
    # Create a pie chart
    plt.figure(figsize=(12, 8))
    plt.pie(sizes, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%', pctdistance=0.85, startangle=45)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Distribution of 17 Prediction Classes')
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()
    print(labels,sum(sizes))
    print(f"Pie chart saved to {save_path}")

def visualize_2d_plus_classes(pca_result, predictions, manifold_decomposition, save_path=None):
    # Get unique classes and assign colors
    unique_classes = sorted(list(set(predictions)))
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
            s=.01,              
            alpha=0.1,
            label=class_name
        )

    plt.title(f'2D {manifold_decomposition} of Embeddings (First {len(pca_result)} samples)')
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  label=class_name, 
                                  markerfacecolor=color_dict[class_name], 
                                  markersize=10)  # Increase this value for larger legend points
                       for class_name in unique_classes]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{manifold_decomposition} visualization saved to {save_path}")

def visualize_2d(result, manifold_decomposition, save_path=None):
    plt.figure(figsize=(12, 10))
    plt.scatter(
        result[:, 0],
        result[:, 1],
        c='black',  # Single color for all points
        s=0.01,    # Small point size
        alpha=0.1  # Low alpha for transparency
    )

    plt.title(f'2D {manifold_decomposition} of Embeddings (First {len(result)} samples)')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{manifold_decomposition} visualization saved to {save_path}")
    plt.close()

def visualize_3d(pca_result, predictions, manifold_decomposition, save_path=None):
    # Get unique classes and assign colors
    unique_classes = sorted(list(set(predictions)))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
    color_dict = dict(zip(unique_classes, colors))

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for class_name in unique_classes:
        mask = np.array(predictions) == class_name
        ax.scatter(
            pca_result[mask, 0], 
            pca_result[mask, 1], 
            pca_result[mask, 2],
            c=[color_dict[class_name]],
            s=1,              
            alpha=0.1,
            label=class_name
        )

    ax.set_title(f'3D {manifold_decomposition} of Embeddings (First {len(pca_result)} samples)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  label=class_name, 
                                  markerfacecolor=color_dict[class_name], 
                                  markersize=10)
                       for class_name in unique_classes]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D {manifold_decomposition} visualization saved to {save_path}")
    plt.close()

def main(embedding_directory, parquet_directory, output_directory, manifold_decomposition, visualize_per_class, should_plot_pie, n_samples=4_000_000):
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

    save_path = os.path.join(output_directory, f"{manifold_decomposition}_visualization_{len(result)}_no_classes.png")
    visualize_2d(result, manifold_decomposition, save_path)
    save_path = os.path.join(output_directory, f"{manifold_decomposition}_visualization_{len(result)}_samples.png")
    visualize_2d_plus_classes(result, predictions, manifold_decomposition, save_path)

    if visualize_per_class:
        visualize_2d_per_class(result, predictions, output_directory, manifold_decomposition)

    if should_plot_pie:
        save_path = os.path.join(output_directory, f"predictions_{len(result)}_pie.png")
        plot_pie(predictions, save_path)


if __name__ == "__main__":
    embedding_directory = "/mnt/sets/embeddings-pdf-corpus-urls/"
    parquet_directory = "/mnt/sets/parquet_pdf/"
    output_directory = "./classes"
    n_samples = 8_500_000 #6_500_000 #9_000_000 
    visualize_per_class = False
    should_plot_pie = True
    manifold_decomposition = 'pca' # 'pca' # 'umap' 
    main(embedding_directory, parquet_directory, output_directory, manifold_decomposition, visualize_per_class, should_plot_pie, n_samples)
