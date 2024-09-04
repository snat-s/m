import os
import gc
import json
import heapq
import multiprocessing

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_files(path, extension):
    return [path+file for file in os.listdir(path) if file.endswith(extension)] 

def load_files(files):
    embeddings = []
    for file in tqdm(files):
        data = np.load(file)
        embed  = data['l14_txt']
        embeddings.append(embed)
        del data
        gc.collect()
    embeddings = np.vstack(embeddings)
    return embeddings

def cluster_partial_fit(files, n_clusters=128):

    batch_size = max(1024, 256 * multiprocessing.cpu_count())
    kmeans = MiniBatchKMeans(
            init='k-means++',
            n_clusters=n_clusters,
            batch_size=batch_size,
            n_init="auto"
    )
    
    for file in tqdm(files, desc="Clustering"):
        data = np.load(file)
        #embeddings = data['l14_txt']
        embeddings_txt = data['l14_txt']
        embeddings_img = data['l14_img']
        embeddings = np.concatenate((embeddings_txt, embeddings_img), axis=1)
        
        kmeans.partial_fit(embeddings)
        
        del data, embeddings
        gc.collect()
    
    #print(f"{kmeans.inertia_=}")
    return kmeans

def sample_from_clusters(kmeans, files, parquet_files, output_classification_path, n_samples=5):
    all_closest_indices = [[] for _ in range(kmeans.n_clusters)]
    all_furthest_indices = [[] for _ in range(kmeans.n_clusters)]
    
    all_data_points = []
    all_cluster_labels = []
    all_uids = []

    for npz_file, parquet_file in tqdm(zip(files, parquet_files), desc="Sampling", total=len(files)):
        data = np.load(npz_file)
        embeddings = data['l14_txt']
        
        df = pl.read_parquet(parquet_file)
        
        distances = cdist(embeddings, kmeans.cluster_centers_)
        labels = kmeans.predict(embeddings)
        
        all_data_points.extend(df['text'].to_list())
        all_uids.extend(df['uid'].to_list())
        all_cluster_labels.extend(labels)
        
        for i in range(kmeans.n_clusters):
            cluster_distances = distances[:, i]
            cluster_mask = labels == i
            
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) == 0:
                continue
            
            cluster_distances = cluster_distances[cluster_mask]
            
            closest = cluster_indices[np.argsort(cluster_distances)[:n_samples]]
            furthest = cluster_indices[np.argsort(cluster_distances)[-n_samples:]]
            
            all_closest_indices[i].extend(df['text'][closest].to_list())
            all_furthest_indices[i].extend(df['text'][furthest].to_list())
        
        del data, embeddings, df
        gc.collect()

    df_labels = pl.DataFrame({
        "uid": all_uids,
        "data_points": all_data_points,
        "cluster_label": all_cluster_labels
    })

    df_labels.write_csv(output_classification_path)
    
    return all_closest_indices, all_furthest_indices

def find_correct_number_clusters(embeddings_files):
    range_n_clusters = list(range(2000, 5001, 500))  
    batch_size = max(1024, 256 * multiprocessing.cpu_count())
    results = []

    for n_clusters in tqdm(range_n_clusters, desc="Grid search for clusters"):
        kmeans = MiniBatchKMeans(
            init='k-means++',
            n_clusters=n_clusters,
            batch_size=batch_size,
            n_init="auto"
        )
        
        for file in tqdm(embeddings_files, desc=f"Clustering with {n_clusters} clusters"):
            data = np.load(file)
            embeddings = data['l14_txt']
            normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            kmeans.partial_fit(embeddings)
            del data, embeddings
            gc.collect()
        
        results.append({
            'n_clusters': n_clusters,
            'inertia': kmeans.inertia_
        })
        
        print(f"For n_clusters = {n_clusters}, the inertia is: {kmeans.inertia_}")
    
    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot([r['n_clusters'] for r in results], [r['inertia'] for r in results], 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.savefig('elbow_plot.png')
    plt.close()

    # Save results to a JSON file
    with open('clustering_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Elbow plot has been saved as elbow_plot.png")
    print("Clustering results have been saved to clustering_results.json")
    
    return results

def process_file(args):
    npz_file, parquet_file, cluster_centers = args
    data = np.load(npz_file)
    embeddings_txt = data['l14_txt']
    embeddings_img = data['l14_img']
    embeddings = np.concatenate((embeddings_txt, embeddings_img), axis=1)
    print(embeddings_txt.shape, embeddings_img, embeddings.shape)
    df = pl.read_parquet(parquet_file)
    uids = df['uid'].to_list()
    
    # Process in smaller chunks to reduce memory usage
    chunk_size = 50
    all_distances = []
    
    for i in trange(0, len(embeddings), chunk_size):
        chunk = embeddings[i:i+chunk_size]
        distances = cdist(chunk, cluster_centers)
        min_distances = np.min(distances, axis=1)
        all_distances.extend(list(zip(min_distances, uids[i:i+chunk_size])))
    
    del data, embeddings, df
    gc.collect()
    
    return all_distances

def self_supervised_prototypes_pruning(kmeans, files, parquet_files, num_clusters=128):
    # From: https://arxiv.org/pdf/2206.14486
    args = [(npz_file, parquet_file, kmeans.cluster_centers_) 
            for npz_file, parquet_file in zip(files, parquet_files)]
    
    all_distances = []
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, arg) for arg in args]
        
        for future in tqdm(as_completed(futures), total=len(files), desc="Computing distances"):
            distances = future.result()
            all_distances.extend(distances)
    
    # Sort distances in descending order (hardest examples first)
    all_distances.sort(reverse=True)
    
    return all_distances

def save_pruned_datasets(all_distances, fractions, output_dir):
    total_samples = len(all_distances)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for fraction in tqdm(fractions, desc="Saving pruned datasets"):
        num_to_keep = int(total_samples * fraction)
        pruned_data = all_distances[:num_to_keep]
        
        pruned_uids = [uid for _, uid in pruned_data]
        pruned_distances = [dist for dist, _ in pruned_data]
        
        output_path = os.path.join(output_dir, f"breaking_laws_{fraction:.1f}.csv")
        df_pruned = pl.DataFrame({
            "uid": pruned_uids,
            "distance": pruned_distances
        })
        df_pruned.write_csv(output_path)

def cluster(path, output_classification_path, output_sample_path, is_self_supervised_prototypes_pruning=False):
    embeddings_files = get_files(path, "npz")
    metadata_files = get_files(path, "parquet") 

    embeddings_files.sort()
    metadata_files.sort()
    #results = find_correct_number_clusters(embeddings_files)
    kmeans = cluster_partial_fit(embeddings_files, n_clusters=1300)

    if not is_self_supervised_prototypes_pruning:
        closest_samples, furthest_samples = sample_from_clusters(kmeans, embeddings_files, metadata_files, output_classification_path)

        clusters_data = {}
        for i in range(kmeans.n_clusters):
            clusters_data[f"Cluster_{i}"] = {
                    "closest_samples": closest_samples[i][:5],
                    "furthest_samples": furthest_samples[i][:5]
            }

        with open(output_sample_path, 'w', encoding='utf-8') as f:
            json.dump(clusters_data, f, ensure_ascii=False, indent=2)

        print(f"Cluster samples have been saved to {output_sample_path}")
    else:
        all_distances = self_supervised_prototypes_pruning(kmeans, embeddings_files, metadata_files, 1300)

        fractions = np.arange(0.1, 1.0, 0.1)
        save_pruned_datasets(all_distances, fractions, output_classification_path)

if __name__ == "__main__":
    PATH = "/mnt/data/small_datacomp/metadata/"
    OUTPUT_SAMPLE_PATH = "./data/cluster_samples.json"
    OUTPUT_CLASSIFICATION_PATH = "breaking_the_laws" #"./data/cluster_labels.csv"
    IS_SELF_SUPERVISED_PROTOTYPES_PRUNING = True
    cluster(PATH, OUTPUT_CLASSIFICATION_PATH, OUTPUT_SAMPLE_PATH, IS_SELF_SUPERVISED_PROTOTYPES_PRUNING)
