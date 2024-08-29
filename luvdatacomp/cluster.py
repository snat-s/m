import os
import gc
import json
import multiprocessing

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

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
        embeddings = data['l14_txt']
        
        kmeans.partial_fit(embeddings)
        
        del data, embeddings
        gc.collect()
    

    return kmeans

def sample_from_clusters(kmeans, files, parquet_files, n_samples=5):
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
    df_labels.write_csv("./data/cluster_labels.csv")
    
    return all_closest_indices, all_furthest_indices

def find_correct_number_clusters(embeddings_files):
    range_n_clusters = list(range(100, 2001, 100))  # From 100 to 2000 in steps of 100
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

if __name__ == "__main__":
    PATH = "/mnt/sets/small_datacomp/metadata/"
    embeddings_files = get_files(PATH, "npz")
    metadata_files = get_files(PATH, "parquet") 

    embeddings_files.sort()
    metadata_files.sort()
    #results = find_correct_number_clusters(embeddings_files)
    kmeans = cluster_partial_fit(embeddings_files, n_clusters=1300)
    closest_samples, furthest_samples = sample_from_clusters(kmeans, embeddings_files, metadata_files)
    
    clusters_data = {}
    for i in range(kmeans.n_clusters):
        clusters_data[f"Cluster_{i}"] = {
            "closest_samples": closest_samples[i][:5],
            "furthest_samples": furthest_samples[i][:5]
        }
    
    with open('./data/cluster_samples.json', 'w', encoding='utf-8') as f:
        json.dump(clusters_data, f, ensure_ascii=False, indent=2)
    
    print("Cluster samples have been saved to cluster_samples.json")
