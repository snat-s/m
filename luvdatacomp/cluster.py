import os
import gc
import json
import numpy as np
import polars as pl
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_files(path, extension):
    return [os.path.join(path, file) for file in os.listdir(path) if file.endswith(extension)]

def cluster_partial_fit(files, n_clusters=128, batch_size=1024):
    kmeans = MiniBatchKMeans(
        init='k-means++',
        n_clusters=n_clusters,
        batch_size=batch_size,
        n_init="auto"
    )
    
    for file in tqdm(files, desc="Clustering"):
        with np.load(file, mmap_mode='r') as data:
            embeddings_txt = data['l14_txt']
            embeddings_img = data['l14_img']
            
            for i in range(0, len(embeddings_txt), batch_size):
                batch_txt = embeddings_txt[i:i+batch_size]
                batch_img = embeddings_img[i:i+batch_size]
                batch = np.concatenate((batch_txt, batch_img), axis=1)
                kmeans.partial_fit(batch)
    
    return kmeans

def process_file(args):
    npz_file, parquet_file, cluster_centers, batch_size = args
    all_distances = []
    
    with np.load(npz_file, mmap_mode='r') as data:
        embeddings_txt = data['l14_txt']
        embeddings_img = data['l14_img']
        total_samples = len(embeddings_txt)
        
        df = pl.read_parquet(parquet_file)
        uids = df['uid'].to_list()
        
        for i in range(0, total_samples, batch_size):
            batch_txt = embeddings_txt[i:i+batch_size]
            batch_img = embeddings_img[i:i+batch_size]
            batch = np.concatenate((batch_txt, batch_img), axis=1)
            
            distances = cdist(batch, cluster_centers)
            min_distances = np.min(distances, axis=1)
            all_distances.extend(list(zip(min_distances, uids[i:i+batch_size])))
    
    return all_distances

def self_supervised_prototypes_pruning(kmeans, files, parquet_files, batch_size=1024):
    args = [(npz_file, parquet_file, kmeans.cluster_centers_, batch_size) 
            for npz_file, parquet_file in zip(files, parquet_files)]
    
    all_distances = []
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, arg) for arg in args]
        
        for future in tqdm(as_completed(futures), total=len(files), desc="Computing distances"):
            distances = future.result()
            all_distances.extend(distances)
    
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

def cluster(path, output_classification_path, n_clusters=1300, batch_size=1024):
    embeddings_files = get_files(path, "npz")
    metadata_files = get_files(path, "parquet") 

    embeddings_files.sort()
    metadata_files.sort()
    
    kmeans = cluster_partial_fit(embeddings_files, n_clusters=n_clusters, batch_size=batch_size)
    
    all_distances = self_supervised_prototypes_pruning(kmeans, embeddings_files, metadata_files, batch_size)

    fractions = np.arange(0.1, 1.0, 0.1)
    save_pruned_datasets(all_distances, fractions, output_classification_path)

if __name__ == "__main__":
    PATH = "/mnt/data/small_datacomp/metadata/"
    OUTPUT_CLASSIFICATION_PATH = "breaking_the_laws"
    N_CLUSTERS = 1300
    BATCH_SIZE = 1024
    
    cluster(PATH, OUTPUT_CLASSIFICATION_PATH, N_CLUSTERS, BATCH_SIZE)
