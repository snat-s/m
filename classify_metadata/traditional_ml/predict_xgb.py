"""
Predicting the labels for the embeddings using xgboost.
"""
import numpy as np
import xgboost as xgb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import glob
from tqdm import tqdm
import os

def predict_probabilities(embedding, class_names, models):
    """
    Predict probabilities for each class given an embedding.
    
    :param embedding: numpy array of shape (n_features,) or (n_samples, n_features)
    :param class_names: list of class names
    :param models: dict of loaded XGBoost models
    :return: dict of class probabilities
    """
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    
    dmatrix = xgb.DMatrix(embedding)
    
    probabilities = {}
    for class_name in class_names:
        probabilities[class_name] = models[class_name].predict(dmatrix)
    
    return probabilities

def load_batched_data(directory, batch_size=500_000):
    batch = []
    for file in tqdm(glob.glob(f"{directory}/*_embeddings.npz")):
        with np.load(file) as data:
            embeddings = data['embeddings']
            batch.extend(embeddings)
            while len(batch) >= batch_size:
                yield np.array(batch[:batch_size])
                batch = batch[batch_size:]
    if batch:  # Yield any remaining embeddings
        yield np.array(batch)

def save_to_parquet(data, output_file):
    """
    Save data to a Parquet file.
    
    :param data: dict containing 'urls', 'embeddings', and 'predictions'
    :param output_file: path to save the Parquet file
    """
    table = pa.Table.from_pydict(data)
    pq.write_table(table, output_file)
    print(f"Saved data to {output_file}")

def get_class_prediction(probs, class_names):
    max_prob_index = np.argmax(probs)
    max_prob = probs[max_prob_index]
    return class_names[max_prob_index]

def main():
    with open('./models/classifier_classes.json', 'r') as f:
        class_names = json.load(f)
    
    models = {}
    for class_name in class_names:
        model = xgb.Booster()
        model.load_model(f'./models/xgboost_classifier_{class_name}.json')
        models[class_name] = model
    
    df = pd.read_csv('../data/cc-provenance-20230303.csv', low_memory=False)
    urls = df['url'].tolist()
    
    output_dir = '/mnt/sets/parquet_pdf/'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Processing batched data...")
    for i, batch in enumerate(load_batched_data('/mnt/sets/embeddings-pdf-corpus-urls/')):
        print(f"Processing batch {i+1} with {len(batch)} embeddings")
        predictions = predict_probabilities(batch, class_names, models)
        
        start_idx = i * len(predictions)
        end_idx = start_idx + len(batch)
        batch_urls = urls[start_idx:end_idx]
        
        data = {
            'url': batch_urls,
            'prediction': [
                get_class_prediction(probs, class_names) 
                for probs in zip(*predictions.values())
            ]
        }
        
        output_file = os.path.join(output_dir, f'output_batch_{i+1}.parquet')
        save_to_parquet(data, output_file)
    
    print("Finished processing")

if __name__ == "__main__":
    main()
