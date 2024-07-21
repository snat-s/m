import argparse
import pandas as pd
from tqdm import trange
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def generate_embeddings(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Initialize the model
    model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
    model = model.to('cuda')
    model.max_length = 4096

    embeddings = []
    batch_size = 1
    save_threshold = 1_000

    for i in trange(0, len(df), batch_size):
        batch_sentences = df['url'][i:i+batch_size].to_list()
        batch_embeddings = model.encode(batch_sentences)
        embeddings.append(batch_embeddings)
        
        if len(embeddings) * batch_size >= save_threshold:
            # Concatenate the embeddings
            embeddings = np.concatenate(embeddings, axis=0)
            file_name = f'./embeddings/{i}_embeddings.npz'
            np.savez_compressed(file_name, embeddings=embeddings)
            
            del embeddings
            embeddings = []

    if embeddings is not None:
        embeddings = np.concatenate(embeddings, axis=0)
        file_name = f'./embeddings/final_embeddings.npz'
        np.savez_compressed(file_name, embeddings=embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings from URL information in a CSV file.")
    parser.add_argument("--csv_path", help="Path to the CSV file containing URL information")
    args = parser.parse_args()

    generate_embeddings(args.csv_path)
