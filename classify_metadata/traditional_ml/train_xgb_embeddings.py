"""
Train an xgbooost classifier based on embeddings.
"""
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import glob
import json
from tqdm import tqdm

def load_batched_data(directory):
    all_embeddings = []
    all_classifications = []
    for file in tqdm(glob.glob(f"{directory}/*_embeddings_with_classifications.npz")):
        with np.load(file) as data:
            all_embeddings.append(data['embeddings'])
            all_classifications.append(data['classifications'])
    return np.concatenate(all_embeddings), np.concatenate(all_classifications)

def create_balanced_dataset_for_label(embeddings, classifications, label):
    positive_samples = embeddings[classifications == label]
    negative_samples = embeddings[classifications != label]
    
    # Determine the number of samples to use (minimum of positive and negative samples)
    n_samples = min(len(positive_samples), len(negative_samples))
    
    # Undersample the majority class
    if len(positive_samples) > len(negative_samples):
        positive_samples = resample(positive_samples, n_samples=n_samples, random_state=42)
    else:
        negative_samples = resample(negative_samples, n_samples=n_samples, random_state=42)
    
    # Combine the balanced datasets
    balanced_embeddings = np.vstack((positive_samples, negative_samples))
    balanced_labels = np.hstack((np.ones(n_samples), np.zeros(n_samples)))
    
    return balanced_embeddings, balanced_labels

# Load the batched data
print("Loading batched data...")
embeddings, classifications = load_batched_data('/mnt/sets/embeddings_train/')

# Get unique classes
unique_classes = np.unique(classifications)

# Dictionary to store classifiers
classifiers = {}
performance_metrics = []

for class_name in tqdm(unique_classes):
    print(f"Training classifier for class: {class_name}")
    
    balanced_embeddings, balanced_labels = create_balanced_dataset_for_label(embeddings, classifications, class_name)


    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(balanced_embeddings, balanced_labels, test_size=0.2, random_state=42)
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Set XGBoost parameters
    params = {
        #'max_depth': 6,
        'eta': 0.3,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'device': 'cuda'
    }
    num_round = 100
    model = xgb.train(params, dtrain, num_round)
    
    # Make predictions
    y_pred = model.predict(dtest)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)

    performance_metrics.append({
        'class': class_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'count_pred': len(y_pred)
    })
    
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    # print()
    
    # Store the classifier
    classifiers[class_name] = model

    for class_name, model in classifiers.items():
        model.save_model(f'./models/xgboost_classifier_{class_name}.json')
    
    with open('./models/classifier_classes.json', 'w') as f:
        json.dump(list(classifiers.keys()), f)

def print_performance_summary(performance_metrics):
    df = pd.DataFrame(performance_metrics)
    print("\nPerformance Summary:")
    print(df)
    print("\nAverage Performance:")
    print(df[['accuracy', 'precision', 'recall', 'f1']].mean())

print_performance_summary(performance_metrics)
