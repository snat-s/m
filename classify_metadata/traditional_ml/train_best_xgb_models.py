"""
I saved the hyperparameters but forgot to save the accuracy
so this is exactly for just retraining the best models.
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

    n_samples = min(len(positive_samples), len(negative_samples))

    if len(positive_samples) > len(negative_samples):
        positive_samples = resample(positive_samples, n_samples=n_samples, random_state=42)
    else:
        negative_samples = resample(negative_samples, n_samples=n_samples, random_state=42)

    balanced_embeddings = np.vstack((positive_samples, negative_samples))
    balanced_labels = np.hstack((np.ones(n_samples), np.zeros(n_samples)))

    return balanced_embeddings, balanced_labels

def print_and_save_performance_summary(performance_metrics, output_file):
    df = pd.DataFrame(performance_metrics)
    summary = "\nPerformance Summary:\n"
    summary += df[['class', 'accuracy', 'precision', 'recall', 'f1', 'count_pred']].to_string()
    summary += "\n\nAverage Performance:\n"
    summary += df[['accuracy', 'precision', 'recall', 'f1']].mean().to_string()
    
    print(summary)
    
    with open(output_file, 'w') as f:
        f.write(summary)

# Load the batched data
print("Loading batched data...")
embeddings, classifications = load_batched_data('/mnt/sets/embeddings_train/')

# Get unique classes
unique_classes = np.unique(classifications)

# Load best parameters
try:
    with open('./models/best_params.json', 'r') as f:
        best_params_dict = json.load(f)
    print("Loaded existing best parameters.")
except FileNotFoundError:
    print("best_params.json not found. Please run the hyperparameter optimization first.")
    exit(1)

# Dictionary to store classifiers and performance metrics
classifiers = {}
performance_metrics = []

for class_name in tqdm(unique_classes):
    print(f"Training classifier for class: {class_name}")

    balanced_embeddings, balanced_labels = create_balanced_dataset_for_label(embeddings, classifications, class_name)
    X_train, X_test, y_train, y_test = train_test_split(balanced_embeddings, balanced_labels, test_size=0.2, random_state=42)

    best_params = best_params_dict.get(class_name, {})
    if not best_params:
        print(f"Warning: No best parameters found for class {class_name}. Using default parameters.")

    print("Training model with best parameters...")
    model = xgb.XGBClassifier(**best_params, device='cuda')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    performance_metrics.append({
        'class': class_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'count_pred': len(y_pred)
    })

    classifiers[class_name] = model
    model.save_model(f'./models/xgboost_classifier_{class_name}.json')

# Save classifier classes
with open('./models/classifier_classes.json', 'w') as f:
    json.dump(list(classifiers.keys()), f)

output_file = './models/performance_summary.txt'
print_and_save_performance_summary(performance_metrics, output_file)

print(f"\nPerformance summary has been saved to '{output_file}'")
