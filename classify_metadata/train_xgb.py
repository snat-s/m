"""
Train an xgbooost classifier based on embeddings
"""
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
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

# Load the batched data
print("Loading batched data...")
embeddings, classifications = load_batched_data('/mnt/sets/embeddings/')

# Get unique classes
unique_classes = np.unique(classifications)

# Dictionary to store classifiers
classifiers = {}

for class_name in unique_classes:
    print(f"Training classifier for class: {class_name}")
    
    # Create binary target: 1 if the sample belongs to the current class, 0 otherwise
    y = (classifications == class_name).astype(int)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Set XGBoost parameters
    params = {
        'max_depth': 6,
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
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print()
    
    # Store the classifier
    classifiers[class_name] = model

    for class_name, model in classifiers.items():
        model.save_model(f'./models/xgboost_classifier_{class_name}.json')
    
    with open('./models/classifier_classes.json', 'w') as f:
        json.dump(list(classifiers.keys()), f)
