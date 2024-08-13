import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import glob
import json
from tqdm import tqdm
import optuna

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

def objective(trial, X_train, X_test, y_train, y_test):
    param = {
        "objective": "binary:logistic",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'device': 'cuda',
        #'early_stopping_rounds': 250,
    }

    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    return f1

def optimize_hyperparameters(X_train, X_test, y_train, y_test, n_trials=100):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=n_trials)

    return study.best_params

def print_and_save_performance_summary(performance_metrics, output_file):
    df = pd.DataFrame(performance_metrics)
    summary = "\nPerformance Summary:\n"
    summary += df[['class', 'accuracy', 'precision', 'recall', 'f1', 'count_pred']].to_string()
    summary += "\n\nAverage Performance:\n"
    summary += df[['accuracy', 'precision', 'recall', 'f1']].mean().to_string()
    summary += "\n\nBest Parameters for Each Class:"
    for metric in performance_metrics:
        summary += f"\n\nClass: {metric['class']}"
        summary += f"\nBest Parameters: {json.dumps(metric['best_params'], indent=2)}"
    
    print(summary)
    
    with open(output_file, 'w') as f:
        f.write(summary)

# Load the batched data
print("Loading batched data...")
embeddings, classifications = load_batched_data('/mnt/sets/embeddings_train/')

# Get unique classes
unique_classes = np.unique(classifications)

# Dictionary to store classifiers and performance metrics
classifiers = {}
performance_metrics = []
best_params_dict = {}  # Dictionary to store best parameters for each class

for class_name in tqdm(unique_classes):
    print(f"Training classifier for class: {class_name}")

    balanced_embeddings, balanced_labels = create_balanced_dataset_for_label(embeddings, classifications, class_name)
    X_train, X_test, y_train, y_test = train_test_split(balanced_embeddings, balanced_labels, test_size=0.2, random_state=42)

    print("Optimizing hyperparameters...")
    best_params = optimize_hyperparameters(X_train, X_test, y_train, y_test)

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
        'count_pred': len(y_pred),
        'best_params': best_params
    })

    classifiers[class_name] = model
    model.save_model(f'./models/xgboost_classifier_{class_name}.json')

    # Store best parameters for this class
    best_params_dict[class_name] = best_params

# Save classifier classes
with open('./models/classifier_classes.json', 'w') as f:
    json.dump(list(classifiers.keys()), f)

# Save best parameters for all classes
with open('./models/best_params.json', 'w') as f:
    json.dump(best_params_dict, f, indent=4)

output_file = './models/performance_summary.txt'
print_and_save_performance_summary(performance_metrics, output_file)

print(f"\nPerformance summary has been saved to '{output_file}'")
print("\nBest parameters have been saved to './models/best_params.json'")
