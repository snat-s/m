import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from scipy.sparse import vstack, csr_matrix

def load_and_vectorize_data(file_path, min_instances=100):
    print("Loading and vectorizing data...")
    df = pd.read_csv(file_path)
    
    # Filter classes with at least min_instances
    class_counts = df['classification'].value_counts()
    valid_classes = class_counts[class_counts >= min_instances].index
    df_filtered = df[df['classification'].isin(valid_classes)]
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_filtered['url'])
    y = df_filtered['classification'].reset_index(drop=True)
    
    return X, y, vectorizer

def balance_dataset_for_class(X, y, class_name):
    # Get indices for the current class
    class_indices = y[y == class_name].index
    
    # Get indices for all other classes
    other_indices = y[y != class_name].index
    
    # Randomly sample from other classes to match the number of current class
    sampled_other_indices = np.random.choice(other_indices, size=len(class_indices), replace=False)
    
    # Combine indices and sort them to maintain the original order
    balanced_indices = np.sort(np.concatenate([class_indices, sampled_other_indices]))
    
    return X[balanced_indices], y.loc[balanced_indices].reset_index(drop=True)

def plot_confusion_matrix(cm, class_name):
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {class_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    os.makedirs('./confusion_matrices', exist_ok=True)
    plt.savefig(f'./confusion_matrices/{class_name}_confusion_matrix.png')
    plt.close()

def train_models(X, y, unique_classes):
    classifiers = {}
    performance_metrics = []
    for class_name in tqdm(unique_classes):
        # Balance the dataset for this class
        X_balanced, y_balanced = balance_dataset_for_class(X, y, class_name)
        
        # Create binary target: 1 if the sample belongs to the current class, 0 otherwise
        y_binary = (y_balanced == class_name).astype(int)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_binary, test_size=0.2, random_state=42)
        
        # model = MultinomialNB(random_state=42,  n_jobs=-1)
        model = LogisticRegression(random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store performance metrics
        performance_metrics.append({
            'class': class_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'count': len(y_pred)
        })
        
        # Compute and plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, class_name)
        
        # Store the classifier
        classifiers[class_name] = model
    
    return classifiers, performance_metrics

def save_models(classifiers, vectorizer):
    print("Saving models and classes...")
    os.makedirs('./models', exist_ok=True)
    for class_name, model in classifiers.items():
        joblib.dump(model, f'./models/logistic_regression_classifier_{class_name}.joblib')
    with open('./models/classifier_classes.json', 'w') as f:
        json.dump(list(classifiers.keys()), f)
    # Save the vectorizer
    print("Saving vectorizer...")
    joblib.dump(vectorizer, './models/url_vectorizer.joblib')
    print("Training and saving completed.")

def print_performance_summary(performance_metrics):
    df = pd.DataFrame(performance_metrics)
    print("\nPerformance Summary:")
    print(df)
    print("\nAverage Performance:")
    print(df[['accuracy', 'precision', 'recall', 'f1']].mean())

def main(file_path, min_instances=100):
    X, y, vectorizer = load_and_vectorize_data(file_path, min_instances)
    unique_classes = y.unique()
    classifiers, performance_metrics = train_models(X, y, unique_classes)
    print_performance_summary(performance_metrics)
    save_models(classifiers, vectorizer)

if __name__ == "__main__":
    file_path = '../data/equally_distributed.csv'  # Update this to your new file path
    min_instances = 100
    main(file_path, min_instances)
