```python
#!pip install evaluate
```


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
)
from datasets import Dataset, load_dataset
import evaluate
import os
from sklearn.metrics import classification_report, confusion_matrix
import wandb
import torch
```


```python
# torch.cuda_set_device(1)
torch.cuda.is_available()
```


```python
# Configuration
BASE_MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"
DATASET_PATH = "./equally_distributed59k.csv"
CHECKPOINT_DIR = "./checkpoints"+BASE_MODEL_NAME
THRESHOLD = 250  # Minimum number of samples per class

# Load and preprocess the data
df = pd.read_csv(DATASET_PATH)
value_counts = df['classification'].value_counts()
classes_to_keep = value_counts[value_counts >= THRESHOLD].index.tolist()
df = df[df['classification'].isin(classes_to_keep)]

# Create label mappings
class_to_label = {cls: i for i, cls in enumerate(classes_to_keep)}
label_to_class = {i: cls for cls, i in class_to_label.items()}
```


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

# Assuming you have already loaded your DataFrame 'df' and defined 'class_to_label' and 'BASE_MODEL_NAME'

# Prepare the dataset
texts = df['url'].tolist()
labels = df['classification'].map(class_to_label).tolist()

# First, split off the test set
train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
    texts, labels, test_size=0.1, stratify=labels, random_state=42
)

# Then split the remaining data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_val_texts, train_val_labels, test_size=0.11111, stratify=train_val_labels, random_state=42
)

# Create datasets
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Prepare data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Print dataset sizes
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
```


```python
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_NAME, 
    num_labels=len(class_to_label),
    trust_remote_code=True
)
```


```python
def compute_metrics(eval_pred):
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    precision = precision_metric.compute(predictions=preds, references=labels, average="macro")["precision"]
    recall = recall_metric.compute(predictions=preds, references=labels, average="macro")["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    
    # Log metrics to wandb
    wandb.log({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    })
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    # print(f"Confusion Matrix:\n{cm}")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

def compute_metrics_t5(eval_pred):
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    
    logits, labels = eval_pred
    
    # Handle T5 output format
    if isinstance(logits, tuple):
        logits = logits[0]
    
    # Ensure logits are 2D
    if len(logits.shape) > 2:
        logits = logits.squeeze()
    
    # Ensure we have the correct number of classes
    num_classes = len(class_to_label)
    if logits.shape[1] != num_classes:
        raise ValueError(f"Expected logits to have shape (batch_size, {num_classes}), but got {logits.shape}")
    
    preds = np.argmax(logits, axis=1)
    
    precision = precision_metric.compute(predictions=preds, references=labels, average="macro")["precision"]
    recall = recall_metric.compute(predictions=preds, references=labels, average="macro")["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    
    # Log metrics to wandb
    wandb.log({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    })
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }
```


```python
model
```


```python
for param in model.new.encoder.parameters():
    param.requires_grad = False

# Ensure the embeddings, pooler, and classifier layers are trainable
for param in model.new.embeddings.parameters():
    param.requires_grad = True
for param in model.new.pooler.parameters():
    param.requires_grad = True
for param in model.classifier.parameters():
    param.requires_grad = True
```


```python
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}")
    #else:
    #    print(f"Frozen: {name}")
```


```python
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=300,
    save_steps=1000,
    logging_steps=25,
    learning_rate=3e-4,
    num_train_epochs=3,
    seed=42,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="wandb",
    
    # torch_compile=True,
    gradient_accumulation_steps=16,
    eval_accumulation_steps=16,
    lr_scheduler_type="cosine",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
```


```python
#from kaggle_secrets import UserSecretsClient
#secret_label = "wandb"
#secret_value = UserSecretsClient().get_secret(secret_label)

#wandb.login(key=)
wandb.init(project="pdf-classification", name=BASE_MODEL_NAME+DATASET_PATH)
```


```python
# Train the model
trainer.train()

# Save the model
trainer.save_model(os.path.join(CHECKPOINT_DIR, "final"))
```


```python
wandb.finish()
```


```python
## Save the label mappings
import json
with open(os.path.join(CHECKPOINT_DIR, 'label_mappings.json'), 'w') as f:
    json.dump({
        'class_to_label': class_to_label,
        'label_to_class': label_to_class
    }, f)

print("Training completed. Model and label mappings saved.")
```
