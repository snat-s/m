import json
import re
import polars as pl

# Read the JSON file
with open('./data/cluster_classifications.json', 'r') as f:
    data = json.load(f)

# Prepare data for DataFrame
df_data = []

for i, (key, value) in enumerate(data.items()):
    reason_match = re.search(r'Reason: (.*?)(?=\n\nClassification:|$)', value, re.DOTALL)
    classification_match = re.search(r'Classification: (.*?)$', value, re.DOTALL)
    
    reason = reason_match.group(1).strip() if reason_match else ''
    classification = classification_match.group(1).strip() if classification_match else ''
    
    df_data.append({"Cluster": i, "Reason": reason, "Classification": classification})

# Create Polars DataFrame
df = pl.DataFrame(df_data)

# Write to CSV
df.write_csv("./data/classification_of_clusters.csv")

print("CSV file has been created successfully.")
