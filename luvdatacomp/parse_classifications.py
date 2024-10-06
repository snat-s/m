import json
import re
import polars as pl

def parse_classifications(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Prepare data for DataFrame
    df_data = []
    for cluster, content in data.items():
        cluster_num = re.search(r'Cluster_(\d+)', cluster).group(1)
        
        # Extract reason and classification
        reason_match = re.search(r'### Reason:\s*(.*?)(?=\n\n### Classification:)', content, re.DOTALL)
        # Remove asterisks before capturing the classification status
        classification_match = re.search(r"Classification:\s*\**(Unacceptable|Acceptable)\**", content, re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else ''
        classification = classification_match.group(1).strip() if classification_match else ''
        
        # Remove any remaining quotes and escape characters
        reason = re.sub(r'\\(.)', r'\1', reason)
        classification = re.sub(r'\\(.)', r'\1', classification)
        
        # Remove all breaklines in reason and replace with \n
        reason = reason.replace('\n', '\\n')
        
        df_data.append({"Cluster": int(cluster_num), "Reason": reason, "Classification": classification})

    # Create Polars DataFrame
    df = pl.DataFrame(df_data)
    
    # Sort by Cluster number
    df = df.sort("Cluster")
    
    # Write to CSV
    df.write_csv(output_file)
    
    print(f"CSV file has been created successfully: {output_file}")

if __name__ == "__main__":
    INPUT_FILE = "./non_normalized_data/cluster_classifications_on_minipile_similar_prompt.json"
    OUTPUT_FILE = "./non_normalized_data/classifications_minipile_style.csv"
    parse_classifications(INPUT_FILE, OUTPUT_FILE)
