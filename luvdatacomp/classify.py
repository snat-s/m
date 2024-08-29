import json
from tqdm import tqdm
from openai import OpenAI

def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def load_prompt(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def classify_cluster(client, prompt_template, cluster_data):
    cluster_json = json.dumps(cluster_data, ensure_ascii=False, indent=2)
    prompt = prompt_template.format(cluster_data=cluster_json)
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that classifies image-text clusters for CLIP model training."},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )
    return response.choices[0].message.content

def main():
    API_KEY = "sk-600a0e3b93b14b6c93902c2c90a04fdb"
    BASE_URL = "https://api.deepseek.com"
    CLUSTERS_FILE = "./data/cluster_samples.json"
    PROMPT_FILE = "prompt.txt"
    OUTPUT_FILE = "./data/cluster_classifications.json"

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    clusters = load_json(CLUSTERS_FILE)
    prompt_template = load_prompt(PROMPT_FILE)

    results = {}
    for cluster_name, cluster_data in tqdm(clusters.items(), desc="Processing Clusters", unit="cluster"):
        classification = classify_cluster(client, prompt_template, cluster_data['closest_samples'])
        results[cluster_name] = classification

    save_json(results, OUTPUT_FILE)
    print(f"All classifications completed and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
