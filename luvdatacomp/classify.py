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
    with open(file_path, "r") as file:
        return file.read()

def classify_cluster(client, prompt_template, cluster_data):
    closest_samples = json.dumps(cluster_data['closest_samples'], ensure_ascii=False, indent=2)
    furthest_samples = json.dumps(cluster_data['furthest_samples'], ensure_ascii=False, indent=2)
    prompt = prompt_template.format(closest_samples=closest_samples, furthest_samples=furthest_samples)
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that classifies image-text clusters for CLIP model training."},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )
    return response.choices[0].message.content

def classify(cluster_samples_path, prompt_file, classifications_file):
    # Load API key and base URL from config file
    config = load_json("config.json")
    api_key = config["API_KEY"]
    base_url = config["BASE_URL"]
    
    client = OpenAI(api_key=api_key, base_url=base_url)

    clusters = load_json(cluster_samples_path)
    prompt_template = load_prompt(prompt_file)

    results = {}
    for cluster_name, cluster_data in tqdm(clusters.items(), desc="Processing Clusters", unit="cluster"):
        classification = classify_cluster(client, prompt_template, cluster_data)
        results[cluster_name] = classification

    save_json(results, classifications_file)
    print(f"All classifications completed and saved to {classifications_file}")

if __name__ == "__main__":
    CLUSTERS_FILE = "./data/cluster_samples.json"
    PROMPT_FILE = "prompt.txt"
    OUTPUT_FILE = "./data/cluster_classifications.json"
    classify(CLUSTERS_FILE, PROMPT_FILE, OUTPUT_FILE)
