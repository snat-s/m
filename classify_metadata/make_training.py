import os
import time
from together import Together
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm, trange
import pandas as pd

client = Together(api_key="<YOUR API KEY HERE :)>")

def read_prompt(file_path='prompt.txt'):
    with open(file_path, 'r') as file:
        return file.read()

PDF_CLASSIFICATION_PROMPT = read_prompt()
CSV_FILE_PATH = './data/cc-provenance-20230303.csv'
CLASSIFIED_ALREADY = './data/classified_pdfs_100k.csv'
OUTPUT_FILE = 'classified_pdfs_100k_Llama3.1_8B_Instruct_Turbo.csv'

def classify_pdf(pdf_url):
    prompt = PDF_CLASSIFICATION_PROMPT.format(pdf_url=pdf_url)
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,  # short tags only
        repetition_penalty=1,
        stop=["<|eot_id|>"],
    )
    return response.choices[0].message.content.strip()

class RateLimiter:
    def __init__(self, queries_per_second=10):
        self.queries_per_second = queries_per_second
        self.query_count = 0
        self.last_reset = time.time()

    def __call__(self, f):
        def wrapper(*args, **kwargs):
            current_time = time.time()
            if current_time - self.last_reset >= 1:
                self.query_count = 0
                self.last_reset = current_time

            if self.query_count >= self.queries_per_second:
                sleep_time = 1 - (current_time - self.last_reset)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self.query_count = 0
                self.last_reset = time.time()

            self.query_count += 1
            return f(*args, **kwargs)
        return wrapper

@RateLimiter(queries_per_second=10)
def rate_limited_classify_pdf(pdf_url):
    return classify_pdf(pdf_url)

def classify_batch(pdf_urls):
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(rate_limited_classify_pdf, url): url for url in pdf_urls}
        results = []
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                tag = future.result()
                results.append((url, tag))
            except Exception as exc:
                print(f'{url} generated an exception: {exc}')

    return results

df = pd.read_csv(CSV_FILE_PATH)
df_classified = pd.read_csv(CLASSIFIED_ALREADY)
df = df[~df['url'].isin(df_classified['url'])]

MIN_SAMPLES = 400_000
sample_size = min(MIN_SAMPLES, len(df))
df_sampled = df.sample(n=sample_size, random_state=42)
pdfs_to_classify = df_sampled['url'].tolist()

batch_size = 100  # Adjusted batch size for better control with new rate limiting
total_pdfs = len(pdfs_to_classify)
all_results = []

for i in trange(0, total_pdfs, batch_size):
    batch = pdfs_to_classify[i:i+batch_size]
    results = classify_batch(batch)
    all_results.extend(results)
    print(f"Processed {min(i+batch_size, total_pdfs)} out of {total_pdfs} PDFs")

results_df = pd.DataFrame(all_results, columns=['url', 'classification'])
df_merged = pd.merge(df_sampled, results_df, on='url', how='left')
df_merged.to_csv(OUTPUT_FILE, index=False)
print(f"Classification results saved to {OUTPUT_FILE}")
print(results_df.head())
