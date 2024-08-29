import polars as pl
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt
import numpy as np

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Read the CSV file
df = pl.read_csv("./data/filtered_high_value_english_data.csv")

def tokenize_text(text):
    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return tokens

# Apply tokenization to the 'text' column
df = df.with_columns(
    pl.col("data_points").map_elements(tokenize_text, return_dtype=pl.List(pl.Int64)).alias("tokenized_text")
)

# Count the number of tokens for each entry and filter out those with more than 1024 tokens
df = df.with_columns(
    pl.col("tokenized_text").map_elements(len).alias("token_count")
)
df_filtered = df.filter(pl.col("token_count") <= 200)

# Get the token counts for the filtered data
token_counts = df_filtered["token_count"].to_list()

# Create the histogram
plt.figure(figsize=(12, 6))
plt.hist(token_counts,bins=200,  edgecolor='black')
plt.title('Histogram of Token Counts (≤ 1024 tokens)')
plt.xlabel('Number of Tokens')
plt.ylabel('Frequency')

# Add some statistics to the plot
mean_count = np.mean(token_counts)
median_count = np.median(token_counts)
plt.axvline(mean_count, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_count:.2f}')
plt.axvline(median_count, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_count:.2f}')

plt.legend()

# Save the plot
plt.savefig('./data/token_count_histogram_filtered.png')
plt.close()

print(f"Total entries: {len(token_counts)}")
print(f"Entries removed: {len(df) - len(df_filtered)}")
print(f"Mean token count: {mean_count:.2f}")
print(f"Median token count: {median_count:.2f}")
print(f"Min token count: {min(token_counts)}")
print(f"Max token count: {max(token_counts)}")
