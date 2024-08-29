import polars as pl
import numpy as np

# Read the CSV file using Polars
df = pl.read_csv('./data/filtered_high_value_english_data.csv')

# Extract UIDs and convert to pairs of 64-bit integers
uids = df['uid'].to_numpy()
processed_uids = np.array([(int(uid[:16], 16), int(uid[16:32], 16)) for uid in uids], 
                          np.dtype("u8,u8"))

# Sort the array
processed_uids.sort()

# Save as .npy file
np.save("./data/subset_file.npy", processed_uids)

print(f"Processed {len(processed_uids)} UIDs and saved to subset_file.npy")
