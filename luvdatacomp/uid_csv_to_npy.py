import argparse
import polars as pl
import numpy as np

def process_uids(input_file, output_file):
    # Read the CSV file using Polars
    df = pl.read_csv(input_file)

    # Extract UIDs and convert to pairs of 64-bit integers
    uids = df['uid'].to_numpy()
    processed_uids = np.array([(int(uid[:16], 16), int(uid[16:32], 16)) for uid in uids], 
                              np.dtype("u8,u8"))

    # Sort the array
    processed_uids.sort()

    # Save as .npy file
    np.save(output_file, processed_uids)

    print(f"Processed {len(processed_uids)} UIDs and saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process UIDs from CSV and save as sorted NPY file.")
    parser.add_argument("--input_file", help="Path to the input CSV file")
    parser.add_argument("--output_file", help="Path to the output NPY file")
    
    args = parser.parse_args()

    process_uids(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
