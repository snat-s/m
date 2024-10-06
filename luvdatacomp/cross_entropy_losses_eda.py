import polars as pl
import matplotlib.pyplot as plt

# Read the CSV data
data = pl.read_csv('cross_entropy_losses.csv')

# Convert 'Step' column to numeric, removing quotes
data = data.with_columns(pl.col('Step').str.replace_all('"', '').cast(pl.Int64))

# List of datasets to plot
datasets = [
    'baseline', '10_most_difficult', '20_most_difficult', '30_most_difficult',
    '50_most_difficult', '70_most_difficult', '90_most_difficult'
]

# Create the plot
plt.figure(figsize=(12, 8))

for dataset in datasets:
    column_name = f"{dataset} - train/contrastive_loss"
    plt_data = data.select(['Step', column_name]).to_pandas()
    plt.plot(plt_data['Step'], plt_data[column_name], label=dataset)

plt.xlabel('Step')
plt.ylabel('Cross-entropy Loss')
plt.title('Cross-entropy Losses: Baseline vs Most Difficult Datasets')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the plot
plt.savefig('assets/cross_entropies_breaking.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
