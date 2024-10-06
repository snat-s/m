import matplotlib.pyplot as plt
import numpy as np

# Data from the table
models = [
    "baseline from DataComp paper", "my_baseline", "minipile_style_only_txt", 
    "minipile_style_txt_img", "txt_top5_all_quality_clusters", 
    "txt_top5_english_quality_clusters"
]
percentages = [100, 81.15, 5.77, 10.08, 22.38, 10.29]
averages = [0.132, 0.137, 0.111, 0.114, 0.126, 0.121]
labels_seen = [12800000, 10386623, 739116, 1290236, 2864016, 1316522]

# Plot with labels (model names) on the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(percentages, averages, color='blue', label='Data Points')

# Annotate each point with the model name
for i, model in enumerate(models):
    plt.annotate(model, (percentages[i], averages[i]), textcoords="offset points", xytext=(0,5), ha='center')

plt.title('Model Performance vs. Dataset Percentage')
plt.xlabel('Percentage of Original Dataset (%)')
plt.ylabel('Average Performance')
plt.xticks(np.arange(0, 101, 10))
plt.grid(True)
plt.savefig('assets/model_performance_vs_percentage.png')
plt.close()

# Plot with labels seen instead of percentages
plt.figure(figsize=(10, 6))
plt.scatter(labels_seen, averages, color='green', label='Data Points')

# Annotate each point with the model name
for i, model in enumerate(models):
    plt.annotate(model, (labels_seen[i], averages[i]), textcoords="offset points", xytext=(0,5), ha='center')

plt.title('Model Performance vs. Labels Seen')
plt.xlabel('Number of Labels Seen')
plt.xscale('log')
plt.ylabel('Average Performance')
plt.grid(True)
plt.savefig('assets/model_performance_vs_labels_seen.png')
plt.close()
