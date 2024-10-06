import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import warnings

# Read the CSV data
df = pd.read_csv('cross_entropy_losses.csv')

# Convert 'Step' to numeric, just in case
df['Step'] = pd.to_numeric(df['Step'])

# Define the models we want to plot
models_to_plot = ['baseline'] + [f'{i}_most_difficult' for i in range(10, 100, 10)]

# List of columns to plot
columns_to_plot = [col for col in df.columns if any(model in col for model in models_to_plot) and 'train/contrastive_loss' in col and not (col.endswith('__MIN') or col.endswith('__MAX'))]

# Define logarithmic function
def log_func(x, a, b, c):
    return a * np.log(x) + b * x + c

# Set up the plot
plt.figure(figsize=(12, 8))

# Plot each column
for column in columns_to_plot:
    label = column.split(' - ')[0]  # Use the part before " - " as the label
    
    x = df['Step'].values
    y = df[column].values
    
    # Plot the actual data
    plt.scatter(x, y, label=f'{label} (data)', alpha=0.6)
    
    # Fit the logarithmic function using curve_fit
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(log_func, x, y, maxfev=10000, p0=[1e-2, 1e-9, 0.1])
        
        # Generate points for the fitted curve
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = log_func(x_fit, *popt)
        
        # Plot the fitted logarithmic curve
        plt.plot(x_fit, y_fit, '--', label=f'{label} (fit)')
        
        print(f"Logarithmic Fit Parameters for {label}: a={popt[0]:.2e}, b={popt[1]:.2e}, c={popt[2]:.2f}")
    except RuntimeError:
        print(f"Logarithmic fit failed for {label}.")

# Customize the plot
plt.title('Cross Entropy Loss vs Training Steps (Log Scale)', fontsize=16)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Cross Entropy Loss', fontsize=12)
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.legend(title='Model', title_fontsize='12', fontsize='10', loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
