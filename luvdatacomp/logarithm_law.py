import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import warnings

plot_average = True    # Set this to True when you want to plot Average performance
plot_imagenet = False  # Set this to True when you want to plot ImageNet performance

def log_func(x, a, b, c):
    return a * np.log(x) + b * x + c

def power_law(x, a, b):
    return a * x**b

def analyze_and_plot(plot_average=True, plot_imagenet=False):
    df = pd.read_csv('data_training.csv')
    
    new_baseline = pd.DataFrame({
        'Model': ['DataComp Baseline'],
        'Real': [12.8e6],
        'Average': [0.132],
        'ImageNet': [0.025]
    })

    df = pd.concat([df, new_baseline], ignore_index=True)
    
    breaking_models = df[df['Model'].str.contains('breaking_|baseline_|DataComp Baseline')]
    other_models = df[~df['Model'].str.contains('breaking_|baseline_|DataComp Baseline')]
    
    plt.figure(figsize=(12, 8))
    
    if plot_average:
        plot_performance(breaking_models, other_models, 'Average', 'blue', 'red')
    
    if plot_imagenet:
        plot_performance(breaking_models, other_models, 'ImageNet', 'green', 'orange', marker='s')
    
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Performance')
    plt.title('Model Performance vs Training Data Size')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    plt.tight_layout()
    if plot_average:
        plt.savefig("assets/plot_average.png")
    if plot_imagenet:
        plt.savefig("assets/plot_imagenet.png")
    plt.show()

def plot_performance(breaking_models, other_models, metric, color1, color2, marker='o'):
    plt.scatter(breaking_models['Real'], breaking_models[metric], c=color1, marker=marker, label=f'Breaking/Baseline Models ({metric})')
    for i, row in breaking_models.iterrows():
        if not pd.isna(row[metric]):
            plt.annotate(row['Model'], (row['Real'], row[metric]), xytext=(5,5), textcoords='offset points')
    
    plt.scatter(other_models['Real'], other_models[metric], c=color2, marker=marker, label=f'Other Experiments ({metric})')
    for i, row in other_models.iterrows():
        if not pd.isna(row[metric]):
            plt.annotate(row['Model'], (row['Real'], row[metric]), xytext=(5,5), textcoords='offset points')
    
    x = breaking_models['Real'].dropna().values
    y = breaking_models[metric].dropna().values
    
    if len(x) > 0 and len(y) > 0:
        x_fit = np.linspace(min(x), max(x)*1.2, 100)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt_log, _ = curve_fit(log_func, x, y, maxfev=10000, p0=[1e-2, 1e-9, 0.1])
            y_fit_log = log_func(x_fit, *popt_log)
            plt.plot(x_fit, y_fit_log, color=color1, linestyle='-.', label=f'Log Fit ({metric}): y = {popt_log[0]:.2e} * ln(x) + {popt_log[1]:.2e} * x + {popt_log[2]:.2f}')
            print(f"Logarithmic Fit Parameters ({metric}): a={popt_log[0]:.2e}, b={popt_log[1]:.2e}, c={popt_log[2]:.2f}")
            
            # Fit power law
            #popt_power, _ = curve_fit(power_law, x, y, maxfev=10000, p0=[1e-2, 0.1])
            #y_fit_power = power_law(x_fit, *popt_power)
            #plt.plot(x_fit, y_fit_power, color='purple', linestyle='--', label=f'Linear ({metric}): y = {popt_power[0]:.2e} * x^{popt_power[1]:.2f}')
            #print(f"Power Law Fit Parameters ({metric}): a={popt_power[0]:.2e}, b={popt_power[1]:.2f}")
        except RuntimeError:
            print(f"Curve fitting for {metric} performance failed.")
    else:
        print(f"Not enough {metric} data points for fitting.")

analyze_and_plot(plot_average=plot_average, plot_imagenet=plot_imagenet)
