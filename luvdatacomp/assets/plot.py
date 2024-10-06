import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data
fractions = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
metrics = {
    'ImageNet': np.array([0.006, 0.018, 0.023, 0.027, 0.029, 0.026, 0.026]),
    'ImageNet dist. shifts': np.array([0.012, 0.026, 0.032, 0.038, 0.038, 0.036, 0.034]),
    'VTAB': np.array([0.116, 0.133, 0.146, 0.154, 0.152, 0.156, 0.148]),
    'Retrieval': np.array([0.061, 0.085, 0.103, 0.110, 0.118, 0.112, 0.112]),
    'Average': np.array([0.093, 0.122, 0.132, 0.140, 0.142, 0.143, 0.137])
}

# Define fitting functions
def log_func(x, a, b, c):
    return a * np.log(b * x) + c

def power_func(x, a, b, c):
    return a * x**b + c

# Create composite score
normalized_metrics = [m / np.max(m) for m in metrics.values()]
composite_score = np.mean(normalized_metrics, axis=0)

# Generate points for smooth curves
x_smooth = np.linspace(0.1, 1.0, 100)

# Plot all individual metrics together
plt.figure(figsize=(15, 10))
colors = ['b', 'g', 'r', 'c', 'm']

for (name, metric), color in zip(metrics.items(), colors):
    plt.scatter(fractions, metric, label=name, color=color)
    
    try:
        log_params, _ = curve_fit(log_func, fractions, metric, maxfev=5000)
        plt.plot(x_smooth, log_func(x_smooth, *log_params), '--', color=color, label=f'{name} (Log)')
    except RuntimeError:
        print(f"Log fitting failed for {name}")
    
    try:
        power_params, _ = curve_fit(power_func, fractions, metric, maxfev=5000)
        plt.plot(x_smooth, power_func(x_smooth, *power_params), ':', color=color, label=f'{name} (Power)')
    except RuntimeError:
        print(f"Power fitting failed for {name}")

plt.xscale('log')
plt.xlabel('Fraction of Dataset Used')
plt.ylabel('Performance')
plt.title('Performance Metrics vs Dataset Size (Log and Power Law Fits)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('all_metrics_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot composite score separately
plt.figure(figsize=(12, 8))
plt.scatter(fractions, composite_score, label='Data', color='blue', s=50)

try:
    log_params, _ = curve_fit(log_func, fractions, composite_score, maxfev=5000)
    plt.plot(x_smooth, log_func(x_smooth, *log_params), 'r--', label='Log Fit')
    log_max = log_func(1.0, *log_params)
    log_optimal = np.interp(0.95 * log_max, log_func(x_smooth, *log_params), x_smooth)
    plt.axvline(log_optimal, color='r', linestyle='--')
    plt.text(log_optimal, plt.ylim()[0], f'Log Optimal: {log_optimal:.2f}', 
             rotation=90, verticalalignment='bottom', color='r')
    print(f"Composite Score - Log fit optimal fraction: {log_optimal:.2f}")
except RuntimeError:
    print("Log fitting failed for Composite Score")

try:
    power_params, _ = curve_fit(power_func, fractions, composite_score, maxfev=5000)
    plt.plot(x_smooth, power_func(x_smooth, *power_params), 'g:', label='Power Fit')
    power_max = power_func(1.0, *power_params)
    power_optimal = np.interp(0.95 * power_max, power_func(x_smooth, *power_params), x_smooth)
    plt.axvline(power_optimal, color='g', linestyle=':')
    plt.text(power_optimal, plt.ylim()[0], f'Power Optimal: {power_optimal:.2f}', 
             rotation=90, verticalalignment='bottom', color='g')
    print(f"Composite Score - Power fit optimal fraction: {power_optimal:.2f}")
except RuntimeError:
    print("Power fitting failed for Composite Score")

plt.xscale('log')
plt.xlabel('Fraction of Dataset Used')
plt.ylabel('Normalized Score')
plt.title('Composite Score vs Dataset Size (Log and Power Law Fits)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('composite_score_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Analysis plots saved as 'all_metrics_analysis.png' and 'composite_score_analysis.png'")
