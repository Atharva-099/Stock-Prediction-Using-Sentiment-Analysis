#!/usr/bin/env python3
"""Quick script to regenerate 06_model_comparison.png with Temporal Transformer"""

import matplotlib.pyplot as plt
import numpy as np

# Results from the latest runs (user's verified values)
results_dict = {
    'sklearn_Linear': {'rmse': 1.83, 'mae': 1.24, 'mape': 0.94, 'r2': 0.9992},
    'SARIMAX': {'rmse': 2.66, 'mae': 1.89, 'mape': 1.18, 'r2': 0.9984},
    'Ensemble': {'rmse': 6.66, 'mae': 5.34, 'mape': 3.45, 'r2': 0.9898},
    'TCN': {'rmse': 21.16, 'mae': 17.42, 'mape': 11.04, 'r2': 0.8969},
    'CNN-LSTM': {'rmse': 7.34, 'mae': 6.01, 'mape': 2.64, 'r2': 0.8939},
    'GRU': {'rmse': 7.63, 'mae': 6.44, 'mape': 2.78, 'r2': 0.9356},
    'BiLSTM': {'rmse': 7.77, 'mae': 6.33, 'mape': 2.81, 'r2': 0.9012},
    'LSTM': {'rmse': 12.12, 'mae': 10.58, 'mape': 4.54, 'r2': 0.8909},
    'Transformer': {'rmse': 8.11, 'mae': 6.47, 'mape': 2.80, 'r2': 0.874}  # Properly configured Transformer
}

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

models = list(results_dict.keys())
rmse_vals = [results_dict[m].get('rmse', 0) for m in models]
mae_vals = [results_dict[m].get('mae', 0) for m in models]
mape_vals = [results_dict[m].get('mape', 0) for m in models]
r2_vals = [results_dict[m].get('r2', 0) for m in models]

# Color coding: green for good, yellow for fair, red for failed
colors = []
for r2 in r2_vals:
    if r2 >= 0.95:
        colors.append('#2ecc71')  # Green - Excellent
    elif r2 >= 0.80:
        colors.append('#3498db')  # Blue - Good
    elif r2 >= 0.60:
        colors.append('#f39c12')  # Yellow - Fair
    else:
        colors.append('#e74c3c')  # Red - Failed

# 2.1 RMSE Comparison
ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.bar(range(len(models)), rmse_vals, color=colors, alpha=0.8, edgecolor='black')
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
ax1.set_ylabel('RMSE ($)', fontsize=12, fontweight='bold')
ax1.set_title('Root Mean Squared Error (Lower = Better)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
best_idx = np.argmin([v if v > 0 else float('inf') for v in rmse_vals])
bars1[best_idx].set_edgecolor('gold')
bars1[best_idx].set_linewidth(4)

# 2.2 R² Comparison
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(range(len(models)), r2_vals, color=colors, alpha=0.8, edgecolor='black')
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax2.set_title('Coefficient of Determination (Higher = Better)', fontsize=13, fontweight='bold')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
ax2.axhline(y=0.95, color='g', linestyle='--', linewidth=1, label='Excellent threshold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend()

# 2.3 MAPE Comparison
ax3 = fig.add_subplot(gs[0, 2])
bars3 = ax3.bar(range(len(models)), mape_vals, color=colors, alpha=0.8, edgecolor='black')
ax3.set_xticks(range(len(models)))
ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
ax3.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
ax3.set_title('Mean Absolute Percentage Error (Lower = Better)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 2.4 Multi-metric comparison (using 3 separate normalized bars with proper scaling)
ax4 = fig.add_subplot(gs[1, 0:2])
x = np.arange(len(models))
width = 0.25

# Normalize each metric to 0-1 scale properly
# For RMSE and MAPE: lower is better, so invert (1 - normalized)
# For R²: higher is better, clip to [0,1]
rmse_max = max([abs(v) for v in rmse_vals if v > 0]) if any(v > 0 for v in rmse_vals) else 1
mape_max = max([abs(v) for v in mape_vals if v > 0]) if any(v > 0 for v in mape_vals) else 1

rmse_normalized = [1 - min(v/rmse_max, 1) if v > 0 else 0 for v in rmse_vals]
r2_normalized = [max(0, min(v, 1)) for v in r2_vals]  # Clip R² to [0,1]
mape_normalized = [1 - min(v/mape_max, 1) if v > 0 else 0 for v in mape_vals]

ax4.barh(x - width, rmse_normalized, width, label='RMSE score', color='#3498db', alpha=0.8)
ax4.barh(x, r2_normalized, width, label='R² score', color='#2ecc71', alpha=0.8)
ax4.barh(x + width, mape_normalized, width, label='MAPE score', color='#9b59b6', alpha=0.8)
ax4.set_yticks(x)
ax4.set_yticklabels(models, fontsize=9)
ax4.set_xlabel('Normalized Score (0-1, Higher = Better)', fontsize=12, fontweight='bold')
ax4.set_xlim(0, 1.15)  # Extra space for legend
ax4.set_title('Multi-Metric Performance Comparison (All metrics normalized to 0-1)', fontsize=13, fontweight='bold')
ax4.legend(loc='upper right', fontsize=9, framealpha=0.95)
ax4.grid(True, alpha=0.3, axis='x')

# 2.5 Summary Table
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')

summary_text = """MODEL RANKINGS
══════════════════════════════

  1. sklearn_Linear
     RMSE: $1.83 | R2: 0.9992

  2. SARIMAX
     RMSE: $2.66 | R2: 0.9984

  3. Ensemble
     RMSE: $6.66 | R2: 0.9898
"""

ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.suptitle('Comprehensive Model Performance Comparison (All 9 Models)', fontsize=16, fontweight='bold')
plt.savefig('results/enhanced/statistical/06_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: results/enhanced/statistical/06_model_comparison.png")
