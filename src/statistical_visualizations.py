"""
Statistical Visualizations for Advanced Analysis
================================================
Comprehensive visualization suite for statisticians and researchers

Features:
- Distribution analysis (QQ plots, histograms, KDE)
- Time series diagnostics (ACF, PACF, decomposition)
- Model diagnostics (residual analysis, homoscedasticity tests)
- Correlation analysis (heatmaps, pairplots)
- Performance comparison (box plots, violin plots)
- Statistical tests visualization
- Confidence intervals and prediction bands
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, kstest
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.2)


def plot_comprehensive_distribution_analysis(data, title, save_path):
    """
    Comprehensive distribution analysis with multiple statistical tests
    
    Args:
        data: Array of values to analyze
        title: Title for the plot
        save_path: Path to save the figure
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Histogram with KDE
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.hist(data, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Fit and plot normal distribution
    mu, std = stats.norm.fit(data)
    xmin, xmax = ax1.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax1.plot(x, p, 'r-', linewidth=2, label=f'Normal fit: μ={mu:.2f}, σ={std:.2f}')
    
    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    ax1.plot(x, kde(x), 'g-', linewidth=2, label='KDE')
    
    ax1.set_xlabel('Value', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution with Normal Fit & KDE', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-Q Plot
    ax2 = fig.add_subplot(gs[0, 2:])
    stats.probplot(data, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normality Check)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add R² for Q-Q plot
    _, (slope, intercept, r) = stats.probplot(data, dist="norm")
    ax2.text(0.05, 0.95, f'R² = {r**2:.4f}', transform=ax2.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Box Plot
    ax3 = fig.add_subplot(gs[1, 0])
    bp = ax3.boxplot(data, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    ax3.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax3.set_title('Box Plot\n(Outlier Detection)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    ax3.text(0.5, 0.02, f'Q1={q1:.2f}\nMedian={median:.2f}\nQ3={q3:.2f}\nIQR={iqr:.2f}',
             transform=ax3.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 4. Violin Plot
    ax4 = fig.add_subplot(gs[1, 1])
    parts = ax4.violinplot([data], positions=[1], showmeans=True, showmedians=True)
    ax4.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax4.set_title('Violin Plot\n(Distribution Shape)', fontsize=13, fontweight='bold')
    ax4.set_xticks([])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Empirical CDF
    ax5 = fig.add_subplot(gs[1, 2])
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax5.plot(sorted_data, y, linewidth=2, color='darkblue')
    ax5.set_xlabel('Value', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax5.set_title('Empirical CDF', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Statistical Tests Summary
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.axis('off')
    
    # Perform statistical tests
    shapiro_stat, shapiro_p = shapiro(data[:5000] if len(data) > 5000 else data)  # Shapiro limited to 5000
    anderson_result = stats.anderson(data, dist='norm')
    jarque_stat, jarque_p = stats.jarque_bera(data)
    
    # Skewness and Kurtosis
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    test_text = f"""
    STATISTICAL TESTS
    ═════════════════════════
    
    Descriptive Statistics:
    • Mean: {np.mean(data):.4f}
    • Std Dev: {np.std(data):.4f}
    • Skewness: {skewness:.4f}
    • Kurtosis: {kurtosis:.4f}
    
    Normality Tests:
    • Shapiro-Wilk:
      W = {shapiro_stat:.4f}
      p-value = {shapiro_p:.4e}
      {'✓ Normal' if shapiro_p > 0.05 else '✗ Not Normal'}
    
    • Jarque-Bera:
      JB = {jarque_stat:.4f}
      p-value = {jarque_p:.4e}
      {'✓ Normal' if jarque_p > 0.05 else '✗ Not Normal'}
    
    • Anderson-Darling:
      Statistic = {anderson_result.statistic:.4f}
      Critical (5%) = {anderson_result.critical_values[2]:.4f}
      {'✓ Normal' if anderson_result.statistic < anderson_result.critical_values[2] else '✗ Not Normal'}
    
    Outliers:
    • Count: {np.sum((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr))}
    • Percentage: {100 * np.sum((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)) / len(data):.2f}%
    """
    
    ax6.text(0.05, 0.95, test_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 7-9. Percentile Analysis
    ax7 = fig.add_subplot(gs[2, :])
    percentiles = np.percentile(data, np.arange(0, 101, 1))
    ax7.plot(np.arange(0, 101, 1), percentiles, linewidth=2, color='darkgreen')
    
    # Mark key percentiles
    key_percentiles = [5, 25, 50, 75, 95]
    key_values = np.percentile(data, key_percentiles)
    ax7.scatter(key_percentiles, key_values, s=100, c='red', zorder=5)
    
    for p, v in zip(key_percentiles, key_values):
        ax7.annotate(f'{p}%: {v:.2f}', xy=(p, v), xytext=(p, v + 0.5),
                    fontsize=9, ha='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax7.set_xlabel('Percentile', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax7.set_title('Percentile Analysis', fontsize=13, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'skewness': skewness,
        'kurtosis': kurtosis,
        'shapiro_p': shapiro_p,
        'jarque_p': jarque_p
    }


def plot_time_series_diagnostics(data, dates, title, save_path):
    """
    Comprehensive time series diagnostic plots
    
    Args:
        data: Time series data
        dates: Corresponding dates
        title: Plot title
        save_path: Save path
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.25)
    
    # 1. Time Series Plot with Statistics
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, data, linewidth=1.5, color='darkblue', alpha=0.8)
    
    # Add rolling statistics
    rolling_mean = pd.Series(data).rolling(window=20).mean()
    rolling_std = pd.Series(data).rolling(window=20).std()
    
    ax1.plot(dates, rolling_mean, 'r-', linewidth=2, label='20-day MA', alpha=0.8)
    ax1.fill_between(dates, rolling_mean - 2*rolling_std, rolling_mean + 2*rolling_std,
                      alpha=0.2, color='red', label='±2σ band')
    
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax1.set_title('Time Series with Rolling Statistics', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. ACF Plot
    ax2 = fig.add_subplot(gs[1, 0])
    plot_acf(data, lags=40, ax=ax2, color='steelblue')
    ax2.set_title('Autocorrelation Function (ACF)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Lag', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. PACF Plot
    ax3 = fig.add_subplot(gs[1, 1])
    plot_pacf(data, lags=40, ax=ax3, color='darkgreen', method='ywm')
    ax3.set_title('Partial Autocorrelation Function (PACF)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Lag', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. First Difference
    ax4 = fig.add_subplot(gs[2, 0])
    diff_data = np.diff(data)
    ax4.plot(dates[1:], diff_data, linewidth=1.5, color='purple', alpha=0.7)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Δ Value', fontsize=11, fontweight='bold')
    ax4.set_title('First Difference (Stationarity Check)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    # Add ADF test result
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(data)
    ax4.text(0.02, 0.98, f'ADF Statistic: {adf_result[0]:.4f}\n'
                          f'p-value: {adf_result[1]:.4f}\n'
                          f'{"✓ Stationary" if adf_result[1] < 0.05 else "✗ Non-Stationary"}',
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 5. Returns Distribution
    ax5 = fig.add_subplot(gs[2, 1])
    returns = np.diff(data) / data[:-1] * 100  # Percentage returns
    ax5.hist(returns, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black')
    
    # Fit normal
    mu_r, std_r = stats.norm.fit(returns)
    xmin, xmax = ax5.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu_r, std_r)
    ax5.plot(x, p, 'r-', linewidth=2, label=f'Normal: μ={mu_r:.3f}%, σ={std_r:.3f}%')
    
    ax5.set_xlabel('Returns (%)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax5.set_title('Returns Distribution', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6-8. Seasonal Decomposition (if enough data)
    if len(data) >= 30:
        try:
            # Create series with proper frequency
            ts = pd.Series(data, index=pd.date_range(start='2024-01-01', periods=len(data), freq='D'))
            decomposition = seasonal_decompose(ts, model='additive', period=min(30, len(data)//2))
            
            ax6 = fig.add_subplot(gs[3, :])
            ax6.plot(dates, decomposition.trend, linewidth=2, color='green', label='Trend')
            ax6.set_title('Trend Component', fontsize=13, fontweight='bold')
            ax6.set_xlabel('Date', fontsize=11, fontweight='bold')
            ax6.set_ylabel('Trend', fontsize=11, fontweight='bold')
            ax6.legend(fontsize=10)
            ax6.grid(True, alpha=0.3)
            plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
            
            ax7 = fig.add_subplot(gs[4, 0])
            ax7.plot(dates, decomposition.seasonal, linewidth=1.5, color='red', alpha=0.7)
            ax7.set_title('Seasonal Component', fontsize=13, fontweight='bold')
            ax7.set_xlabel('Date', fontsize=11, fontweight='bold')
            ax7.set_ylabel('Seasonal', fontsize=11, fontweight='bold')
            ax7.grid(True, alpha=0.3)
            plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45)
            
            ax8 = fig.add_subplot(gs[4, 1])
            ax8.plot(dates, decomposition.resid, linewidth=1.5, color='brown', alpha=0.7)
            ax8.axhline(y=0, color='k', linestyle='--', linewidth=1)
            ax8.set_title('Residual Component', fontsize=13, fontweight='bold')
            ax8.set_xlabel('Date', fontsize=11, fontweight='bold')
            ax8.set_ylabel('Residual', fontsize=11, fontweight='bold')
            ax8.grid(True, alpha=0.3)
            plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45)
        except:
            # If decomposition fails, show placeholder
            ax6 = fig.add_subplot(gs[3:, :])
            ax6.text(0.5, 0.5, 'Seasonal decomposition requires more data points',
                    ha='center', va='center', fontsize=14,
                    transform=ax6.transAxes)
            ax6.axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_residual_analysis(y_true, y_pred, title, save_path):
    """
    Comprehensive residual analysis for model diagnostics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Save path
    """
    residuals = y_true - y_pred
    standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Residuals vs Fitted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_pred, residuals, alpha=0.6, s=30, color='steelblue')
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    # Add lowess smooth
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smooth = lowess(residuals, y_pred, frac=0.3)
    ax1.plot(smooth[:, 0], smooth[:, 1], 'g-', linewidth=2, label='LOWESS')
    
    ax1.set_xlabel('Fitted Values', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Residuals', fontsize=11, fontweight='bold')
    ax1.set_title('Residuals vs Fitted\n(Homoscedasticity Check)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-Q Plot of Residuals
    ax2 = fig.add_subplot(gs[0, 1])
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Normal Q-Q Plot\n(Residuals Normality)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Scale-Location Plot
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.6, s=30, color='coral')
    ax3.set_xlabel('Fitted Values', fontsize=11, fontweight='bold')
    ax3.set_ylabel('√|Standardized Residuals|', fontsize=11, fontweight='bold')
    ax3.set_title('Scale-Location Plot\n(Spread Consistency)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Residuals Histogram
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(residuals, bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
    
    # Fit normal
    mu, std = stats.norm.fit(residuals)
    xmin, xmax = ax4.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax4.plot(x, p, 'r-', linewidth=2, label=f'Normal: μ={mu:.3f}, σ={std:.3f}')
    
    ax4.set_xlabel('Residuals', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax4.set_title('Residuals Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. ACF of Residuals
    ax5 = fig.add_subplot(gs[1, 1])
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals, lags=min(40, len(residuals)//2), ax=ax5, color='steelblue')
    ax5.set_title('ACF of Residuals\n(Independence Check)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Residuals vs Index
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(range(len(residuals)), residuals, alpha=0.6, s=30, color='purple')
    ax6.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax6.set_xlabel('Observation Index', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Residuals', fontsize=11, fontweight='bold')
    ax6.set_title('Residuals vs Order\n(Serial Correlation)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. Actual vs Predicted
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.scatter(y_true, y_pred, alpha=0.6, s=30, color='darkblue')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax7.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax7.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax7.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax7.set_xlabel('Actual Values', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
    ax7.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # 8. Statistical Tests Summary
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')
    
    # Perform tests
    shapiro_stat, shapiro_p = shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
    
    # Breusch-Pagan test for heteroscedasticity
    from scipy.stats import pearsonr
    bp_stat, bp_p = pearsonr(y_pred, residuals**2)
    
    # Durbin-Watson statistic
    dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
    
    test_text = f"""
    RESIDUAL DIAGNOSTICS
    ══════════════════════════════════
    
    Basic Statistics:
    • Mean: {np.mean(residuals):.6f}
    • Std Dev: {np.std(residuals):.4f}
    • Min: {np.min(residuals):.4f}
    • Max: {np.max(residuals):.4f}
    
    Normality (Shapiro-Wilk):
    • W-statistic: {shapiro_stat:.4f}
    • p-value: {shapiro_p:.4e}
    • Result: {'✓ Normal (α=0.05)' if shapiro_p > 0.05 else '✗ Not Normal'}
    
    Homoscedasticity:
    • Correlation(fitted, res²): {bp_stat:.4f}
    • p-value: {bp_p:.4e}
    • Result: {'✓ Homoscedastic' if abs(bp_stat) < 0.3 else '⚠ Check variance'}
    
    Independence (Durbin-Watson):
    • DW statistic: {dw:.4f}
    • Expected: ~2.0
    • Result: {'✓ No autocorrelation' if 1.5 < dw < 2.5 else '⚠ Check autocorrelation'}
    
    Model Fit:
    • R²: {r2:.4f}
    • RMSE: {np.sqrt(np.mean(residuals**2)):.4f}
    • MAE: {np.mean(np.abs(residuals)):.4f}
    • MAPE: {np.mean(np.abs(residuals / y_true)) * 100:.2f}%
    """
    
    ax8.text(0.05, 0.95, test_text, transform=ax8.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_heatmap(df, features, title, save_path):
    """
    Create comprehensive correlation analysis
    
    Args:
        df: DataFrame with features
        features: List of feature names to analyze
        title: Plot title
        save_path: Save path
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # Filter to existing features
    available_features = [f for f in features if f in df.columns]
    corr_matrix = df[available_features].corr()
    
    # 1. Full correlation heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title('Correlation Matrix (Lower Triangle)', fontsize=14, fontweight='bold')
    
    # 2. Sorted correlations
    # Get upper triangle correlations
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))
    
    # Sort by absolute correlation
    corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
    
    # Plot top correlations
    top_n = min(20, len(corr_pairs_sorted))
    labels = [f"{pair[0][:10]}–{pair[1][:10]}" for pair in corr_pairs_sorted[:top_n]]
    values = [pair[2] for pair in corr_pairs_sorted[:top_n]]
    colors = ['red' if v < 0 else 'green' for v in values]
    
    y_pos = np.arange(top_n)
    axes[1].barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels, fontsize=9)
    axes[1].set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Top {top_n} Feature Correlations', fontsize=14, fontweight='bold')
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=1)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Add values on bars
    for i, v in enumerate(values):
        axes[1].text(v + 0.02 if v > 0 else v - 0.02, i, f'{v:.3f}',
                    va='center', ha='left' if v > 0 else 'right', fontsize=8)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_performance_comparison(results_dict, save_path):
    """
    Create comprehensive model performance comparison
    
    Args:
        results_dict: Dictionary of {model_name: metrics_dict}
        save_path: Save path
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    models = list(results_dict.keys())
    metrics = ['rmse', 'mae', 'mape', 'r2']
    
    # Extract values
    rmse_vals = [results_dict[m].get('rmse', 0) for m in models]
    mae_vals = [results_dict[m].get('mae', 0) for m in models]
    mape_vals = [results_dict[m].get('mape', 0) for m in models]
    r2_vals = [results_dict[m].get('r2', 0) for m in models]
    
    # 1. RMSE Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars1 = ax1.bar(range(len(models)), rmse_vals, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('RMSE ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Root Mean Squared Error', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Highlight best
    best_idx = np.argmin(rmse_vals)
    bars1[best_idx].set_edgecolor('gold')
    bars1[best_idx].set_linewidth(4)
    
    # Add values
    for i, v in enumerate(rmse_vals):
        ax1.text(i, v + max(rmse_vals)*0.02, f'${v:.2f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold' if i == best_idx else 'normal')
    
    # 2. MAE Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(range(len(models)), mae_vals, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('MAE ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Absolute Error', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    best_idx_mae = np.argmin(mae_vals)
    bars2[best_idx_mae].set_edgecolor('gold')
    bars2[best_idx_mae].set_linewidth(4)
    
    # 3. MAPE Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    bars3 = ax3.bar(range(len(models)), mape_vals, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax3.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Mean Absolute Percentage Error', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    best_idx_mape = np.argmin(mape_vals)
    bars3[best_idx_mape].set_edgecolor('gold')
    bars3[best_idx_mape].set_linewidth(4)
    
    # 4. R² Comparison
    ax4 = fig.add_subplot(gs[1, 0])
    bars4 = ax4.bar(range(len(models)), r2_vals, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_xticks(range(len(models)))
    ax4.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax4.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax4.set_title('Coefficient of Determination', fontsize=13, fontweight='bold')
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax4.grid(True, alpha=0.3, axis='y')
    
    best_idx_r2 = np.argmax(r2_vals)
    bars4[best_idx_r2].set_edgecolor('gold')
    bars4[best_idx_r2].set_linewidth(4)
    
    # 5. Radar Chart (Spider Plot)
    ax5 = fig.add_subplot(gs[1, 1], projection='polar')
    
    # Normalize metrics for radar chart
    norm_rmse = 1 - np.array(rmse_vals) / max(rmse_vals)
    norm_mae = 1 - np.array(mae_vals) / max(mae_vals)
    norm_mape = 1 - np.array(mape_vals) / max(mape_vals)
    norm_r2 = np.array(r2_vals)
    
    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False).tolist()
    angles += angles[:1]
    
    for i, model in enumerate(models[:3]):  # Show top 3 for clarity
        values = [norm_rmse[i], norm_mae[i], norm_mape[i], norm_r2[i]]
        values += values[:1]
        ax5.plot(angles, values, 'o-', linewidth=2, label=model)
        ax5.fill(angles, values, alpha=0.15)
    
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(['RMSE\n(inverted)', 'MAE\n(inverted)', 'MAPE\n(inverted)', 'R²'], fontsize=10)
    ax5.set_ylim(0, 1)
    ax5.set_title('Multi-Metric Comparison\n(Top 3 Models)', fontsize=13, fontweight='bold', pad=20)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax5.grid(True)
    
    # 6. Summary Table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Create summary table
    summary_text = "MODEL RANKINGS\n" + "="*50 + "\n\n"
    
    # Best by each metric
    summary_text += f"🏆 Best RMSE: {models[best_idx]}\n"
    summary_text += f"   Value: ${rmse_vals[best_idx]:.2f}\n\n"
    
    summary_text += f"🏆 Best MAE: {models[best_idx_mae]}\n"
    summary_text += f"   Value: ${mae_vals[best_idx_mae]:.2f}\n\n"
    
    summary_text += f"🏆 Best MAPE: {models[best_idx_mape]}\n"
    summary_text += f"   Value: {mape_vals[best_idx_mape]:.2f}%\n\n"
    
    summary_text += f"🏆 Best R²: {models[best_idx_r2]}\n"
    summary_text += f"   Value: {r2_vals[best_idx_r2]:.4f}\n\n"
    
    summary_text += "="*50 + "\n"
    summary_text += f"Overall Winner: {models[best_idx]}\n"
    summary_text += f"(Based on lowest RMSE)"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



