#!/usr/bin/env python3
"""
Weather in Szeged - Comprehensive Data Analysis
Includes: Statistics, Visualization, Correlation, Regression, and ANOVA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory
import os
os.makedirs('/workspace/output', exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 80)
print("WEATHER IN SZEGED - DATA ANALYSIS")
print("=" * 80)

df = pd.read_csv('/workspace/data/weatherHistory.csv')

# ============================================================================
# PART 1: DATASET STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("ЧАСТЬ 1: СТАТИСТИКА ПО ДАТАСЕТУ")
print("=" * 80)

# Number of objects
n_objects = len(df)

# Number of features
n_features = len(df.columns)

# Classes in Precip Type
precip_type_counts = df['Precip Type'].value_counts(dropna=False)

# Percentage of objects with NaN values
nan_mask = df.isnull().any(axis=1)
nan_percentage = (nan_mask.sum() / n_objects) * 100

# Create statistics table
stats_data = {
    'Показатель': [
        'Число объектов в датасете',
        'Число признаков',
        'Число целевых классов по Precip Type',
        'Процент объектов с NaN значениями'
    ],
    'Значение': [
        n_objects,
        n_features,
        df['Precip Type'].nunique(dropna=True),
        f'{nan_percentage:.2f}%'
    ]
}
stats_df = pd.DataFrame(stats_data)
print("\n--- Общая статистика датасета ---")
print(stats_df.to_string(index=False))

# Precip Type classes
print("\n--- Число объектов в каждом классе Precip Type ---")
precip_table = pd.DataFrame({
    'Класс': precip_type_counts.index.tolist(),
    'Количество объектов': precip_type_counts.values
})
print(precip_table.to_string(index=False))

# NaN statistics per column
print("\n--- NaN значения по столбцам ---")
nan_stats = df.isnull().sum()
nan_stats_df = pd.DataFrame({
    'Столбец': nan_stats.index,
    'NaN значений': nan_stats.values,
    'Процент NaN': (nan_stats.values / n_objects * 100).round(2)
})
nan_stats_df = nan_stats_df[nan_stats_df['NaN значений'] > 0]
if len(nan_stats_df) > 0:
    print(nan_stats_df.to_string(index=False))
else:
    print("Нет столбцов с NaN значениями")

print("\n--- Первые 5 строк датасета ---")
print(df.head())

print("\n--- Типы данных ---")
print(df.dtypes)

# ============================================================================
# PART 2: SELECT FEATURES AND CLEAN DATA
# ============================================================================
print("\n" + "=" * 80)
print("ЧАСТЬ 2: ВЫБОР ПРИЗНАКОВ И ОЧИСТКА ДАННЫХ")
print("=" * 80)

# Select 4 quantitative features
selected_features = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']
target = 'Precip Type'

print(f"\nВыбранные количественные признаки: {selected_features}")
print(f"Целевой класс: {target}")

# Create dataset with selected features
df_selected = df[selected_features + [target]].copy()

# Remove rows with NaN values
df_clean = df_selected.dropna()

print(f"\nЧисло объектов до очистки: {len(df_selected)}")
print(f"Число объектов после удаления NaN: {len(df_clean)}")
print(f"Удалено объектов: {len(df_selected) - len(df_clean)}")

# Check classes in cleaned data
print("\n--- Классы в очищенном датасете ---")
print(df_clean[target].value_counts())

# ============================================================================
# PART 3: PAIRPLOT VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("ЧАСТЬ 3: ВИЗУАЛИЗАЦИЯ ДАТАСЕТА (PAIRPLOT)")
print("=" * 80)

# Create pairplot
fig = plt.figure(figsize=(16, 14))
g = sns.pairplot(
    df_clean, 
    hue=target,
    vars=selected_features,
    diag_kind='kde',
    plot_kws={'alpha': 0.6, 's': 30},
    diag_kws={'common_norm': False, 'fill': True, 'alpha': 0.5},
    palette='husl',
    markers=['o', 's']
)

# Add histograms with more bins
for i, feature in enumerate(selected_features):
    for j, ax in enumerate([g.axes[i, i]]):
        ax.clear()
        for precip_type in df_clean[target].unique():
            data = df_clean[df_clean[target] == precip_type][feature]
            ax.hist(data, bins=30, alpha=0.5, label=precip_type, density=True)
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')

g.fig.suptitle('Визуализация Weather in Szeged по парам признаков', y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig('/workspace/output/pairplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Pairplot сохранен в /workspace/output/pairplot.png")

# Alternative: Create detailed pairplot with better styling
fig, axes = plt.subplots(len(selected_features), len(selected_features), figsize=(16, 14))

colors = {'rain': '#3498db', 'snow': '#e74c3c'}
markers = {'rain': 'o', 'snow': 's'}

for i, feat_i in enumerate(selected_features):
    for j, feat_j in enumerate(selected_features):
        ax = axes[i, j]
        
        if i == j:
            # Diagonal: histograms/KDE
            for precip_type in df_clean[target].unique():
                data = df_clean[df_clean[target] == precip_type][feat_i]
                ax.hist(data, bins=25, alpha=0.5, label=precip_type, 
                       color=colors[precip_type], density=True, edgecolor='white')
                # Add KDE line
                if len(data) > 1:
                    kde = stats.gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 100)
                    ax.plot(x_range, kde(x_range), color=colors[precip_type], linewidth=2)
            if i == 0:
                ax.legend()
        else:
            # Off-diagonal: scatter plots
            for precip_type in df_clean[target].unique():
                mask = df_clean[target] == precip_type
                ax.scatter(
                    df_clean.loc[mask, feat_j], 
                    df_clean.loc[mask, feat_i],
                    alpha=0.4, 
                    s=15,
                    c=colors[precip_type],
                    marker=markers[precip_type],
                    label=precip_type
                )
        
        if j == 0:
            ax.set_ylabel(feat_i, fontsize=9)
        if i == len(selected_features) - 1:
            ax.set_xlabel(feat_j, fontsize=9)

plt.suptitle('Визуализация Weather in Szeged по парам признаков\n(Гистограммы на диагонали, Scatter plots вне диагонали)', 
             y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig('/workspace/output/pairplot_detailed.png', dpi=150, bbox_inches='tight')
plt.close()
print("Детальный pairplot сохранен в /workspace/output/pairplot_detailed.png")

# ============================================================================
# PART 4: CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("ЧАСТЬ 4: КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("=" * 80)

def calculate_correlations(data, features, title):
    """Calculate Pearson and Spearman correlations with p-values"""
    n = len(features)
    
    # Pearson
    pearson_corr = np.zeros((n, n))
    pearson_pval = np.zeros((n, n))
    
    # Spearman
    spearman_corr = np.zeros((n, n))
    spearman_pval = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                pearson_corr[i, j] = 1.0
                spearman_corr[i, j] = 1.0
                pearson_pval[i, j] = 0.0
                spearman_pval[i, j] = 0.0
            else:
                x = data[features[i]].values
                y = data[features[j]].values
                
                pc, pp = pearsonr(x, y)
                sc, sp = spearmanr(x, y)
                
                pearson_corr[i, j] = pc
                pearson_pval[i, j] = pp
                spearman_corr[i, j] = sc
                spearman_pval[i, j] = sp
    
    return pearson_corr, pearson_pval, spearman_corr, spearman_pval

def print_correlation_table(features, corr, pval, corr_type):
    """Print correlation table with p-values"""
    print(f"\n--- {corr_type} корреляция ---")
    
    # Create formatted table
    table_data = []
    for i, feat_i in enumerate(features):
        row = [feat_i[:15]]  # Truncate feature name
        for j, feat_j in enumerate(features):
            row.append(f"{corr[i, j]:.3f}")
        table_data.append(row)
    
    headers = [''] + [f[:15] for f in features]
    corr_df = pd.DataFrame(table_data, columns=headers)
    print(corr_df.to_string(index=False))
    
    print(f"\n--- {corr_type} p-значения ---")
    table_data = []
    for i, feat_i in enumerate(features):
        row = [feat_i[:15]]
        for j, feat_j in enumerate(features):
            if i == j:
                row.append("-")
            else:
                p = pval[i, j]
                if p < 0.001:
                    row.append("<0.001")
                else:
                    row.append(f"{p:.4f}")
        table_data.append(row)
    
    pval_df = pd.DataFrame(table_data, columns=headers)
    print(pval_df.to_string(index=False))

# Calculate correlations for entire dataset
print("\n=== КОРРЕЛЯЦИИ НА ВСЕМ ДАТАСЕТЕ ===")
pc, pp, sc, sp = calculate_correlations(df_clean, selected_features, "Весь датасет")
print_correlation_table(selected_features, pc, pp, "Pearson")
print_correlation_table(selected_features, sc, sp, "Spearman")

# Create heatmaps for entire dataset
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pearson heatmap
sns.heatmap(pc, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            xticklabels=[f[:15] for f in selected_features],
            yticklabels=[f[:15] for f in selected_features],
            ax=axes[0], vmin=-1, vmax=1)
axes[0].set_title('Корреляция Пирсона\n(весь датасет)', fontsize=12)

# Spearman heatmap
sns.heatmap(sc, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            xticklabels=[f[:15] for f in selected_features],
            yticklabels=[f[:15] for f in selected_features],
            ax=axes[1], vmin=-1, vmax=1)
axes[1].set_title('Корреляция Спирмана\n(весь датасет)', fontsize=12)

plt.tight_layout()
plt.savefig('/workspace/output/correlation_heatmaps_all.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nТепловые карты (весь датасет) сохранены в /workspace/output/correlation_heatmaps_all.png")

# Correlations by class
classes = df_clean[target].unique()
for precip_type in classes:
    print(f"\n=== КОРРЕЛЯЦИИ ДЛЯ КЛАССА: {precip_type.upper()} ===")
    df_class = df_clean[df_clean[target] == precip_type]
    pc, pp, sc, sp = calculate_correlations(df_class, selected_features, precip_type)
    print_correlation_table(selected_features, pc, pp, "Pearson")
    print_correlation_table(selected_features, sc, sp, "Spearman")
    
    # Create heatmaps for this class
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(pc, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                xticklabels=[f[:15] for f in selected_features],
                yticklabels=[f[:15] for f in selected_features],
                ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title(f'Корреляция Пирсона\n(класс: {precip_type})', fontsize=12)
    
    sns.heatmap(sc, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                xticklabels=[f[:15] for f in selected_features],
                yticklabels=[f[:15] for f in selected_features],
                ax=axes[1], vmin=-1, vmax=1)
    axes[1].set_title(f'Корреляция Спирмана\n(класс: {precip_type})', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'/workspace/output/correlation_heatmaps_{precip_type}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Тепловые карты (класс {precip_type}) сохранены в /workspace/output/correlation_heatmaps_{precip_type}.png")

# ============================================================================
# PART 5: LINEAR REGRESSION WITH PREDICTION INTERVALS
# ============================================================================
print("\n" + "=" * 80)
print("ЧАСТЬ 5: ЛИНЕЙНАЯ РЕГРЕССИЯ С ПРОГНОЗНЫМИ ИНТЕРВАЛАМИ (95%)")
print("=" * 80)

def add_regression_with_prediction_interval(ax, x, y, color, label, alpha=0.05):
    """Add regression line with prediction interval"""
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    if len(x) < 3:
        return None
    
    # Fit regression
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    
    # Create prediction range
    x_pred = np.linspace(x.min(), x.max(), 100)
    X_pred = sm.add_constant(x_pred)
    
    # Get predictions with prediction interval
    predictions = model.get_prediction(X_pred)
    pred_summary = predictions.summary_frame(alpha=alpha)
    
    # Plot regression line
    ax.plot(x_pred, pred_summary['mean'], color=color, linewidth=2, label=f'{label} регрессия')
    
    # Plot prediction interval
    ax.fill_between(x_pred, pred_summary['obs_ci_lower'], pred_summary['obs_ci_upper'],
                    color=color, alpha=0.15, label=f'{label} 95% PI')
    
    return model

# Create pairplot with prediction intervals
fig, axes = plt.subplots(len(selected_features), len(selected_features), figsize=(18, 16))

colors = {'rain': '#3498db', 'snow': '#e74c3c'}
markers = {'rain': 'o', 'snow': 's'}

for i, feat_i in enumerate(selected_features):
    for j, feat_j in enumerate(selected_features):
        ax = axes[i, j]
        
        if i == j:
            # Diagonal: histograms
            for precip_type in df_clean[target].unique():
                data = df_clean[df_clean[target] == precip_type][feat_i]
                ax.hist(data, bins=25, alpha=0.5, label=precip_type,
                       color=colors[precip_type], density=True, edgecolor='white')
            if i == 0:
                ax.legend(fontsize=8)
        else:
            # Off-diagonal: scatter with regression and prediction intervals
            for precip_type in df_clean[target].unique():
                mask = df_clean[target] == precip_type
                x_data = df_clean.loc[mask, feat_j].values
                y_data = df_clean.loc[mask, feat_i].values
                
                ax.scatter(x_data, y_data, alpha=0.3, s=10,
                          c=colors[precip_type], marker=markers[precip_type])
                
                # Add regression with prediction interval
                add_regression_with_prediction_interval(ax, x_data, y_data, 
                                                        colors[precip_type], precip_type)
        
        if j == 0:
            ax.set_ylabel(feat_i, fontsize=9)
        if i == len(selected_features) - 1:
            ax.set_xlabel(feat_j, fontsize=9)

plt.suptitle('Линейная регрессия с 95% прогнозными интервалами (Prediction Intervals)', 
             y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig('/workspace/output/regression_prediction_intervals.png', dpi=150, bbox_inches='tight')
plt.close()
print("График с прогнозными интервалами сохранен в /workspace/output/regression_prediction_intervals.png")

# ============================================================================
# PART 6: LINEAR REGRESSION WITH CONFIDENCE INTERVALS
# ============================================================================
print("\n" + "=" * 80)
print("ЧАСТЬ 6: ЛИНЕЙНАЯ РЕГРЕССИЯ С ДОВЕРИТЕЛЬНЫМИ ИНТЕРВАЛАМИ (95%)")
print("=" * 80)

def add_regression_with_confidence_interval(ax, x, y, color, label, alpha=0.05):
    """Add regression line with confidence interval"""
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    if len(x) < 3:
        return None
    
    # Fit regression
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    
    # Create prediction range
    x_pred = np.linspace(x.min(), x.max(), 100)
    X_pred = sm.add_constant(x_pred)
    
    # Get predictions with confidence interval
    predictions = model.get_prediction(X_pred)
    pred_summary = predictions.summary_frame(alpha=alpha)
    
    # Plot regression line
    ax.plot(x_pred, pred_summary['mean'], color=color, linewidth=2, label=f'{label} регрессия')
    
    # Plot confidence interval (mean_ci for confidence of mean)
    ax.fill_between(x_pred, pred_summary['mean_ci_lower'], pred_summary['mean_ci_upper'],
                    color=color, alpha=0.3, label=f'{label} 95% CI')
    
    return model

# Create pairplot with confidence intervals
fig, axes = plt.subplots(len(selected_features), len(selected_features), figsize=(18, 16))

for i, feat_i in enumerate(selected_features):
    for j, feat_j in enumerate(selected_features):
        ax = axes[i, j]
        
        if i == j:
            # Diagonal: histograms
            for precip_type in df_clean[target].unique():
                data = df_clean[df_clean[target] == precip_type][feat_i]
                ax.hist(data, bins=25, alpha=0.5, label=precip_type,
                       color=colors[precip_type], density=True, edgecolor='white')
            if i == 0:
                ax.legend(fontsize=8)
        else:
            # Off-diagonal: scatter with regression and confidence intervals
            for precip_type in df_clean[target].unique():
                mask = df_clean[target] == precip_type
                x_data = df_clean.loc[mask, feat_j].values
                y_data = df_clean.loc[mask, feat_i].values
                
                ax.scatter(x_data, y_data, alpha=0.3, s=10,
                          c=colors[precip_type], marker=markers[precip_type])
                
                # Add regression with confidence interval
                add_regression_with_confidence_interval(ax, x_data, y_data, 
                                                        colors[precip_type], precip_type)
        
        if j == 0:
            ax.set_ylabel(feat_i, fontsize=9)
        if i == len(selected_features) - 1:
            ax.set_xlabel(feat_j, fontsize=9)

plt.suptitle('Линейная регрессия с 95% доверительными интервалами (Confidence Intervals)', 
             y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig('/workspace/output/regression_confidence_intervals.png', dpi=150, bbox_inches='tight')
plt.close()
print("График с доверительными интервалами сохранен в /workspace/output/regression_confidence_intervals.png")

# ============================================================================
# PART 7: REGRESSION FORMULAS WITH STATISTICAL CRITERIA
# ============================================================================
print("\n" + "=" * 80)
print("ЧАСТЬ 7: ФОРМУЛЫ РЕГРЕССИЙ И СТАТИСТИЧЕСКИЕ КРИТЕРИИ")
print("=" * 80)

def calculate_lr_test(model):
    """Calculate Likelihood Ratio test"""
    # LR = 2 * (LL_full - LL_null)
    ll_full = model.llf
    n = model.nobs
    # Null model: just intercept
    y = model.model.endog
    y_mean = np.mean(y)
    residuals_null = y - y_mean
    sigma2_null = np.sum(residuals_null**2) / n
    ll_null = -n/2 * (np.log(2*np.pi*sigma2_null) + 1)
    
    lr_stat = 2 * (ll_full - ll_null)
    df = 1  # One coefficient (slope)
    p_value = 1 - stats.chi2.cdf(lr_stat, df)
    
    return lr_stat, df, p_value

def calculate_score_test(model):
    """Calculate Score (Lagrange Multiplier) test"""
    # For simple linear regression, score test is asymptotically equivalent to LR test
    # We approximate it using the restricted model's score
    y = model.model.endog
    X = model.model.exog
    n = model.nobs
    
    # Under null hypothesis (slope = 0)
    y_mean = np.mean(y)
    residuals_null = y - y_mean
    sigma2_null = np.sum(residuals_null**2) / n
    
    # Score function gradient
    x_centered = X[:, 1] - np.mean(X[:, 1]) if X.shape[1] > 1 else X[:, 0]
    score = np.sum(x_centered * residuals_null) / sigma2_null
    
    # Information matrix approximation
    info = np.sum(x_centered**2) / sigma2_null
    
    # Score statistic
    score_stat = (score**2) / info if info > 0 else 0
    df = 1
    p_value = 1 - stats.chi2.cdf(score_stat, df)
    
    return score_stat, df, p_value

# Store all regression results
regression_results = []

for precip_type in classes:
    print(f"\n{'='*60}")
    print(f"КЛАСС: {precip_type.upper()}")
    print('='*60)
    
    df_class = df_clean[df_clean[target] == precip_type]
    
    for i, feat_i in enumerate(selected_features):
        for j, feat_j in enumerate(selected_features):
            if i != j:
                x = df_class[feat_j].values
                y = df_class[feat_i].values
                
                # Remove NaN
                mask = ~(np.isnan(x) | np.isnan(y))
                x = x[mask]
                y = y[mask]
                
                if len(x) < 3:
                    continue
                
                # Fit model
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()
                
                # Extract coefficients
                intercept = model.params[0]
                slope = model.params[1]
                
                # R-squared
                r2 = model.rsquared
                
                # Wald test (F-test)
                f_stat = model.fvalue
                f_pvalue = model.f_pvalue
                df_model = model.df_model
                df_resid = model.df_resid
                
                # LR test
                lr_stat, lr_df, lr_pvalue = calculate_lr_test(model)
                
                # Score test
                score_stat, score_df, score_pvalue = calculate_score_test(model)
                
                # Store results
                regression_results.append({
                    'Класс': precip_type,
                    'Y': feat_i[:20],
                    'X': feat_j[:20],
                    'Intercept': intercept,
                    'Slope': slope,
                    'R2': r2,
                    'F-stat': f_stat,
                    'F-df': (df_model, df_resid),
                    'F-pvalue': f_pvalue,
                    'LR-stat': lr_stat,
                    'LR-df': lr_df,
                    'LR-pvalue': lr_pvalue,
                    'Score-stat': score_stat,
                    'Score-df': score_df,
                    'Score-pvalue': score_pvalue
                })

# Print results as table
print("\n" + "=" * 80)
print("СВОДНАЯ ТАБЛИЦА РЕГРЕССИЙ")
print("=" * 80)

for precip_type in classes:
    print(f"\n--- Класс: {precip_type} ---")
    class_results = [r for r in regression_results if r['Класс'] == precip_type]
    
    for r in class_results:
        print(f"\nФормула: {r['Y'][:15]} = {r['Intercept']:.4f} + ({r['Slope']:.6f}) * {r['X'][:15]}")
        print(f"  R² = {r['R2']:.4f}")
        print(f"  F-критерий Вальда: F = {r['F-stat']:.4f}, df = {r['F-df']}, p = {r['F-pvalue']:.2e}")
        print(f"  LR-критерий: χ² = {r['LR-stat']:.4f}, df = {r['LR-df']}, p = {r['LR-pvalue']:.2e}")
        print(f"  Score-критерий: χ² = {r['Score-stat']:.4f}, df = {r['Score-df']}, p = {r['Score-pvalue']:.2e}")

# Create summary DataFrame
summary_data = []
for r in regression_results:
    summary_data.append({
        'Класс': r['Класс'],
        'Формула': f"{r['Y'][:12]}={r['Intercept']:.2f}+{r['Slope']:.4f}*{r['X'][:12]}",
        'R²': f"{r['R2']:.4f}",
        'F(Wald)': f"{r['F-stat']:.2f}",
        'F p-value': f"{r['F-pvalue']:.2e}",
        'LR χ²': f"{r['LR-stat']:.2f}",
        'LR p-value': f"{r['LR-pvalue']:.2e}",
        'Score χ²': f"{r['Score-stat']:.2f}",
        'Score p-value': f"{r['Score-pvalue']:.2e}"
    })

summary_df = pd.DataFrame(summary_data)
print("\n--- Компактная сводка ---")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(summary_df.to_string(index=False))

# Save to CSV
summary_df.to_csv('/workspace/output/regression_summary.csv', index=False)
print("\nСводка сохранена в /workspace/output/regression_summary.csv")

# ============================================================================
# PART 8: ANOVA ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("ЧАСТЬ 8: ДИСПЕРСИОННЫЙ АНАЛИЗ (ANOVA)")
print("=" * 80)

# Select a quantitative feature for ANOVA
anova_feature = 'Temperature (C)'
print(f"\nВыбранный признак для ANOVA: {anova_feature}")

# One-way ANOVA for all classes
groups = [df_clean[df_clean[target] == c][anova_feature].values for c in classes]
f_stat, p_value = f_oneway(*groups)

print(f"\n--- One-way ANOVA для всех классов ---")
print(f"F-статистика: {f_stat:.4f}")
print(f"Степени свободы: df_between = {len(classes) - 1}, df_within = {len(df_clean) - len(classes)}")
print(f"p-значение: {p_value:.2e}")

# Alternative using statsmodels for more detailed output
df_clean['Precip_Type_clean'] = df_clean[target].str.replace(' ', '_')
formula = f'Q("{anova_feature}") ~ C(Precip_Type_clean)'
model = ols(formula, data=df_clean).fit()
anova_table = anova_lm(model, typ=2)
print("\n--- ANOVA таблица ---")
print(anova_table)

# Pairwise comparisons (for each pair of classes)
print("\n--- Попарные сравнения классов ---")
from itertools import combinations

class_list = list(classes)
for c1, c2 in combinations(class_list, 2):
    group1 = df_clean[df_clean[target] == c1][anova_feature].values
    group2 = df_clean[df_clean[target] == c2][anova_feature].values
    
    f_pair, p_pair = f_oneway(group1, group2)
    n1, n2 = len(group1), len(group2)
    df_between = 1
    df_within = n1 + n2 - 2
    
    print(f"\n{c1} vs {c2}:")
    print(f"  F-статистика: {f_pair:.4f}")
    print(f"  Степени свободы: df_between = {df_between}, df_within = {df_within}")
    print(f"  p-значение: {p_pair:.2e}")

# Create raincloud plot with boxplots
print("\n--- Создание Raincloud Plot ---")

fig, ax = plt.subplots(figsize=(14, 8))

# Prepare data for raincloud
data_list = []
positions = []
for i, precip_type in enumerate(classes):
    data = df_clean[df_clean[target] == precip_type][anova_feature].values
    data_list.append(data)
    positions.append(i)

# Colors for each class
rain_colors = ['#3498db', '#e74c3c']

# Plot components for each class
for i, (data, pos, precip_type, color) in enumerate(zip(data_list, positions, classes, rain_colors)):
    # Violin (half violin for raincloud effect)
    parts = ax.violinplot([data], positions=[pos], points=100, widths=0.6,
                          showmeans=False, showmedians=False, showextrema=False)
    
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(0.3)
        # Make it a half violin (cloud)
        m = np.mean(pc.get_paths()[0].vertices[:, 0])
        pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], -np.inf, m)
    
    # Boxplot
    bp = ax.boxplot([data], positions=[pos + 0.15], widths=0.2, 
                    patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor(color)
    bp['boxes'][0].set_alpha(0.7)
    bp['medians'][0].set_color('black')
    bp['medians'][0].set_linewidth(2)
    
    # Jittered points (rain)
    jitter = np.random.uniform(-0.1, 0.1, size=len(data))
    ax.scatter(np.full(len(data), pos - 0.2) + jitter, data, 
               alpha=0.3, s=10, c=color, zorder=1)
    
    # Add statistics text
    median = np.median(data)
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    
    # 95% CI for mean
    mean = np.mean(data)
    se = stats.sem(data)
    ci_low, ci_high = stats.t.interval(0.95, len(data)-1, loc=mean, scale=se)
    
    # Add annotation
    stats_text = f'Median: {median:.2f}\nIQR: {iqr:.2f}\n95% CI: [{ci_low:.2f}, {ci_high:.2f}]'
    ax.annotate(stats_text, xy=(pos + 0.4, median), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xticks(positions)
ax.set_xticklabels(classes)
ax.set_ylabel(anova_feature, fontsize=12)
ax.set_xlabel('Precip Type', fontsize=12)
ax.set_title(f'Raincloud Plot: {anova_feature} по классам Precip Type\n'
             f'(ANOVA: F = {f_stat:.2f}, p = {p_value:.2e})', fontsize=14)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=rain_colors[i], alpha=0.5, label=c) 
                   for i, c in enumerate(classes)]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('/workspace/output/anova_raincloud.png', dpi=150, bbox_inches='tight')
plt.close()
print("Raincloud plot сохранен в /workspace/output/anova_raincloud.png")

# Alternative: More detailed raincloud using ptitprince if available
try:
    import ptitprince as pt
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create raincloud plot using ptitprince
    pt.RainCloud(x=target, y=anova_feature, data=df_clean, 
                 palette=rain_colors, bw=.2, width_viol=.6, 
                 ax=ax, orient='h', alpha=.65, dodge=True,
                 pointplot=True, move=.2)
    
    ax.set_title(f'Raincloud Plot (ptitprince): {anova_feature} по классам Precip Type\n'
                 f'(ANOVA: F = {f_stat:.2f}, p = {p_value:.2e})', fontsize=14)
    
    # Add statistics annotations
    for i, precip_type in enumerate(classes):
        data = df_clean[df_clean[target] == precip_type][anova_feature].values
        median = np.median(data)
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        mean = np.mean(data)
        se = stats.sem(data)
        ci_low, ci_high = stats.t.interval(0.95, len(data)-1, loc=mean, scale=se)
        
        print(f"\n{precip_type}:")
        print(f"  Медиана: {median:.2f}")
        print(f"  IQR: {iqr:.2f}")
        print(f"  95% CI для среднего: [{ci_low:.2f}, {ci_high:.2f}]")
    
    plt.tight_layout()
    plt.savefig('/workspace/output/anova_raincloud_ptitprince.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nRaincloud plot (ptitprince) сохранен в /workspace/output/anova_raincloud_ptitprince.png")
    
except ImportError:
    print("ptitprince не установлен, используется ручная реализация raincloud")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("АНАЛИЗ ЗАВЕРШЕН")
print("=" * 80)
print("\nСозданные файлы:")
print("  - /workspace/output/pairplot.png")
print("  - /workspace/output/pairplot_detailed.png")
print("  - /workspace/output/correlation_heatmaps_all.png")
for c in classes:
    print(f"  - /workspace/output/correlation_heatmaps_{c}.png")
print("  - /workspace/output/regression_prediction_intervals.png")
print("  - /workspace/output/regression_confidence_intervals.png")
print("  - /workspace/output/regression_summary.csv")
print("  - /workspace/output/anova_raincloud.png")
