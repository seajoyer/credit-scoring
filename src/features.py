import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cap_outliers(data, feature,
                 lower_percentile=None,
                 upper_percentile=None,
                 known_limits=None):

    if lower_percentile is None and upper_percentile is None:
        raise ValueError("At least one limit must be specified: upper_percentile or lower_percentile")
    if lower_percentile is not None and (lower_percentile < 0 or lower_percentile > 1):
        raise ValueError("lower_percentile must be between 0 and 1")
    if upper_percentile is not None and (upper_percentile < 0 or upper_percentile > 1):
        raise ValueError("upper_percentile must be between 0 and 1")
    if feature not in data.columns:
        raise KeyError(f"Feature '{feature}' not found in data")
        
    data = data.copy()

    if known_limits:
        lower, upper = known_limits
    else:
        lower = data[feature].quantile(lower_percentile) if lower_percentile is not None else None
        upper = data[feature].quantile(upper_percentile) if upper_percentile is not None else None

    data[feature] = data[feature].clip(lower=lower, upper=upper)

    outliers_capped = ((data[feature] == lower) | (data[feature] == upper)).sum()

    lower_str = f"lower={lower:.4f}" if lower is not None else "lower=None"
    upper_str = f"upper={upper:.4f}" if upper is not None else "upper=None"
    
    print(f"Capped {outliers_capped} outliers in `{feature}` ({lower_str}, {upper_str})")
    
    return data


def plot_log_odds(data, feature, target,
                  clip_min=None, clip_max=None,
                  bins=20,       epsilon=1e-6,
                  xlabel=None,   ylabel='Log-Odds',
                  title=None,    ax=None,
                  color=None,    edgecolors='black'):
    
    data_ = data[[feature, target]].copy()

    feature_clipped = data_[feature]
    
    if clip_min is not None:
        feature_clipped = np.maximum(feature_clipped, clip_min)
    else:
        clip_min = feature_clipped.min()
        
    if clip_max is not None:
        feature_clipped = np.minimum(feature_clipped, clip_max)
    else:
        clip_max = feature_clipped.max()

    if not isinstance(bins, list):
        bins = np.linspace(clip_min, clip_max, bins + 1)
    
    data_['feature_binned'] = pd.cut(feature_clipped, bins=bins)
    
    bin_stats = data_.groupby('feature_binned', observed=True).agg(
        probability=(target, 'mean'),
        count=(target, 'count')
    ).reset_index()
    
    bin_midpoints = [interval.mid for interval in bin_stats['feature_binned']]
    probs_safe = np.clip(bin_stats['probability'], epsilon, 1 - epsilon)
    log_odds = np.log(probs_safe / (1 - probs_safe))
    
    min_size, max_size = 50, 500
    counts = bin_stats['count'].values
    sizes = min_size + (counts - counts.min()) / (counts.max() - counts.min() + 1e-6) * (max_size - min_size)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    scatter = ax.scatter(bin_midpoints, log_odds, s=sizes, alpha=0.6, 
                        edgecolors=edgecolors, color=color, linewidths=0.5)

    ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel is not None else feature)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    scatter_color = scatter.get_facecolors()[0]
    
    legend_counts = np.linspace(counts.min(), counts.max(), 4).astype(int)
    legend_sizes = min_size + (legend_counts - counts.min()) / (counts.max() - counts.min() + 1e-6) * (max_size - min_size)
    
    legend_handles = [plt.scatter([], [], s=size, color=scatter_color, alpha=0.6, 
                                 edgecolors='black', linewidths=0.5) 
                      for size in legend_sizes]
    ax.legend(legend_handles, [f'{count:,}' for count in legend_counts], 
              title="Bin Count", loc='best', framealpha=0.9, scatterpoints=1, 
              labelspacing=1.3, borderpad=1.2)
    
    return ax, bin_stats


def plot_WoE(data, feature, target,
             clip_min=None, clip_max=None,
             bins=20,       epsilon=1e-6,
             xlabel=None,   ylabel='Weight of Evidence',
             title=None,    ax=None,
             color=None,    edgecolors='black'):
    
    data_ = data[[feature, target]].copy()

    feature_clipped = data_[feature]
    
    if clip_min is not None:
        feature_clipped = np.maximum(feature_clipped, clip_min)
    else:
        clip_min = feature_clipped.min()
        
    if clip_max is not None:
        feature_clipped = np.minimum(feature_clipped, clip_max)
    else:
        clip_max = feature_clipped.max()

    if not isinstance(bins, list):
        bins = np.linspace(clip_min, clip_max, bins + 1)
    
    data_['feature_binned'] = pd.cut(feature_clipped, bins=bins)
    
    bin_stats = data_.groupby('feature_binned', observed=True).agg(
        bad   = (target, 'sum'),
        count = (target, 'count')
    ).reset_index()
    bin_stats['good'] = bin_stats['count'] - bin_stats['bad']
    
    total_bad = data_[target].sum()
    total_good = data_[target].count() - total_bad
    
    bin_stats['bad_pct'] = (bin_stats['bad'] + epsilon) / (total_bad + epsilon * len(bin_stats))
    bin_stats['good_pct'] = (bin_stats['good'] + epsilon) / (total_good + epsilon * len(bin_stats))
    
    bin_midpoints = [interval.mid for interval in bin_stats['feature_binned']]
    
    WoE = np.log(bin_stats['good_pct'] / bin_stats['bad_pct'])
    
    min_size, max_size = 50, 500
    counts = bin_stats['count'].values
    sizes = min_size + (counts - counts.min()) / (counts.max() - counts.min() + 1e-6) * (max_size - min_size)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    scatter = ax.scatter(bin_midpoints, WoE, s=sizes, alpha=0.6, 
                         edgecolors=edgecolors, linewidths=0.5, color=color)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel is not None else feature)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    scatter_color = scatter.get_facecolors()[0]
    
    legend_counts = np.linspace(counts.min(), counts.max(), 4).astype(int)
    legend_sizes = min_size + (legend_counts - counts.min()) / (counts.max() - counts.min() + 1e-6) * (max_size - min_size)
    
    legend_handles = [plt.scatter([], [], s=size, color=scatter_color, alpha=0.6, 
                                 edgecolors='black', linewidths=0.5) 
                      for size in legend_sizes]
    ax.legend(legend_handles, [f'{count:,}' for count in legend_counts], 
              title="Bin Count", loc='best', framealpha=0.9, scatterpoints=1, 
              labelspacing=1.3, borderpad=1.2)
    
    return ax, bin_stats
