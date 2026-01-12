import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Dict

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

try:
    from optbinning import OptimalBinning
    OPTBINNING_AVAILABLE = True
except ImportError:
    OPTBINNING_AVAILABLE = False

from src.config import Config

class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clip outliers based on thresholds.
    """
    def __init__(self, 
                 age_lower: int = 21, 
                 age_upper: int = 87, 
                 debt_ratio_upper: float = 2.0,
                 utilization_upper: float = 1.5,
                 income_upper_percentile: float = 95.0):
        self.age_lower = age_lower
        self.age_upper = age_upper
        self.debt_ratio_upper = debt_ratio_upper
        self.utilization_upper = utilization_upper
        self.income_upper_percentile = income_upper_percentile
        self.clip_params_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        # Store dynamic thresholds
        if 'MonthlyIncome' in X.columns:
            upper = np.nanpercentile(X['MonthlyIncome'], self.income_upper_percentile)
            self.clip_params_['MonthlyIncome'] = {'upper': upper}
        return self

    def transform(self, X):
        X = X.copy()
        # Static Clipping
        if 'age' in X.columns:
            X['age'] = X['age'].clip(lower=self.age_lower, upper=self.age_upper)
        if 'DebtRatio' in X.columns:
            X['DebtRatio'] = X['DebtRatio'].clip(upper=self.debt_ratio_upper)
        if 'RevolvingUtilizationOfUnsecuredLines' in X.columns:
            col = 'RevolvingUtilizationOfUnsecuredLines'
            X[col] = X[col].clip(upper=self.utilization_upper)
            
        # Dynamic Clipping
        for col, params in self.clip_params_.items():
            if col in X.columns:
                X[col] = X[col].clip(upper=params['upper'])
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features

class ClippingIndicator(BaseEstimator, TransformerMixin):
    """Add binary indicators for clipped values."""
    def __init__(self, age_lower: int = 21, age_upper: int = 87, debt_ratio_upper: float = 2.0):
        self.age_lower = age_lower
        self.age_upper = age_upper
        self.debt_ratio_upper = debt_ratio_upper
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        if 'age' in X.columns:
            X['age_clipped_young'] = (X['age'] < self.age_lower).astype(int)
            X['age_clipped_old'] = (X['age'] > self.age_upper).astype(int)
        if 'DebtRatio' in X.columns:
            X['debt_ratio_clipped'] = (X['DebtRatio'] > self.debt_ratio_upper).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        features = list(input_features) if input_features is not None else []
        return np.array(features + ['age_clipped_young', 'age_clipped_old', 'debt_ratio_clipped'])

class CreditFeatureAdder(BaseEstimator, TransformerMixin):
    """Create new domain features."""
    def __init__(self, 
                 add_age_buckets: bool = True,
                 add_debt_income_ratio: bool = True, 
                 add_utilization_squared: bool = True,
                 add_delinquency_features: bool = True):
        self.add_age_buckets = add_age_buckets
        self.add_debt_income_ratio = add_debt_income_ratio
        self.add_utilization_squared = add_utilization_squared
        self.add_delinquency_features = add_delinquency_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        if self.add_debt_income_ratio and 'DebtRatio' in X.columns and 'MonthlyIncome' in X.columns:
            X['debt_income_ratio'] = X['DebtRatio'] * X['MonthlyIncome']
            
        if self.add_age_buckets and 'age' in X.columns:
            X['is_young'] = (X['age'] < 30).astype(int)
            X['is_senior'] = (X['age'] >= 60).astype(int)
            
        if self.add_utilization_squared and 'RevolvingUtilizationOfUnsecuredLines' in X.columns:
            X['utilization_squared'] = X['RevolvingUtilizationOfUnsecuredLines'] ** 2
            
        if self.add_delinquency_features:
            delinq_cols = ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse']
            available = [c for c in delinq_cols if c in X.columns]
            if available:
                X['total_delinquencies'] = X[available].sum(axis=1)
                X['has_delinquency'] = (X['total_delinquencies'] > 0).astype(int)
        
        return X

class CustomImputer(BaseEstimator, TransformerMixin):
    """Handle missing values with configurable strategies."""
    def __init__(self, 
                 income_imputation: str = 'median', 
                 dependents_imputation: str = 'constant'):
        self.income_imputation = income_imputation
        self.dependents_imputation = dependents_imputation
        self.imputers_ = {}

    def _get_imputer(self, strategy: str) -> SimpleImputer:
        """Helper to create imputer based on strategy string."""
        if strategy == 'mode':
            return SimpleImputer(strategy='most_frequent')
        elif strategy == 'zero' or strategy == 'constant':
            return SimpleImputer(strategy='constant', fill_value=0)
        else:
            # Passes 'mean', 'median', 'most_frequent' directly
            return SimpleImputer(strategy=strategy)

    def fit(self, X, y=None):
        # MonthlyIncome
        if 'MonthlyIncome' in X.columns:
            self.imputers_['MonthlyIncome'] = self._get_imputer(self.income_imputation)
            self.imputers_['MonthlyIncome'].fit(X[['MonthlyIncome']])
            
        # Dependents
        if 'NumberOfDependents' in X.columns:
            self.imputers_['NumberOfDependents'] = self._get_imputer(self.dependents_imputation)
            self.imputers_['NumberOfDependents'].fit(X[['NumberOfDependents']])
        return self

    def transform(self, X):
        X = X.copy()
        for col, imputer in self.imputers_.items():
            if col in X.columns:
                X[col] = imputer.transform(X[[col]]).ravel()
        return X

class WoEBinningTransformer(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence Binning Wrapper for OptBinning.
    """
    def __init__(self, numeric_features: List[str], enabled: bool = False):
        self.numeric_features = numeric_features
        self.enabled = enabled
        self.binners_ = {}

        if self.enabled and not OPTBINNING_AVAILABLE:
            warnings.warn(
                "WoEBinningTransformer is enabled but optbinning is not installed."
                "Features will be passed through unchanged.",
                UserWarning
            )
        
    def fit(self, X, y=None):
        if not self.enabled or not OPTBINNING_AVAILABLE:
            return self
            
        for col in self.numeric_features:
            if col in X.columns:
                optb = OptimalBinning(name=col, dtype="numerical", solver="cp")
                optb.fit(X[col], y)
                self.binners_[col] = optb
        return self
        
    def transform(self, X):
        if not self.enabled or not self.binners_:
            return X
        
        X = X.copy()
        for col, binner in self.binners_.items():
            if col in X.columns:
                X[col] = binner.transform(X[col], metric="woe")
        return X

def create_preprocessing_pipeline(config: Config) -> Pipeline:
    """
    Factory to create the pipeline with initial defaults from config.
    """
    fe = config.feature_engineering
    
    pipeline = Pipeline([
        ('indicators', ClippingIndicator(
            age_lower=fe.age_lower, 
            age_upper=fe.age_upper,
            debt_ratio_upper=fe.debt_ratio_upper
        )),
        ('clipper', OutlierClipper(
            age_lower=fe.age_lower,
            age_upper=fe.age_upper,
            debt_ratio_upper=fe.debt_ratio_upper,
            utilization_upper=fe.utilization_upper,
            income_upper_percentile=fe.income_upper_percentile
        )),
        ('imputer', CustomImputer(
            income_imputation=fe.income_imputation,
            dependents_imputation=fe.dependents_imputation
        )),
        ('feature_adder', CreditFeatureAdder(
            add_age_buckets=fe.add_age_buckets,
            add_debt_income_ratio=fe.add_debt_income_ratio,
            add_utilization_squared=fe.add_utilization_squared,
            add_delinquency_features=fe.add_delinquency_features
        )),
        ('woe_binning', WoEBinningTransformer(
            numeric_features=config.data.numeric_features,
            enabled=fe.use_woe_binning
        ))
    ])
    
    if fe.scale_features:
        if fe.scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif fe.scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        pipeline.steps.append(('scaler', scaler))
        
    return pipeline












# ============================================================================
# Helping Functions
# ============================================================================

def apply_woe_binning(df, feature, target, 
                     max_n_bins=10, monotonic_trend='auto',
                     min_prebin_size=0.05, special_codes=None, 
                     binning_obj=None, verbose=True):
    """
    Apply WOE binning to a feature using OptimalBinning.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe containing the feature
    feature : str
        Name of the feature to bin
    target : Series or array
        Target variable (only needed if binning_obj is None)
    max_n_bins : int, default=10
        Maximum number of bins
    monotonic_trend : str, default='auto'
        Monotonic trend constraint ('auto', 'ascending', 'descending', None)
    min_prebin_size : float, default=0.05
        Minimum fraction of total observations in each pre-bin (0.05 = 5%)
        Helps prevent bins with too few observations, improving WOE stability
    special_codes : list or None, default=None
        List of special values to treat as separate bins (e.g., [-999, -1])
        Common in credit scoring for missing values or special indicators
    binning_obj : OptimalBinning object, default=None
        Pre-fitted binning object for transform only (e.g., for test data)
    verbose : bool, default=True
        Whether to print binning diagnostics
    
    Returns:
    --------
    woe_values : array
        WOE-transformed feature values
    binning_obj : OptimalBinning object
        Fitted binning object (can be reused on test/validation data)
    """
    if binning_obj is None:
        binning_obj = OptimalBinning(
            name=feature,
            max_n_bins=max_n_bins,
            monotonic_trend=monotonic_trend,
            min_prebin_size=min_prebin_size,
            special_codes=special_codes
        )
        woe_values = binning_obj.fit_transform(df[feature], target, metric='woe')
        
        if verbose:
            print(f'Feature: {feature}')
            print(f'Binning status: {binning_obj.status}, N bins: {len(binning_obj.splits)}')
            np.set_printoptions(suppress=True)
            print(f'Bins: {binning_obj.splits}\n')
            np.set_printoptions(suppress=False)
    else:
        woe_values = binning_obj.transform(df[feature], metric='woe')
        
        if verbose:
            print(f'Feature: {feature} - transformed using existing binning')
    
    return woe_values, binning_obj



def plot_feature_analysis(df, feature, target=None, target_name='SeriousDlqin2yrs',
                          n_quantiles=40, max_value=None, upper_percentile=None,
                          discrete=False, figsize=(6, 4), ax=None, title=None):
    """
    Create distribution and default rate analysis plot for a feature.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (features only, or including target)
    feature : str
        Name of the feature to analyze
    target : pd.Series or array-like, optional
        Target variable as a separate series. If None, looks for target_name in df
    target_name : str, default='SeriousDlqin2yrs'
        Name of the target variable (used when target is in df or for labeling)
    n_quantiles : int, default=40
        Number of quantiles for binning (ignored if discrete=True)
    max_value : float, optional
        Explicit upper threshold for filtering (takes precedence over upper_percentile)
    upper_percentile : float, optional
        Upper percentile threshold for filtering (e.g., 0.99 for 99th percentile)
    discrete : bool, default=False
        If True, treats feature as discrete/categorical (no binning)
    figsize : tuple, default=(6, 4)
        Figure size (only used if ax is None)
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates a new figure
        
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes object used for plotting
    """
    # Handle target: either passed separately or in dataframe
    if target is not None:
        data = df[[feature]].copy()
        data[target_name] = target
    else:
        data = df[[feature, target_name]].copy()
    
    if max_value is not None:
        upper = max_value
    elif upper_percentile is not None:
        upper = data[feature].quantile(upper_percentile)
    else:
        upper = data[feature].max()
    
    plot_data = data.loc[data[feature].le(upper)].copy()
    
    if discrete:
        default_rate = plot_data.groupby(feature)[target_name].mean().reset_index()
        default_rate = default_rate.rename(columns={target_name: 'DefaultRate'})
        x_col = feature
    else:
        plot_data['bin_feature'] = pd.qcut(plot_data[feature], q=n_quantiles, duplicates='drop')
        plot_data['bin_feature_mid'] = pd.to_numeric(plot_data.bin_feature.apply(lambda x: x.mid))
        
        default_rate = plot_data.groupby('bin_feature_mid')[target_name].mean().reset_index()
        default_rate = default_rate.rename(columns={target_name: 'DefaultRate'})
        x_col = 'bin_feature_mid'
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    sns.histplot(x=plot_data[feature], element='step', discrete=discrete, ax=ax)
    ax.set_xlabel(feature)
    
    ax_twin = ax.twinx()
    sns.lineplot(default_rate, x=x_col, y='DefaultRate',
                 color='orange', linewidth=2, marker='o' if discrete else None, ax=ax_twin)
    ax_twin.set_ylabel('Default Rate', color='orange')
    ax_twin.grid(True, alpha=0.6)

    if title is None:
        threshold_text = f'{upper:.2f}' if upper < 1000 else f'{upper:.0f}'
        feature_type = 'discrete' if discrete else 'continuous'
        ax.set_title(rf'{feature} ({feature_type}, filtered $\leq {threshold_text}$)')
    else:
        ax.set_title(rf'{title}')
    
    return ax




def cap_outliers(df, feature,
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
        
    data = df.copy()

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



def plot_log_odds(df, feature,   target='SeriousDlqin2yrs',
                  clip_min=None, clip_max=None,
                  bins=20,       epsilon=1e-6,
                  xlabel=None,   ylabel='Log-Odds',
                  title=None,    ax=None,
                  color=None,    edgecolors='black'):
    
    data = df[[feature, target]].copy()

    feature_clipped = data[feature]
    
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
    
    data['feature_binned'] = pd.cut(feature_clipped, bins=bins)
    
    bin_stats = data.groupby('feature_binned', observed=True).agg(
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


    
def plot_WoE(df, feature,   target='SeriousDlqin2yrs',
             clip_min=None, clip_max=None,
             bins=20,       epsilon=1e-6,
             xlabel=None,   ylabel='Weight of Evidence',
             title=None,    ax=None,
             color=None,    edgecolors='black'):
    
    data = df[[feature, target]].copy()

    feature_clipped = data[feature]
    
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
    
    data['feature_binned'] = pd.cut(feature_clipped, bins=bins)
    
    bin_stats = data.groupby('feature_binned', observed=True).agg(
        bad   = (target, 'sum'),
        count = (target, 'count')
    ).reset_index()
    bin_stats['good'] = bin_stats['count'] - bin_stats['bad']
    
    total_bad = data[target].sum()
    total_good = data[target].count() - total_bad
    
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
