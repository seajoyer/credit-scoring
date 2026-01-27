import joblib
import numpy as np
import warnings
from typing import List, Optional, Dict
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
    from optbinning import OptimalBinning
    OPTBINNING_AVAILABLE = True
except ImportError:
    OPTBINNING_AVAILABLE = False

from .config import (
    Config,
    OutlierClippingConfig,
    ImputationConfig,
    FeatureCreationConfig,
    WoEConfig
)


class SchemaValidator(BaseEstimator, TransformerMixin):
    """
    Validate input data schema and enforce column order.
    
    Attributes:
        expected_features: List of required feature names
        enforce_order: Whether to enforce column ordering
    """
    
    def __init__(self, expected_features: List[str], enforce_order: bool = True):
        self.expected_features = expected_features
        self.enforce_order = enforce_order
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        expected_set = set(self.expected_features)
        actual_set = set(X.columns)
        
        missing = expected_set - actual_set
        if missing:
            raise ValueError(
                f"Missing required features: {sorted(missing)}\n"
                f"Expected: {sorted(self.expected_features)}\n"
                f"Got: {sorted(X.columns)}"
            )
        
        extra = actual_set - expected_set
        if extra:
            warnings.warn(
                f"Unexpected features (will be dropped): {sorted(extra)}",
                UserWarning
            )
        
        return X[self.expected_features]
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.expected_features)


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clip outliers based on configuration.
    
    Attributes:
        config: OutlierClippingConfig object
        fitted_thresholds_: Dict of learned threshold values (fitted)
    """
    
    def __init__(self, config: OutlierClippingConfig):
        self.config = config
        self.fitted_thresholds_ = {}
    
    def fit(self, X, y=None):
        """Learn dynamic thresholds from training data."""
        X = X.copy()
        
        if 'MonthlyIncome' in X.columns:
            income_upper = np.nanpercentile(
                X['MonthlyIncome'], 
                self.config.income_upper_percentile
            )
            self.fitted_thresholds_['MonthlyIncome'] = income_upper
        
        self.fitted_thresholds_['age_lower'] = self.config.age_bounds[0]
        self.fitted_thresholds_['age_upper'] = self.config.age_bounds[1]
        self.fitted_thresholds_['debt_ratio_upper']  = self.config.debt_ratio_upper
        self.fitted_thresholds_['utilization_upper'] = self.config.utilization_upper
        
        return self
    
    def transform(self, X):
        """Apply learned thresholds."""
        if not self.fitted_thresholds_:
            raise ValueError("Transformer not fitted. Call fit() first.")
        
        X = X.copy()
        
        if 'age' in X.columns:
            X['age'] = X['age'].clip(
                lower=self.fitted_thresholds_['age_lower'],
                upper=self.fitted_thresholds_['age_upper']
            )
        
        if 'DebtRatio' in X.columns:
            X['DebtRatio'] = X['DebtRatio'].clip(
                upper=self.fitted_thresholds_['debt_ratio_upper']
            )
        
        if 'RevolvingUtilizationOfUnsecuredLines' in X.columns:
            X['RevolvingUtilizationOfUnsecuredLines'] = X['RevolvingUtilizationOfUnsecuredLines'].clip(
                upper=self.fitted_thresholds_['utilization_upper']
            )
        
        if 'MonthlyIncome' in X.columns and 'MonthlyIncome' in self.fitted_thresholds_:
            X['MonthlyIncome'] = X['MonthlyIncome'].clip(
                upper=self.fitted_thresholds_['MonthlyIncome']
            )
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names unchanged."""
        return input_features if input_features is not None else []
    
    def get_fitted_info(self) -> Dict:
        """Return fitted thresholds for inspection/logging."""
        return self.fitted_thresholds_.copy()


class ClippingIndicator(BaseEstimator, TransformerMixin):
    """
    Add binary indicators for clipped values.
    
    Attributes:
        config: OutlierClippingConfig object
    """
    
    def __init__(self, config: OutlierClippingConfig):
        self.config = config
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if 'age' in X.columns:
            X['age_clipped_young'] = (X['age'] <= self.config.age_bounds[0]).astype(int)
            X['age_clipped_old']   = (X['age'] >= self.config.age_bounds[1]).astype(int)
        
        if 'DebtRatio' in X.columns:
            X['debt_ratio_clipped'] = (X['DebtRatio'] >= self.config.debt_ratio_upper).astype(int)
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        features = list(input_features) if input_features is not None else []
        new_features = []
        
        if 'age' in features:
            new_features.extend(['age_clipped_young', 'age_clipped_old'])
        if 'DebtRatio' in features:
            new_features.append('debt_ratio_clipped')
        
        return np.array(features + new_features)


class CustomImputer(BaseEstimator, TransformerMixin):
    """
    Handle missing values with configurable strategies.
    
    Attributes:
        config: ImputationConfig object
        fitted_imputers_: Dict of fitted SimpleImputer objects (fitted)
    """
    
    def __init__(self, config: ImputationConfig):
        self.config = config
        self.fitted_imputers_ = {}
    
    def _create_imputer(self, strategy: str, fill_value: float = 0) -> SimpleImputer:
        """Factory method to create imputer from strategy string."""
        if strategy == 'constant':
            return SimpleImputer(strategy='constant', fill_value=fill_value)
        elif strategy == 'most_frequent':
            return SimpleImputer(strategy='most_frequent')
        else:
            return SimpleImputer(strategy=strategy)
    
    def fit(self, X, y=None):
        """Fit imputers for each column based on config."""
        if 'MonthlyIncome' in X.columns:
            self.fitted_imputers_['MonthlyIncome'] = self._create_imputer(
                self.config.income_strategy,
                self.config.constant_fill_value
            )
            self.fitted_imputers_['MonthlyIncome'].fit(X[['MonthlyIncome']])
        
        if 'NumberOfDependents' in X.columns:
            self.fitted_imputers_['NumberOfDependents'] = self._create_imputer(
                self.config.dependents_strategy,
                self.config.constant_fill_value
            )
            self.fitted_imputers_['NumberOfDependents'].fit(X[['NumberOfDependents']])
        
        return self
    
    def transform(self, X):
        """Apply fitted imputers."""
        if not self.fitted_imputers_:
            raise ValueError("Transformer not fitted. Call fit() first.")
        
        X = X.copy()
        for col, imputer in self.fitted_imputers_.items():
            if col in X.columns:
                X[col] = imputer.transform(X[[col]]).ravel()
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else []
    
    def get_fitted_info(self) -> Dict:
        """Return fitted imputer statistics."""
        info = {}
        for col, imputer in self.fitted_imputers_.items():
            if hasattr(imputer, 'statistics_'):
                info[col] = {
                    'strategy': imputer.strategy,
                    'fill_value': float(imputer.statistics_[0])
                }
        return info


class FeatureAdder(BaseEstimator, TransformerMixin):
    """
    Create domain-specific features.
    
    Attributes:
        config: FeatureCreationConfig object
    """
    
    def __init__(self, config: FeatureCreationConfig):
        self.config = config
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Age buckets
        if self.config.add_age_buckets and 'age' in X.columns:
            X['is_young'] = (X['age'] < self.config.age_young_threshold).astype(int)
            X['is_senior'] = (X['age'] >= self.config.age_senior_threshold).astype(int)
        
        # Debt-income interaction
        if self.config.add_debt_income_ratio:
            if 'DebtRatio' in X.columns and 'MonthlyIncome' in X.columns:
                X['debt_income_ratio'] = X['DebtRatio'] * X['MonthlyIncome']
        
        # Utilization squared
        if self.config.add_utilization_squared:
            if 'RevolvingUtilizationOfUnsecuredLines' in X.columns:
                X['utilization_squared'] = X['RevolvingUtilizationOfUnsecuredLines'] ** 2
        
        # Delinquency features
        if self.config.add_delinquency_features:
            delinq_cols = [
                'NumberOfTime30-59DaysPastDueNotWorse',
                'NumberOfTimes90DaysLate',
                'NumberOfTime60-89DaysPastDueNotWorse'
            ]
            available = [c for c in delinq_cols if c in X.columns]
            if available:
                X['total_delinquencies'] = X[available].sum(axis=1)
                X['has_delinquency'] = (X['total_delinquencies'] > 0).astype(int)
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Return all feature names including newly created ones."""
        features = list(input_features) if input_features is not None else []
        
        if self.config.add_age_buckets:
            features.extend(['is_young', 'is_senior'])
        if self.config.add_debt_income_ratio:
            features.append('debt_income_ratio')
        if self.config.add_utilization_squared:
            features.append('utilization_squared')
        if self.config.add_delinquency_features:
            features.extend(['total_delinquencies', 'has_delinquency'])
        
        return np.array(features)


class WoEBinningTransformer(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence Binning wrapper for OptBinning.
    
    Requires optbinning package to be installed.
    
    Attributes:
        config: WoEConfig object
        numeric_features: List of numeric features to potentially bin
        binners_: Dict of fitted OptimalBinning objects (fitted)
    """
    
    def __init__(self, config: WoEConfig, numeric_features: List[str]):
        self.config = config
        self.numeric_features = numeric_features
        self.binners_ = {}
        
        if self.config.enabled and not OPTBINNING_AVAILABLE:
            warnings.warn(
                "WoEBinningTransformer is enabled but optbinning is not installed. "
                "Features will be passed through unchanged. "
                "Install with: pip install optbinning",
                UserWarning
            )
    
    def fit(self, X, y=None):
        if not self.config.enabled or not OPTBINNING_AVAILABLE:
            return self
        
        if y is None:
            warnings.warn(
                "WoE binning requires target variable y but none provided. "
                "Skipping WoE binning.",
                UserWarning
            )
            return self
        
        features_to_bin = self.config.features if self.config.features else self.numeric_features
        
        for col in features_to_bin:
            if col in X.columns:
                try:
                    optb = OptimalBinning(
                        name=col,
                        dtype="numerical",
                        solver=self.config.solver
                    )
                    optb.fit(X[col], y)
                    self.binners_[col] = optb
                except Exception as e:
                    warnings.warn(
                        f"Failed to fit WoE binning for {col}: {e}. Skipping this feature.",
                        UserWarning
                    )
        
        return self
    
    def transform(self, X):
        if not self.config.enabled or not self.binners_:
            return X
        
        X = X.copy()
        for col, binner in self.binners_.items():
            if col in X.columns:
                try:
                    X[col] = binner.transform(X[col], metric="woe")
                except Exception as e:
                    warnings.warn(
                        f"Failed to transform {col} with WoE binning: {e}. "
                        f"Keeping original values.",
                        UserWarning
                    )
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else []


def create_validation_pipeline(expected_features: List[str]) -> Pipeline:
    """
    Create validation pipeline.
    
    Args:
        expected_features: List of expected feature names
    
    Returns:
        Pipeline with schema validation
    """
    return Pipeline([
        ('schema', SchemaValidator(expected_features, enforce_order=True))
    ])


def create_cleaning_pipeline(config: Config) -> Pipeline:
    """
    Create cleaning pipeline.
    
    Args:
        config: Configuration object
    
    Returns:
        Pipeline with outlier clipping and imputation
    """
    pp = config.preprocessing
    
    return Pipeline([
        ('clipper', OutlierClipper(pp.outlier_clipping)),
        ('imputer', CustomImputer(pp.imputation))
    ])


def create_feature_pipeline(config: Config) -> Pipeline:
    """
    Create feature engineering pipeline.
    
    Args:
        config: Configuration object
    
    Returns:
        Pipeline with feature creation and optional WoE binning
    """
    pp = config.preprocessing
    
    steps = [
        ('feature_adder', FeatureAdder(pp.feature_creation))
    ]
    
    if pp.woe.enabled:
        steps.append(
            ('woe', WoEBinningTransformer(pp.woe, config.data.numeric_features))
        )
    
    return Pipeline(steps)


def create_scaling_pipeline(config: Config) -> Pipeline:
    """
    Create scaling pipeline.
    
    Args:
        config: Configuration object
    
    Returns:
        Pipeline with feature scaling
    """
    from sklearn.preprocessing import FunctionTransformer
    
    pp = config.preprocessing
    
    if not pp.scaling.enabled:
        return Pipeline([('passthrough', FunctionTransformer())])
    
    if pp.scaling.method == 'minmax':
        scaler = MinMaxScaler()
    elif pp.scaling.method == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    return Pipeline([('scaler', scaler)])


def create_preprocessing_pipeline(config: Config) -> Pipeline:
    """
    Create complete preprocessing pipeline with all stages.
    
    Args:
        config: Configuration object
    
    Returns:
        Complete preprocessing pipeline
    """
    pp = config.preprocessing
    
    steps = [
        ('clipper', OutlierClipper(pp.outlier_clipping)),
        ('imputer', CustomImputer(pp.imputation)),
        ('feature_adder', FeatureAdder(pp.feature_creation)),
    ]
    
    if pp.woe.enabled:
        steps.append(
            ('woe', WoEBinningTransformer(pp.woe, config.data.numeric_features))
        )
    
    if pp.scaling.enabled:
        if pp.scaling.method == 'minmax':
            scaler = MinMaxScaler()
        elif pp.scaling.method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        steps.append(('scaler', scaler))
    
    return Pipeline(steps)


def create_full_pipeline(config: Config) -> Pipeline:
    """
    Create full modular pipeline.
    
    Architecture:
    1. Validation: Schema checks
    2. Cleaning: Outlier clipping and imputation
    3. Features: Feature engineering
    4. Scaling: Feature standardization
    
    Args:
        config: Configuration object
    
    Returns:
        Complete modular pipeline
    """
    return Pipeline([
        ('validation', create_validation_pipeline(config.data.numeric_features)),
        ('cleaning', create_cleaning_pipeline(config)),
        ('features', create_feature_pipeline(config)),
        ('scaling', create_scaling_pipeline(config))
    ])


def save_pipeline(pipeline: Pipeline, path: Path, config: Optional[Config] = None):
    """
    Save pipeline with optional configuration.
    
    Args:
        pipeline: Fitted pipeline to save
        path: Path to save to
        config: Optional configuration to save alongside
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(pipeline, path)
    
    if config is not None:
        config_path = path.parent / f"{path.stem}_config.json"
        config.preprocessing.save(config_path)
    
    print(f"Pipeline saved to {path}")


def load_pipeline(path: Path) -> Pipeline:
    """
    Load pipeline from disk.
    
    Args:
        path: Path to load from
    
    Returns:
        Loaded pipeline
    """
    return joblib.load(path)
