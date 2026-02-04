from pyparsing.tools.cvt_pyparsing_pep8_names import special_changes
import warnings
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, FunctionTransformer

try:
    from optbinning import OptimalBinning

    OPTBINNING_AVAILABLE = True
except ImportError:
    OPTBINNING_AVAILABLE = False

from .config import (
    Config,
    FeatureCreationConfig,
    ImputationConfig,
    OutlierClippingConfig,
    WoEConfig,
)


class SchemaValidator(BaseEstimator, TransformerMixin):
    """
    Validate input data schema and enforce column order.

    Attributes:
        expected_features: List of required feature names
        enforce_order: Whether to enforce column ordering
        feature_names_out_: Feature names after transformation (fitted)
    """

    def __init__(self, expected_features: list[str], enforce_order: bool = True):
        self.expected_features = expected_features
        self.enforce_order = enforce_order

    def fit(self, X, y=None):
        """Store the feature names that will be output."""
        self.feature_names_out_ = np.array(self.expected_features, dtype=object)
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
            warnings.warn(f"Unexpected features (will be dropped): {sorted(extra)}", UserWarning, stacklevel=2)

        return X[self.expected_features]

    def get_feature_names_out(self, input_features=None):
        """Return the expected feature names after validation."""
        if hasattr(self, "feature_names_out_"):
            return self.feature_names_out_
        return np.array(self.expected_features, dtype=object)


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clip outliers based on configuration.

    Attributes:
        config: OutlierClippingConfig object
        fitted_thresholds_: Dict of learned threshold values (fitted)
        feature_names_in_: Input feature names (fitted)
        feature_names_out_: Output feature names (fitted)
    """

    def __init__(self, config: OutlierClippingConfig):
        self.config = config
        self.fitted_thresholds_ = {}

    def fit(self, X, y=None):
        """Learn dynamic thresholds from training data."""
        X = X.copy()

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        else:
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])], dtype=object)

        self.feature_names_out_ = self.feature_names_in_.copy()

        if not self.config.enabled:
            return self

        if self.config.clip_age and "age" in X.columns:
            age_upper = np.nanpercentile(X["age"], self.config.age_clip_upper_percentile)
            self.fitted_thresholds_["age_upper"] = age_upper

        if self.config.clip_income and "MonthlyIncome" in X.columns:
            income_upper = np.nanpercentile(X["MonthlyIncome"], self.config.income_clip_upper_percentile)
            self.fitted_thresholds_["MonthlyIncome_upper"] = income_upper

            if self.config.income_strategy == "median":
                self.fitted_thresholds_["MonthlyIncome_median"] = np.nanmedian(X["MonthlyIncome"])

        if (
            self.config.clip_debt_ratio
            and "DebtRatio" in X.columns
            and self.config.debt_ratio_strategy == "custom_median"
        ):
            self.fitted_thresholds_["DebtRatio_custom_median"] = np.nanmedian(
                X.loc[X["DebtRatio"].le(self.config.debt_ratio_clip_upper), "DebtRatio"]
            )

        if self.config.clip_deps and "NumberOfDependents" in X.columns:
            deps_upper = np.nanpercentile(X["NumberOfDependents"], self.config.deps_clip_upper_percentile)
            self.fitted_thresholds_["deps_upper"] = deps_upper

        if self.config.clip_loans and "NumberOfOpenCreditLinesAndLoans" in X.columns:
            loans_upper = np.nanpercentile(
                X["NumberOfOpenCreditLinesAndLoans"], self.config.loans_clip_upper_percentile
            )
            self.fitted_thresholds_["loans_upper"] = loans_upper

        if self.config.clip_estate and "NumberRealEstateLoansOrLines" in X.columns:
            estate_upper = np.nanpercentile(X["NumberRealEstateLoansOrLines"], self.config.estate_clip_upper_percentile)
            self.fitted_thresholds_["estate_upper"] = estate_upper

        if self.config.clip_pd30_59 and "NumberOfTime30-59DaysPastDueNotWorse" in X.columns:
            pd30_59_upper = np.nanpercentile(
                X["NumberOfTime30-59DaysPastDueNotWorse"], self.config.pd30_59_clip_upper_percentile
            )
            self.fitted_thresholds_["pd30_59_upper"] = pd30_59_upper

        if self.config.clip_pd60_89 and "NumberOfTime60-89DaysPastDueNotWorse" in X.columns:
            pd60_89_upper = np.nanpercentile(
                X["NumberOfTime60-89DaysPastDueNotWorse"], self.config.pd60_89_clip_upper_percentile
            )
            self.fitted_thresholds_["pd60_89_upper"] = pd60_89_upper

        if self.config.clip_pd90 and "NumberOfTimes90DaysLate" in X.columns:
            pd90_upper = np.nanpercentile(X["NumberOfTimes90DaysLate"], self.config.pd90_clip_upper_percentile)
            self.fitted_thresholds_["pd90_upper"] = pd90_upper

        return self

    def transform(self, X):
        """Apply learned thresholds."""
        if not self.config.enabled:
            return X

        if not self.fitted_thresholds_:
            raise ValueError("Transformer not fitted. Call fit() first.")

        X = X.copy()

        if self.config.clip_age and "age" in X.columns:
            X["age"] = X["age"].clip(
                lower=self.config.age_clip_lower,
                upper=self.fitted_thresholds_["age_upper"],
            )

        if (
            self.config.clip_income
            and "MonthlyIncome" in X.columns
            and "MonthlyIncome_upper" in self.fitted_thresholds_
        ):
            mask = X["MonthlyIncome"] > self.fitted_thresholds_["MonthlyIncome_upper"]

            if self.config.income_strategy == "clip":
                X["MonthlyIncome"] = X["MonthlyIncome"].clip(upper=self.fitted_thresholds_["MonthlyIncome_upper"])
            elif self.config.income_strategy == "median":
                X.loc[mask, "MonthlyIncome"] = self.fitted_thresholds_["MonthlyIncome_median"]
            elif self.config.income_strategy == "zero":
                X.loc[mask, "MonthlyIncome"] = 0.0

        if self.config.clip_util and "RevolvingUtilizationOfUnsecuredLines" in X.columns:
            X["RevolvingUtilizationOfUnsecuredLines"] = X["RevolvingUtilizationOfUnsecuredLines"].clip(
                upper=self.config.util_clip_upper
            )

        if self.config.clip_debt_ratio and "DebtRatio" in X.columns:
            mask = X["DebtRatio"] > self.config.debt_ratio_clip_upper

            if self.config.debt_ratio_strategy == "clip":
                X["DebtRatio"] = X["DebtRatio"].clip(upper=self.fitted_thresholds_["debt_ratio_upper"])
            elif self.config.debt_ratio_strategy == "custom_median":
                X.loc[mask, "DebtRatio"] = self.fitted_thresholds_["DebtRatio_custom_median"]
            elif self.config.debt_ratio_strategy == "zero":
                X.loc[mask, "DebtRatio"] = 0.0

        if self.config.clip_deps and "NumberOfDependents" in X.columns:
            X["NumberOfDependents"] = X["NumberOfDependents"].clip(
                upper=self.fitted_thresholds_["deps_upper"],
            )

        if self.config.clip_loans and "NumberOfOpenCreditLinesAndLoans" in X.columns:
            X["NumberOfOpenCreditLinesAndLoans"] = X["NumberOfOpenCreditLinesAndLoans"].clip(
                upper=self.fitted_thresholds_["loans_upper"]
            )

        if self.config.clip_estate and "NumberRealEstateLoansOrLines" in X.columns:
            X["NumberRealEstateLoansOrLines"] = X["NumberRealEstateLoansOrLines"].clip(
                upper=self.fitted_thresholds_["estate_upper"]
            )

        if self.config.clip_pd30_59 and "NumberOfTime30-59DaysPastDueNotWorse" in X.columns:
            X["NumberOfTime30-59DaysPastDueNotWorse"] = X["NumberOfTime30-59DaysPastDueNotWorse"].clip(
                upper=self.fitted_thresholds_["pd30_59_upper"]
            )

        if self.config.clip_pd60_89 and "NumberOfTime60-89DaysPastDueNotWorse" in X.columns:
            X["NumberOfTime60-89DaysPastDueNotWorse"] = X["NumberOfTime60-89DaysPastDueNotWorse"].clip(
                upper=self.fitted_thresholds_["pd60_89_upper"]
            )

        if self.config.clip_pd90 and "NumberOfTimes90DaysLate" in X.columns:
            X["NumberOfTimes90DaysLate"] = X["NumberOfTimes90DaysLate"].clip(
                upper=self.fitted_thresholds_["pd90_upper"]
            )

        return X

    def get_feature_names_out(self, input_features=None):
        """Return feature names unchanged."""
        if hasattr(self, "feature_names_out_"):
            return self.feature_names_out_
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.array([], dtype=object)

    def get_fitted_info(self) -> dict:
        """Return fitted thresholds for inspection/logging."""
        return self.fitted_thresholds_.copy()


class IndicatorExtractor(BaseEstimator, TransformerMixin):
    """
    Create binary indicator features.

    Attributes:
        features_to_track: List of feature names to create missing indicators for
        feature_names_in_: Input feature names (fitted)
        feature_names_out_: Output feature names including indicators (fitted)
    """

    def __init__(self, config: FeatureCreationConfig, features_to_track: list[str] | None = None):
        """
        Initialize binary indicator creator.

        Args:
            features_to_track: List of features to track. If None, tracks all features with missing values.
        """
        self.config = config
        self.fitted_thresholds_ = {}
        self.features_to_track = features_to_track

    def fit(self, X, y=None):
        """Learn dynamic thresholds from training data."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        else:
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])], dtype=object)

        if not self.config.enabled:
            self.feature_names_out_ = self.feature_names_in_.copy()
            return self

        if self.features_to_track is None:
            self.features_to_track_ = [col for col in X.columns if X[col].isna().any()]
        else:
            self.features_to_track_ = [col for col in self.features_to_track if col in X.columns]

        if "MonthlyIncome" in X.columns:
            income_upper = np.nanpercentile(X["MonthlyIncome"], self.config.income_upper_percentile)
            self.fitted_thresholds_["income_upper"] = income_upper

            income_low = np.nanpercentile(X["MonthlyIncome"], self.config.income_low_percentile)
            self.fitted_thresholds_["income_low"] = income_low

        if "DebtRatio" in X.columns:
            self.fitted_thresholds_["DebtRatio_custom_median"] = np.nanmedian(
                X.loc[X["DebtRatio"].le(self.config.debt_ratio_clip_upper), "DebtRatio"]
            )

        output_features = list(self.feature_names_in_)
        for col in self.features_to_track_:
            output_features.append(f"{col}_missing")

        self.feature_names_out_ = self._get_output_features(self.feature_names_in_)

        return self

    def transform(self, X):
        """Create binary indicators."""
        if not self.config.enabled:
            return X

        X = X.copy()

        # Missing
        for col in self.features_to_track_:
            if col in X.columns:
                X[f"{col}_missing"] = X[col].isna().astype(int)

        # Age buckets
        if self.config.add_age_buckets and "age" in X.columns:
            X["is_young"] = (X["age"] < self.config.age_young_threshold).astype(int)
            X["is_senior"] = (X["age"] > self.config.age_senior_threshold).astype(int)

        # Income clipped
        if self.config.add_income_clipped and "MonthlyIncome" in X.columns:
            X["MonthlyIncome_clipped"] = X["MonthlyIncome"].gt(self.fitted_thresholds_["income_upper"]).astype(int)

        # Income zero
        if self.config.add_income_zero and "MonthlyIncome" in X.columns:
            X["MonthlyIncome_zero"] = X["MonthlyIncome"].eq(0.0).astype(int)

        # Income low
        if self.config.add_income_low and "MonthlyIncome" in X.columns:
            X["LowIncome"] = X["MonthlyIncome"].le(self.fitted_thresholds_["income_low"]).astype(int)

        # Utilization clipped
        if self.config.add_utilization_clipped and "RevolvingUtilizationOfUnsecuredLines" in X.columns:
            X["utilization_clipped"] = (
                X["RevolvingUtilizationOfUnsecuredLines"].gt(self.config.util_clip_upper).astype(int)
            )

        # Utilization 0.99999990
        if self.config.add_utilization_99999990 and "RevolvingUtilizationOfUnsecuredLines" in X.columns:
            X["utilization_99999990"] = X["RevolvingUtilizationOfUnsecuredLines"].eq(0.9999999).astype(int)

        # DebtRatio buckets
        if self.config.add_debt_ratio_buckets and "DebtRatio" in X.columns:
            X["DebtRatio_low"] = (X["DebtRatio"] < self.fitted_thresholds_["DebtRatio_custom_median"]).astype(int)
            X["DebtRatio_clipped"] = (X["DebtRatio"] > self.config.debt_ratio_clip_upper).astype(int)

        # DebtRatio zero
        if self.config.add_debt_ratio_buckets and "DebtRatio" in X.columns:
            X["DebtRatio_zero"] = X["DebtRatio"].eq(0.0).astype(int)

        # DebtRatio whole
        if self.config.add_debt_ratio_buckets and "DebtRatio" in X.columns:
            X["DebtRatio_whole"] = ((np.isclose(X["DebtRatio"] % 1, 0.0, atol=1e-9)) & (X["DebtRatio"].gt(0.0))).astype(
                int
            )

        return X

    def get_feature_names_out(self, input_features=None):
        """Return all feature names including newly created ones."""
        return self._get_output_features(input_features)

    def _get_output_features(self, input_features):
        """Build the list of output feature names based on config."""
        features = list(input_features) if input_features is not None else []

        # Missing indicators
        for col in self.features_to_track_:
            features.append(f"{col}_missing")

        # Other indicators
        if self.config.add_age_buckets:
            features.extend(["is_young", "is_senior"])

        if self.config.add_income_clipped:
            features.append("MonthlyIncome_clipped")
        if self.config.add_income_zero:
            features.append("MonthlyIncome_zero")
        if self.config.add_income_low:
            features.append("LowIncome")

        if self.config.add_utilization_clipped:
            features.append("utilization_clipped")
        if self.config.add_utilization_99999990:
            features.append("utilization_99999990")

        if self.config.add_debt_ratio_buckets:
            features.extend(["DebtRatio_low", "DebtRatio_clipped"])
        if self.config.add_debt_ratio_zero:
            features.append("DebtRatio_zero")
        if self.config.add_debt_ratio_whole:
            features.append("DebtRatio_whole")

        return np.array(features, dtype=object)


class CustomImputer(BaseEstimator, TransformerMixin):
    """
    Handle missing values with configurable strategies.

    Attributes:
        config: ImputationConfig object
        fitted_imputers_: Dict of fitted SimpleImputer objects (fitted)
        feature_names_in_: Input feature names (fitted)
        feature_names_out_: Output feature names (fitted)
    """

    def __init__(self, config: ImputationConfig):
        self.config = config
        self.fitted_imputers_ = {}

    def _create_imputer(self, strategy: str, fill_value: float = 0) -> SimpleImputer:
        """Factory method to create imputer from strategy string."""
        if strategy == "constant":
            return SimpleImputer(strategy="constant", fill_value=fill_value)
        elif strategy == "most_frequent":
            return SimpleImputer(strategy="most_frequent")
        else:
            return SimpleImputer(strategy=strategy)

    def fit(self, X, y=None):
        """Fit imputers for each column based on config."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        else:
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])], dtype=object)

        if "MonthlyIncome" in X.columns:
            self.fitted_imputers_["MonthlyIncome"] = self._create_imputer(
                self.config.income_strategy, self.config.constant_fill_value
            )
            self.fitted_imputers_["MonthlyIncome"].fit(X[["MonthlyIncome"]])

        if "NumberOfDependents" in X.columns:
            self.fitted_imputers_["NumberOfDependents"] = self._create_imputer(
                self.config.dependents_strategy, self.config.constant_fill_value
            )
            self.fitted_imputers_["NumberOfDependents"].fit(X[["NumberOfDependents"]])

        self.feature_names_out_ = self.feature_names_in_.copy()

        return self

    def transform(self, X):
        """Apply fitted imputers."""
        if not self.fitted_imputers_:
            raise ValueError("Transformer not fitted. Call fit() first.")

        X = X.copy()
        for col, imputer in self.fitted_imputers_.items():
            if col not in self.config.features_skip_imputation and col in X.columns:
                X[col] = imputer.transform(X[[col]]).ravel()

        return X

    def get_feature_names_out(self, input_features=None):
        """Return feature names unchanged."""
        if hasattr(self, "feature_names_out_"):
            return self.feature_names_out_
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.array([], dtype=object)

    def get_fitted_info(self) -> dict:
        """Return fitted imputer statistics."""
        info = {}
        for col, imputer in self.fitted_imputers_.items():
            if hasattr(imputer, "statistics_"):
                info[col] = {
                    "strategy": imputer.strategy,
                    "fill_value": float(imputer.statistics_[0]),
                }
        return info


class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    Create domain-specific features.

    Attributes:
        config: FeatureCreationConfig object
        feature_names_in_: Input feature names (fitted)
        feature_names_out_: Output feature names including created features (fitted)
    """

    def __init__(self, config: FeatureCreationConfig):
        self.config = config
        self.fitted_thresholds_ = {}

    def fit(self, X, y=None):
        """Learn dynamic thresholds from training data."""
        X = X.copy()

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        else:
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])], dtype=object)

        if not self.config.enabled:
            self.feature_names_out_ = self.feature_names_out_.copy()
            return self

        self.feature_names_out_ = self._get_output_features(self.feature_names_in_)

        return self

    def transform(self, X):
        if not self.config.enabled:
            return X

        X = X.copy()

        # Age polynomial
        if self.config.add_age_polynomial and "age" in X.columns:
            X["age^2"] = X["age"] ** 2
            X["age^3"] = X["age"] ** 3

        # DebtRatio polynomial
        if self.config.add_debt_ratio_polynomial and "DebtRatio" in X.columns:
            X["DebtRatio^2"] = X["DebtRatio"] ** 2
            X["DebtRatio^3"] = X["DebtRatio"] ** 3

        # Loans polynomial
        if self.config.add_loans_polynomial and "NumberOfOpenCreditLinesAndLoans" in X.columns:
            X["NumberOfOpenCreditLinesAndLoans^2"] = X["NumberOfOpenCreditLinesAndLoans"] ** 2
            X["NumberOfOpenCreditLinesAndLoans^3"] = X["NumberOfOpenCreditLinesAndLoans"] ** 3

        # Estate polynomial
        if self.config.add_estate_polynomial and "NumberRealEstateLoansOrLines" in X.columns:
            X["NumberRealEstateLoansOrLines^2"] = X["NumberRealEstateLoansOrLines"] ** 2
            X["NumberRealEstateLoansOrLines^3"] = X["NumberRealEstateLoansOrLines"] ** 3

        # Delinquency features
        if self.config.add_delinquency_features:
            delinq_weights = {
                "NumberOfTime30-59DaysPastDueNotWorse": 1,
                "NumberOfTime60-89DaysPastDueNotWorse": 2,
                "NumberOfTimes90DaysLate": 3,
            }

            col_weight_pairs = [(col, weight) for col, weight in delinq_weights.items() if col in X.columns]

            if col_weight_pairs:
                cols, weights = zip(*col_weight_pairs, strict=False)
                X["total_delinquencies"] = X[list(cols)].sum(axis=1)
                X["has_delinquency"] = (X["total_delinquencies"] > 0).astype(int)
                X["weighted_delinquencies"] = X[list(cols)].dot(list(weights))

        return X

    def get_feature_names_out(self, input_features=None):
        """Return all feature names including newly created ones."""
        return self._get_output_features(input_features)

    def _get_output_features(self, input_features):
        """Build the list of output feature names based on config."""
        features = list(input_features) if input_features is not None else []

        if self.config.add_age_polynomial:
            features.extend(["age^2", "age^3"])
        if self.config.add_debt_ratio_polynomial:
            features.extend(["DebtRatio^2", "DebtRatio^3"])
        if self.config.add_loans_polynomial:
            features.extend(["NumberOfOpenCreditLinesAndLoans^2", "NumberOfOpenCreditLinesAndLoans^3"])
        if self.config.add_estate_polynomial:
            features.extend(["NumberRealEstateLoansOrLines^2", "NumberRealEstateLoansOrLines^3"])
        if self.config.add_delinquency_features:
            features.extend(["total_delinquencies", "has_delinquency", "weighted_delinquencies"])

        return np.array(features, dtype=object)


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Apply different transformations to specific features.

    Supports:
    - log1p: np.log1p(x) - safe log transformation for non-negative values
    - yeo-johnson: Power transformation that works with negative values
    - log: np.log(x) - requires positive values
    - sqrt: np.sqrt(x) - requires non-negative values
    - none: No transformation (passthrough)

    Attributes:
        transformations: Dict mapping feature names to transformation types
        yeo_johnson_standardize: Whether to standardize after Yeo-Johnson
        power_transformers_: Dict of fitted PowerTransformer objects (fitted)
        feature_names_in_: Input feature names (fitted)
        feature_names_out_: Output feature names (fitted)
    """

    def __init__(self, config: FeatureCreationConfig):
        self.config = config
        self.power_transformers_ = {}

    def fit(self, X, y=None):
        """Fit Yeo-Johnson transformers if needed."""
        from sklearn.preprocessing import PowerTransformer

        X = X.copy()

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        else:
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])], dtype=object)

        if self.config.enabled:
            for feature, transform_type in self.config.transformations.items():
                if transform_type == "yeo-johnson" and feature in X.columns:
                    pt = PowerTransformer(method="yeo-johnson", standardize=self.config.yeo_johnson_standardize)
                    pt.fit(X[[feature]])
                    self.power_transformers_[feature] = pt

        self.feature_names_out_ = self.feature_names_in_.copy()

        return self

    def transform(self, X):
        """Apply transformations to specified features."""
        if not self.config.enabled:
            return X

        X = X.copy()

        for feature, transform_type in self.config.transformations.items():
            if feature not in X.columns:
                warnings.warn(
                    f"Feature '{feature}' not found in data. Skipping transformation.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            if transform_type == "log1p":
                if (X[feature] < 0).any():
                    warnings.warn(
                        f"Feature '{feature}' contains negative values. log1p may produce NaN/inf values.",
                        UserWarning,
                        stacklevel=2,
                    )
                X[feature] = np.log1p(X[feature])

            elif transform_type == "yeo-johnson":
                if feature not in self.power_transformers_:
                    raise ValueError(f"PowerTransformer for '{feature}' not fitted. Call fit() first.")
                X[feature] = self.power_transformers_[feature].transform(X[[feature]]).ravel()

            elif transform_type == "log":
                if (X[feature] <= 0).any():
                    warnings.warn(
                        f"Feature '{feature}' contains non-positive values. Adding small epsilon before log transform.",
                        UserWarning,
                        stacklevel=2,
                    )
                    X[feature] = np.log(X[feature] + 1e-8)
                else:
                    X[feature] = np.log(X[feature])

            elif transform_type == "sqrt":
                if (X[feature] < 0).any():
                    warnings.warn(
                        f"Feature '{feature}' contains negative values. Taking absolute value before sqrt transform.",
                        UserWarning,
                        stacklevel=2,
                    )
                    X[feature] = np.sqrt(np.abs(X[feature]))
                else:
                    X[feature] = np.sqrt(X[feature])

            elif transform_type == "none":
                pass

            else:
                warnings.warn(
                    f"Unknown transformation type '{transform_type}' for feature '{feature}'. Skipping transformation.",
                    UserWarning,
                    stacklevel=2,
                )

        return X

    def get_feature_names_out(self, input_features=None):
        """Return feature names unchanged."""
        if hasattr(self, "feature_names_out_"):
            return self.feature_names_out_
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.array([], dtype=object)

    def get_fitted_info(self) -> dict:
        """Return fitted Yeo-Johnson parameters for inspection."""
        info = {}
        for feature, pt in self.power_transformers_.items():
            info[feature] = {
                "lambda": float(pt.lambdas_[0]),
                "standardize": self.yeo_johnson_standardize,
            }
        return info


class WoEBinningTransformer(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence Binning wrapper for OptBinning.

    Supports per-feature configuration for:
    - max_n_bins: Maximum number of bins
    - monotonic_trend: 'auto', 'ascending', 'descending', or None
    - min_bin_size: Minimum fraction of observations per bin

    Attributes:
        config: WoEConfig object
        numeric_features: List of numeric features to potentially bin
        binners_: Dict of fitted OptimalBinning objects (fitted)
        feature_names_in_: Input feature names (fitted)
        feature_names_out_: Output feature names (fitted)
    """

    def __init__(self, config: WoEConfig, numeric_features: list[str]):
        self.config = config
        self.numeric_features = numeric_features
        self.binners_ = {}

        if self.config.enabled and not OPTBINNING_AVAILABLE:
            warnings.warn(
                "WoEBinningTransformer is enabled but optbinning is not installed."
                "Features will be passed through unchanged.",
                UserWarning,
                stacklevel=2,
            )

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        else:
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])], dtype=object)

        self.feature_names_out_ = self.feature_names_in_.copy()

        if not self.config.enabled or not OPTBINNING_AVAILABLE:
            return self

        if y is None:
            warnings.warn(
                "WoE binning requires target variable y but none provided. Skipping WoE binning.",
                UserWarning,
                stacklevel=2,
            )
            return self

        features_to_bin = self.config.features if self.config.features else self.numeric_features

        for col in features_to_bin:
            if col not in X.columns:
                continue

            # Get feature-specific config or use defaults
            feature_config = self.config.feature_configs.get(col, {})
            max_n_bins = feature_config.get("max_n_bins", self.config.max_n_bins)
            min_prebin_size = feature_config.get("min_prebin_size", self.config.min_prebin_size)
            special_codes = feature_config.get("special_codes", self.config.special_codes)
            monotonic_trend = feature_config.get("monotonic_trend", "auto")

            try:
                optb = OptimalBinning(
                    name=col,
                    dtype="numerical",
                    solver=self.config.solver,
                    max_n_bins=max_n_bins,
                    min_prebin_size=min_prebin_size,
                    special_codes=special_codes,
                    monotonic_trend=monotonic_trend,
                )
                optb.fit(X[col], y)
                self.binners_[col] = optb

            except Exception as e:
                warnings.warn(
                    f"Failed to fit WoE binning for {col}: {e}. Skipping this feature.",
                    UserWarning,
                    stacklevel=2,
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
                        f"Failed to transform {col} with WoE binning: {e}. Keeping original values.",
                        UserWarning,
                        stacklevel=2,
                    )

        return X

    def get_feature_names_out(self, input_features=None):
        """Return feature names unchanged."""
        if hasattr(self, "feature_names_out_"):
            return self.feature_names_out_
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.array([], dtype=object)

    def get_binning_table(self, feature: str) -> pd.DataFrame:
        """Get binning table for a specific feature."""
        if feature not in self.binners_:
            raise ValueError(f"No binner found for feature '{feature}'")
        return self.binners_[feature].binning_table.build()

    def print_summary(self):
        """Print summary of all binners."""
        for col, binner in self.binners_.items():
            self.get_binning_table(col)
            print(f"\n{'=' * 60}")
            print(f"Feature: {col}")
            print(f"{'=' * 60}")
            binner.binning_table.plot()


def create_validation_pipeline(config: Config) -> Pipeline:
    """
    Create validation pipeline.
    """
    return Pipeline([("schema", SchemaValidator(config.data.numeric_features, enforce_order=True))])


def create_indicator_pipeline(config: Config) -> Pipeline:
    """
    Create indicator extraction pipeline.
    """
    pp = config.preprocessing

    if not pp.indicator_extraction.enabled:
        return Pipeline([("passthrough", FunctionTransformer())])

    return Pipeline([("indicator_extractor", IndicatorExtractor(pp.indicator_extraction))])


def create_cleaning_pipeline(config: Config) -> Pipeline:
    """
    Create cleaning pipeline.
    """
    pp = config.preprocessing

    steps = []

    if pp.outlier_clipping.enabled:
        steps.append(("clipper", OutlierClipper(pp.outlier_clipping)))

    steps.append(("imputer", CustomImputer(pp.imputation)))

    return Pipeline(steps)


def create_feature_pipeline(config: Config) -> Pipeline:
    """
    Create feature engineering pipeline.
    """
    pp = config.preprocessing

    if not pp.feature_creation.enabled:
        return Pipeline([("passthrough", FunctionTransformer())])

    return Pipeline([("feature_creator", FeatureCreator(pp.feature_creation))])


def create_transformation_pipeline(config: Config) -> Pipeline:
    """
    Create scaling pipeline.
    """
    pp = config.preprocessing

    steps = []

    if pp.feature_transformation.enabled and pp.feature_transformation.transformations:
        steps.append(
            (
                "feature_transformer",
                FeatureTransformer(pp.feature_transformation),
            )
        )
        return Pipeline([("passthrough", FunctionTransformer())])

    if pp.woe.enabled:
        steps.append(("woe", WoEBinningTransformer(pp.woe, config.data.numeric_features)))

    if not steps:
        return Pipeline([("passthrough", FunctionTransformer())])

    return Pipeline(steps)


def create_scaling_pipeline(config: Config) -> Pipeline:
    """
    Create scaling pipeline.
    """
    pp = config.preprocessing

    if not pp.scaling.enabled:
        return Pipeline([("passthrough", FunctionTransformer())])

    if pp.scaling.method == "minmax":
        scaler = MinMaxScaler()
    elif pp.scaling.method == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    return Pipeline([("scaler", scaler)])


def create_preprocessing_pipeline(config: Config) -> Pipeline:
    """
    Create complete preprocessing pipeline.
    """
    return Pipeline(
        [
            ("validation", create_validation_pipeline(config)),
            ("indicators", create_indicator_pipeline(config)),
            ("cleaning", create_cleaning_pipeline(config)),
            ("features", create_feature_pipeline(config)),
            ("transformation", create_transformation_pipeline(config)),
            ("scaling", create_scaling_pipeline(config)),
        ]
    )


def save_pipeline(pipeline: Pipeline, path: Path, config: Config | None = None):
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
