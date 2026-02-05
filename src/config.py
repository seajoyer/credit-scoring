import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class OutlierClippingConfig:
    """
    Configuration for outlier clipping behavior.

    Attributes:
        age_bounds: (lower, upper) age limits
        debt_ratio_upper: Maximum debt ratio to allow
        utilization_upper: Maximum credit utilization to allow
        income_upper_percentile: Percentile to clip income at (dynamic)
    """
    enabled: bool = True

    clip_age: bool = True
    age_clip_lower: int = 21
    age_clip_upper_percentile: float = 99.0

    clip_income: bool = True
    income_clip_upper_percentile: float = 90.0

    clip_util: bool = True
    util_clip_upper: float = 1.0

    clip_debt_ratio: bool = True
    debt_ratio_clip_upper: float = 4.0

    clip_deps: bool = True
    deps_clip_upper_percentile: float = 97.5

    clip_loans: bool = True
    loans_clip_upper_percentile: float = 99.0

    clip_estate: bool = True
    estate_clip_upper_percentile: float = 99.0

    clip_pd30_59: bool = True
    pd30_59_clip_upper_percentile: float = 99.0

    clip_pd60_89: bool = True
    pd60_89_clip_upper_percentile: float = 99.0

    clip_pd90: bool = True
    pd90_clip_upper_percentile: float = 99.0

    # Replacement strategies
    debt_ratio_strategy: Literal["clip", "custom_median", "zero"] = "custom_median"
    income_strategy: Literal["clip", "median", "zero"] = "clip"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass(frozen=True)
class ImputationConfig:
    """
    Configuration for missing value imputation.

    Attributes:
        income_strategy: How to impute missing income values
        dependents_strategy: How to impute missing dependent counts
        constant_fill_value: Value to use for constant imputation
    """

    income_strategy: Literal["mean", "median", "constant"] = "constant"
    dependents_strategy: Literal["median", "constant", "most_frequent"] = "constant"
    features_skip_imputation: list[str] = field(default_factory=list)

    constant_fill_value: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass(frozen=True)
class IndicatorExtractionConfig:
    """
    Configuration for feature engineering.

    Attributes:
        add_age_buckets: Add age category indicators
        add_income_missing: Add indicator for missing income values
        add_income_clipped: Add indicator for clipped income values
        add_income_zero: Add indicator for zero income values
        add_income_low: Add indicator for low income values
        add_utilization_clipped: Add indicator for clipped utilization values
        add_utilization_99999990: Add indicator for utilization == 0.99999990
        add_deps_missing: Add indicator for missing NumberOfDependents values
        add_debt_ratio_buckets: Create DebtRatio category indicators
        add_debt_ratio_zero: Category with zero DebtRatio values
        add_debt_ratio_whole: Category with whole DebtRatio values

        age_young_threshold: Age below which is considered "young"
        age_senior_threshold: Age above which is considered "senior"
    """

    enabled: bool = True

    add_missing: bool = True
    add_age_buckets: bool = True
    add_income_clipped: bool = True
    add_income_zero: bool = True
    add_income_low: bool = True
    add_utilization_clipped: bool = True
    add_utilization_99999990: bool = True
    add_debt_ratio_buckets: bool = True
    add_debt_ratio_zero: bool = True
    add_debt_ratio_whole: bool = True
    add_high_debt_low_income: bool = True

    age_young_threshold: int = 30
    age_senior_threshold: int = 59

    income_upper_percentile: float = OutlierClippingConfig().income_clip_upper_percentile
    income_low_percentile: float = 10.0

    util_clip_upper: float = OutlierClippingConfig().util_clip_upper

    debt_ratio_clip_upper: float = OutlierClippingConfig().debt_ratio_clip_upper
    debt_ratio_high_percentile: float = 80.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass(frozen=True)
class FeatureCreationConfig:
    """
    Configuration for feature engineering.

    Attributes:
        add_age_polynomial: Create age^2 and age^3 features
        add_debt_ratio_polynomial: Create DebtRatio^2 and DebtRatio^3 features
        add_loans_polynomial: Create NumberOfOpenCreditLinesAndLoans^2 and NumberOfOpenCreditLinesAndLoans^3 features
        add_estate_polynomial: Create NumberRealEstateLoansOrLines^2 and NumberRealEstateLoansOrLines^3 features

        add_delinquency_features: Create delinquency aggregations
    """

    enabled: bool = False

    add_age_polynomial: bool = True
    add_debt_ratio_polynomial: bool = True
    add_loans_polynomial: bool = True
    add_estate_polynomial: bool = True

    add_income_per_person: bool = True
    add_delinquency_features: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass(frozen=True)
class ScalingConfig:
    """
    Configuration for feature scaling.

    Attributes:
        enabled: Whether to apply scaling
        method: Type of scaling to use
    """

    enabled: bool = True
    method: Literal["standard", "robust", "minmax"] = "standard"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass(frozen=True)
class FeatureTransformationConfig:
    """
    Configuration for feature-specific transformations.

    Attributes:
        transformations: Dict mapping feature names to transformation types
        yeo_johnson_standardize: Whether to standardize after Yeo-Johnson
    """
    enabled: bool = True

    transformations: dict[str, Literal["log1p", "yeo-johnson", "log", "sqrt", "none"]] = field(
        default_factory=lambda: {
            "DebtRatio": "yeo-johnson",
            "DebtRatio^2": "yeo-johnson",
            "DebtRatio^3": "yeo-johnson",
            "NumberOfOpenCreditLinesAndLoans": "log1p",
            "NumberOfOpenCreditLinesAndLoans^2": "log1p",
            "NumberOfOpenCreditLinesAndLoans^3": "log1p",
            "NumberOfTimes90DaysLate": "log1p",
        }
    )

    yeo_johnson_standardize: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass(frozen=True)
class WoEConfig:
    """
    Configuration for Weight of Evidence binning.

    Attributes:
        enabled: Whether to apply WoE binning
        features: List of features to bin (empty = all numeric)
        solver: Optimization solver to use
        max_n_bins: Default maximum number of bins
        min_prebin_size: Minimum fraction of observations per bin
        feature_configs: Dict mapping feature names to their specific configs
    """

    enabled: bool = False
    solver: Literal["cp", "mip"] = "cp"
    max_n_bins: int = 10
    min_prebin_size: float = 0.05
    special_codes: dict = None
    feature_configs: dict[str, dict] = field(
        default_factory=lambda: {
            "age": {"max_n_bins": 10, "monotonic_trend": "descending"},
            "MonthlyIncome": {"max_n_bins": 10, "monotonic_trend": "auto"},
            "income_per_person": {"max_n_bins": 10, "monotonic_trend": "auto"},
            "total_delinquencies": {"max_n_bins": 10, "monotonic_trend": "auto"},
            "weighted_delinquencies": {"max_n_bins": 10, "monotonic_trend": "auto"},
            "RevolvingUtilizationOfUnsecuredLines": {"max_n_bins": 10, "monotonic_trend": "ascending"},
            "DebtRatio": {"max_n_bins": 10, "special_codes": {"zero": 0}},
            "NumberOfDependents": {"max_n_bins": 10, "min_prebin_size": 0.02, "monotonic_trend": "ascending"},
            "NumberOfOpenCreditLinesAndLoans": {"max_n_bins": 10, "monotonic_trend": "descending"},
            "NumberRealEstateLoansOrLines": {"max_n_bins": 10, "monotonic_trend": "auto"},
            "NumberOfTime30-59DaysPastDueNotWorse": {
                "max_n_bins": 10,
                "min_prebin_size": 0.005,
                "special_codes": {98: 98, 96: 96},
                "monotonic_trend": "ascending",
            },
            "NumberOfTime60-89DaysPastDueNotWorse": {
                "max_n_bins": 10,
                "min_prebin_size": 0.01,
                "special_codes": {98: 98, 96: 96},
                "monotonic_trend": "ascending",
            },
            "NumberOfTimes90DaysLate": {
                "max_n_bins": 10,
                "min_prebin_size": 0.005,
                "special_codes": {98: 98, 96: 96},
                "monotonic_trend": "ascending",
            },
        }
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result["features"] = list(result["features"])
        return result


@dataclass
class PreprocessingConfig:
    """
    Master preprocessing configuration combining all sub-configs.
    """

    indicator_extraction: IndicatorExtractionConfig = field(default_factory=IndicatorExtractionConfig)
    outlier_clipping: OutlierClippingConfig = field(default_factory=OutlierClippingConfig)
    imputation: ImputationConfig = field(default_factory=ImputationConfig)
    feature_creation: FeatureCreationConfig = field(default_factory=FeatureCreationConfig)
    feature_transformation: FeatureTransformationConfig = field(default_factory=FeatureTransformationConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    woe: WoEConfig = field(default_factory=WoEConfig)

    def to_dict(self) -> dict:
        """Serialize entire preprocessing config to dictionary."""
        return {
            "outlier_clipping": self.outlier_clipping.to_dict(),
            "imputation": self.imputation.to_dict(),
            "feature_creation": self.feature_creation.to_dict(),
            "feature_transformation": self.feature_transformation.to_dict(),
            "scaling": self.scaling.to_dict(),
            "woe": self.woe.to_dict(),
        }

    def save(self, path: Path):
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "PreprocessingConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)

        return cls(
            outlier_clipping=OutlierClippingConfig(**data["outlier_clipping"]),
            imputation=ImputationConfig(**data["imputation"]),
            feature_creation=FeatureCreationConfig(**data["feature_creation"]),
            feature_transformation=FeatureTransformationConfig(**data.get("feature_transformation", {})),
            scaling=ScalingConfig(**data["scaling"]),
            woe=WoEConfig(**data["woe"]),
        )


@dataclass
class Paths:
    """
    Automatically create directory structure on initialization.
    """

    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    train_file: Path = field(init=False)
    test_file: Path = field(init=False)

    def __post_init__(self):
        """Initialize derived paths."""
        self.data_dir = self.project_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.models_dir = self.project_root / "models"
        self.figures_dir = self.project_root / "figures"
        self.logs_dir = self.project_root / "logs"

        self.train_file = self.raw_data_dir / "cs-training.csv"
        self.test_file = self.raw_data_dir / "cs-test.csv"

    def create_directories(self):
        """Create all project directories if they don't exist."""
        for path in [self.raw_data_dir, self.processed_data_dir, self.models_dir, self.figures_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_name: str, version: int = 1) -> Path:
        """Get path for a model file."""
        return self.models_dir / f"{model_name}_v{version}.pkl"

    def get_preprocessor_path(self, version: int = 1) -> Path:
        """Get path for a preprocessor file."""
        return self.models_dir / f"preprocessor_v{version}.pkl"

    def get_config_path(self, config_name: str = "preprocessing") -> Path:
        """Get path for a configuration file."""
        return self.models_dir / f"{config_name}_config.json"


@dataclass
class DataConfig:
    """
    Data-related parameters.

    Attributes:
        target_col: Name of target variable
        numeric_features: List of numeric feature names
        test_size: Fraction of data for test set
        stratify: Whether to stratify train/test split
    """

    target_col: str = "SeriousDlqin2yrs"
    numeric_features: list[str] = field(
        default_factory=lambda: [
            "age",
            "MonthlyIncome",
            "RevolvingUtilizationOfUnsecuredLines",
            "DebtRatio",
            "NumberOfDependents",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberRealEstateLoansOrLines",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
        ]
    )
    test_size: float = 0.2
    random_seed: int = 42
    stratify: bool = True


@dataclass
class ModelRegistry:
    """
    Registry of model configurations.

    Provides default hyperparameters for common models.
    """

    logistic_regression: dict = field(
        default_factory=lambda: {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": "balanced",
        }
    )

    lgbm: dict = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "num_leaves": 31,
            "class_weight": "balanced",
            "random_state": 42,
        }
    )

    xgboost: dict = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "scale_pos_weight": 14.0,
            "random_state": 42,
        }
    )

    random_forest: dict = field(
        default_factory=lambda: {"n_estimators": 100, "max_depth": 10, "class_weight": "balanced", "random_state": 42}
    )


@dataclass
class TrainingConfig:
    """
    Training and hyperparameter search configuration.

    Attributes:
        random_seed: Random seed for reproducibility
        search_method: Type of hyperparameter search
        n_jobs: Number of parallel jobs (-1 = all cores)
        cv_folds: Number of cross-validation folds
    """

    random_seed: int = 42
    search_method: Literal["grid", "random", "bayesian"] = "grid"
    n_jobs: int = -1
    cv_folds: int = 5


@dataclass
class Config:
    """
    Master configuration object for entire pipeline.

    Combines all sub-configurations and ensures directory structure exists.
    """

    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    models: ModelRegistry = field(default_factory=ModelRegistry)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        """Initialize configuration."""
        self.paths.create_directories()
        set_seeds(self.training.random_seed)

    def save(self, directory: Path, name: str = "config"):
        """
        Save complete configuration.

        Args:
            directory: Directory to save to
            name: Base name for config files
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        preprocessing_path = directory / f"{name}_preprocessing.json"
        self.preprocessing.save(preprocessing_path)

        print(f"Configuration saved to {directory}/")

    @classmethod
    def load(cls, directory: Path, name: str = "config") -> "Config":
        """
        Load configuration.

        Args:
            directory: Directory to load from
            name: Base name for config files

        Returns:
            Loaded Config object
        """
        directory = Path(directory)

        preprocessing_path = directory / f"{name}_preprocessing.json"
        preprocessing = PreprocessingConfig.load(preprocessing_path)

        return cls(preprocessing=preprocessing)


def get_config() -> Config:
    """Get default configuration."""
    return Config()


def set_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch  # ty:ignore[unresolved-import]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
