from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Literal
import numpy as np
import random
import json


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
    age_bounds: tuple[int, int] = (21, 87)
    debt_ratio_upper: float = 2.0
    utilization_upper: float = 1.5
    income_upper_percentile: float = 99.0
    
    def to_dict(self) -> Dict:
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
    income_strategy: Literal['mean', 'median', 'constant'] = 'median'
    dependents_strategy: Literal['median', 'constant', 'most_frequent'] = 'constant'
    constant_fill_value: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass(frozen=True)
class FeatureCreationConfig:
    """
    Configuration for feature engineering.
    
    Attributes:
        add_age_buckets: Create age category indicators
        add_debt_income_ratio: Create debt*income interaction
        add_utilization_squared: Create utilization^2 feature
        add_delinquency_features: Create delinquency aggregations
        age_young_threshold: Age below which is considered "young"
        age_senior_threshold: Age above which is considered "senior"
    """
    add_age_buckets: bool = True
    add_debt_income_ratio: bool = True
    add_utilization_squared: bool = True
    add_delinquency_features: bool = True
    
    age_young_threshold: int = 30
    age_senior_threshold: int = 60
    
    def to_dict(self) -> Dict:
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
    method: Literal['standard', 'robust', 'minmax'] = 'standard'
    
    def to_dict(self) -> Dict:
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
    """
    enabled: bool = False
    features: List[str] = field(default_factory=list)
    solver: Literal['cp', 'mip'] = 'cp'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['features'] = list(result['features'])
        return result


@dataclass
class PreprocessingConfig:
    """
    Master preprocessing configuration combining all sub-configs.
    """
    outlier_clipping: OutlierClippingConfig = field(default_factory=OutlierClippingConfig)
    imputation: ImputationConfig = field(default_factory=ImputationConfig)
    feature_creation: FeatureCreationConfig = field(default_factory=FeatureCreationConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    woe: WoEConfig = field(default_factory=WoEConfig)
    
    def to_dict(self) -> Dict:
        """Serialize entire preprocessing config to dictionary."""
        return {
            'outlier_clipping': self.outlier_clipping.to_dict(),
            'imputation': self.imputation.to_dict(),
            'feature_creation': self.feature_creation.to_dict(),
            'scaling': self.scaling.to_dict(),
            'woe': self.woe.to_dict()
        }
    
    def save(self, path: Path):
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'PreprocessingConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            outlier_clipping=OutlierClippingConfig(**data['outlier_clipping']),
            imputation=ImputationConfig(**data['imputation']),
            feature_creation=FeatureCreationConfig(**data['feature_creation']),
            scaling=ScalingConfig(**data['scaling']),
            woe=WoEConfig(**data['woe'])
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
        self.data_dir = self.project_root / 'data'
        self.raw_data_dir = self.data_dir / 'raw'
        self.processed_data_dir = self.data_dir / 'processed'
        self.models_dir = self.project_root / 'models'
        self.figures_dir = self.project_root / 'figures'
        self.logs_dir = self.project_root / 'logs'
        
        self.train_file = self.raw_data_dir / 'cs-training.csv'
        self.test_file = self.raw_data_dir / 'cs-test.csv'
    
    def create_directories(self):
        """Create all project directories if they don't exist."""
        for path in [self.raw_data_dir, self.processed_data_dir, 
                     self.models_dir, self.figures_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str, version: int = 1) -> Path:
        """Get path for a model file."""
        return self.models_dir / f'{model_name}_v{version}.pkl'
    
    def get_preprocessor_path(self, version: int = 1) -> Path:
        """Get path for a preprocessor file."""
        return self.models_dir / f'preprocessor_v{version}.pkl'
    
    def get_config_path(self, config_name: str = 'preprocessing') -> Path:
        """Get path for a configuration file."""
        return self.models_dir / f'{config_name}_config.json'


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
    target_col: str = 'SeriousDlqin2yrs'
    numeric_features: List[str] = field(default_factory=lambda: [
        'RevolvingUtilizationOfUnsecuredLines',
        'age',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio',
        'MonthlyIncome',
        'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents'
    ])
    test_size: float = 0.2
    stratify: bool = True


@dataclass
class ModelRegistry:
    """
    Registry of model configurations.
    
    Provides default hyperparameters for common models.
    """
    logistic_regression: Dict = field(default_factory=lambda: {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 1000,
        'class_weight': 'balanced'
    })
    
    lgbm: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'num_leaves': 31,
        'class_weight': 'balanced',
        'random_state': 42
    })
    
    xgboost: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'scale_pos_weight': 14.0,
        'random_state': 42
    })
    
    random_forest: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 10,
        'class_weight': 'balanced',
        'random_state': 42
    })


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
    search_method: Literal['grid', 'random', 'bayesian'] = 'grid'
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
    
    def save(self, directory: Path, name: str = 'config'):
        """
        Save complete configuration.
        
        Args:
            directory: Directory to save to
            name: Base name for config files
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        preprocessing_path = directory / f'{name}_preprocessing.json'
        self.preprocessing.save(preprocessing_path)
        
        print(f"Configuration saved to {directory}/")
    
    @classmethod
    def load(cls, directory: Path, name: str = 'config') -> 'Config':
        """
        Load configuration.
        
        Args:
            directory: Directory to load from
            name: Base name for config files
        
        Returns:
            Loaded Config object
        """
        directory = Path(directory)
        
        preprocessing_path = directory / f'{name}_preprocessing.json'
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
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
