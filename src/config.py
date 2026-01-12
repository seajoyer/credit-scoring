from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Literal
import numpy as np
import random

@dataclass
class Paths:
    """File paths and directory structure."""
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
        self.data_dir = self.project_root / 'data'
        self.raw_data_dir = self.data_dir / 'raw'
        self.processed_data_dir = self.data_dir / 'processed'
        self.models_dir = self.project_root / 'models'
        self.figures_dir = self.project_root / 'figures'
        self.logs_dir = self.project_root / 'logs'
        
        self.train_file = self.raw_data_dir / 'cs-training.csv'
        self.test_file = self.raw_data_dir / 'cs-test.csv'
    
    def create_directories(self):
        for path in [self.raw_data_dir, self.processed_data_dir, 
                     self.models_dir, self.figures_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)
            
    def get_model_path(self, model_name: str, version: int = 1) -> Path:
        return self.models_dir / f'{model_name}_v{version}.pkl'
    
    def get_preprocessor_path(self, version: int = 1) -> Path:
        return self.models_dir / f'preprocessor_v{version}.pkl'

@dataclass
class DataConfig:
    """Data-related parameters."""
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
class FeatureEngineeringConfig:
    """Parameters for feature preprocessing and creation."""
    
    # Clipping defaults
    age_lower: int = 21
    age_upper: int = 87
    debt_ratio_upper: float = 2.0
    utilization_upper: float = 1.5
    income_upper_percentile: float = 99.0
    
    # Imputation defaults
    imputation_strategy: Literal['mean', 'median', 'most_frequent'] = 'median'
    income_imputation: Literal['mean', 'median', 'constant'] = 'median'
    dependents_imputation: Literal['median', 'constant', 'most_frequent'] = 'constant'
    
    # Scaling
    scale_features: bool = True
    scaling_method: Literal['standard', 'robust', 'minmax'] = 'standard'
    
    # Feature creation flags
    add_age_buckets: bool = True
    add_debt_income_ratio: bool = True
    add_utilization_squared: bool = True
    add_delinquency_features: bool = True
    
    # WoE Binning (Specific for Credit Scoring)
    use_woe_binning: bool = False
    woe_binning_features: List[str] = field(default_factory=lambda: []) # If empty, applies to all suitable

@dataclass
class ModelRegistry:
    """Registry of available model configurations."""
    # Placeholders for simple access, can be expanded as needed
    logistic_regression: Dict = field(default_factory=lambda: {
        'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000, 'class_weight': 'balanced'
    })
    lgbm: Dict = field(default_factory=lambda: {
        'n_estimators': 100, 'learning_rate': 0.1, 'class_weight': 'balanced'
    })
    xgboost: Dict = field(default_factory=lambda: {
        'n_estimators': 100, 'learning_rate': 0.1, 'scale_pos_weight': 14.0
    })

@dataclass
class TrainingConfig:
    random_seed: int = 42
    search_method: Literal['grid', 'random', 'bayesian'] = 'grid'
    n_jobs: int = -1

@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    models: ModelRegistry = field(default_factory=ModelRegistry)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        self.paths.create_directories()

def get_config() -> Config:
    return Config()

def set_seeds(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass