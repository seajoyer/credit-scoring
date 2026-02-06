# Модель кредитного скоринга

Проект по построению модели кредитного скоринга на основе датасета ["Give Me Some Credit"](https://www.kaggle.com/competitions/GiveMeSomeCredit/data). Проект включает полный цикл разработки ML-решения: от исследовательского анализа данных до реализации воспроизводимого пайплайна предобработки данных и обучения моделей машинного обучения.

## Описание проекта

- Бизнес-задача: Разработать модель для предсказания вероятности дефолта заемщика в течение двух лет. Это поможет банку принимать более обоснованные решения о выдаче кредита и минимизировать финансовые риски.

- Техническая задача: Сформировать признаки, обучить и сравнить несколько моделей машинного обучения: от простого бейзлайна до ансамблевых методов. Оценить качество моделей с помощью метрики ROC-AUC.

## Датасет ["Give Me Some Credit"](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)

- 150,000 записей клиентов банка  
- дисбаланс классов: 6.68% положительных случаев

### Целевая переменная

`SeriousDlqin2yrs` — бинарный индикатор серьезной просрочки (90+ дней) в течение последних 2 лет.

### Признаки

- `age` — возраст заемщика
- `MonthlyIncome` — ежемесячный доход
- `NumberOfDependents` — количество иждивенцев
- `RevolvingUtilizationOfUnsecuredLines` — коэффициент использования необеспеченных кредитных линий
- `DebtRatio` — отношение долговой нагрузки к доходу
- `NumberOfOpenCreditLinesAndLoans` — количество открытых кредитов и кредитных линий
- `NumberRealEstateLoansOrLines` — количество ипотечных кредитов
- `NumberOfTime30-59DaysPastDueNotWorse` — количество просрочек 30-59 дней
- `NumberOfTime60-89DaysPastDueNotWorse` — количество просрочек 60-89 дней
- `NumberOfTimes90DaysLate` — количество просрочек 90+ дней

## Основные результаты

Сформировано пространство признаков и обучен ряд моделей. На обработанных данных все модели демонстрируют схожее качество с **ROC-AUC 0.86-0.87**, что указывает на хорошую способность ранжировать заемщиков по уровню риска.

| Модель | Precision | Recall | F1 Score | ROC-AUC |
|--------|-----------|--------|----------|---------|
| XGBoost | 0.2158 | 0.7835 | 0.3384 | **0.8699** |
| CatBoost | 0.2142 | 0.7865 | 0.3367 | 0.8696 |
| Logistic Regression (tuned) | 0.2209 | 0.7646 | 0.3428 | 0.8658 |
| Logistic Regression (WoE) | 0.2226 | 0.7546 | 0.3438 | 0.8618 |
| Logistic Regression (baseline) | 0.2184 | 0.7506 | 0.3383 | 0.8618 |

### Ключевые выводы

- **Высокий Recall (75-78%)** — модели успешно выявляют большинство потенциальных дефолтов;
- **Низкий Precision (~21-22%)** — компромисс между выявлением рисков и ложными срабатываниями;
- **Градиентный бустинг** показали лишь незначительное преимущество перед логистической регрессией;
- **Итоговая рекомендация**: в силу наилучшей интерпретируемости и сравнительного качества, в продакшене целесообразно использовать логистическую регрессию на WoE признаках.

## Методология

### 1. Exploratory Data Analysis

**Ключевые инсайты:**

- Выборка несбалансирована: доля дефолтов **6.68%**
- Пропуски: `MonthlyIncome` (19.82%), `NumberOfDependents` (2.62%)
- Выявлены аномалии в данных, например:
  - Пики в распределении `RevolvingUtilizationOfUnsecuredLines` при 0.00 и 0.99999990
  - Возраст "0 лет"
  - Экстремальные значения `DebtRatio` (max: 329664)
  - Специальные коды (96, 98) в признаках просрочек
  - И другие...

Подробнее см. [ноутбук с EDA](notebooks/1.0-eda.ipynb).

### 2. Feature Engineering

На основе [EDA](notebooks/1.0-eda.ipynb) и [исследований](notebooks/2.0-feature-engineering.ipynb), реализован модульный pipeline предобработки с конфигурируемыми компонентами:

#### Обработка выбросов (`OutlierClipper`)
- Обработка по динамическим перцентилям и фиксированным значениям
- Поддержка различных методов замены (clip, median, zero)
- Специальные стратегии для `DebtRatio` и `MonthlyIncome`

#### Импутация (`CustomImputer`)
- Конфигурируемые стратегии для каждого признака
- Опциональное исключение признаков из импутации

#### Создание индикаторов (`IndicatorExtractor`)
- Индикаторы пропущенных значений
- Флаги клиппированных и аномальных значений
- Специальные категории (low/high/zero/whole/...)

#### Генерация признаков (`FeatureCreator`)
- Полиномиальные признаки (степени 2 и 3)
- `income_per_person` — доход на члена семьи
- Агрегации просрочек (total, weighted, has_delinquency)

#### Трансформации (`FeatureTransformer`)
- Конфигурируемые трансформации для каждого признака
- Поддержка Log1p, Yeo-Johnson и sqrt

#### WoE-биннинг (`WoEBinningTransformer`)
- Оптимальный биннинг с монотонными зависимостями
- Обработка специальных кодов и пропусков
- Индивидуальные настройки для каждого признака

#### Масштабирование (`ScalingConfig`)
- Standard, Robust, MinMax scaler
- Опциональное отключение

Пример использования см. в [ноутбуке с моделированием](notebooks/3.0-modeling.ipynb).

### 3. Modeling

#### Baseline
Логистическая регрессия с предобработкой (импутация, клиппинг + масштабирование), но без тюнинга и новых признаков.

#### Tuned Logistic Regression
- Полная цепочка feature engineering с добавлением новых признаков
- Grid Search по гиперпараметрам (C, l1_ratio)
- Feature selection на основе важности признаков

#### Logistic Regression + WoE
- Weight of Evidence трансформация признаков
- Grid Search по гиперпараметрам (C, l1_ratio)
- Feature selection на основе важности признаков

#### XGBoost
- Использование новых признаков, без предобработки данных
- Кастомный учет дисбаланса классов
- Grid Search по гиперпараметрам
- Feature selection на основе важности признаков

#### CatBoost
- Использование новых признаков, без предобработки данных
- Автоматическая обработка дисбаланса
- Randomized Search по гиперпараметрам
- Early stopping
- Feature selection на основе важности признаков

Подробнее см. [ноутбук с моделированием](notebooks/3.0-modeling.ipynb).

## Технологический стек

- **Python 3.12**
- **Data Processing**: pandas, numpy, optbinning
- **ML**: scikit-learn, xgboost, catboost
- **Visualization**: matplotlib, seaborn
- **Interpretability**: shap

## Структура проекта
```
├── data/                   # Данные (не включены в репозиторий)
├── models/                 # Сохраненные модели
├── notebooks/              # Jupyter notebooks
│   ├── 1.0-eda.ipynb       # Exploratory Data Analysis
│   ├── 2.0-feature-engineering.ipynb
│   └── 3.0-modeling.ipynb  # Model Training & Evaluation
├── src/                    # Исходный код
│   ├── __init__.py
│   ├── config.py           # Конфигурация pipeline
│   ├── features.py         # Классы Feature engineering
│   └── helpers.py          # Вспомогательные функции
├── pyproject.toml          # Зависимости проекта
└── README.md
```

### Пример обучения модели
```python
from src import get_config, create_preprocessing_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Загрузка конфигурации
config = get_config()

# Создание preprocessing pipeline
preprocessor = create_preprocessing_pipeline(config)

# Полный pipeline для обучения
model = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced'))
])

# Обучение
model.fit(X_train, y_train)

# Предсказание
y_pred_proba = model.predict_proba(X_test)[:, 1]
```

### Пример настройки preprocessing
```python
from src import Config, OutlierClippingConfig, WoEConfig

config = Config()

# Настройка обработки выбросов
config.preprocessing.outlier_clipping = OutlierClippingConfig(
    enabled=True,
    clip_age=True,
    age_clip_upper_percentile=99.0
)

# Использование WoE-биннинга
config.preprocessing.woe = WoEConfig(
    enabled=True,
    max_n_bins=10,
    solver='cp'
    feature_configs={
        "age": {"monotonic_trend": "descending"},
        "MonthlyIncome": {"max_n_bins": 7, "monotonic_trend": "auto"},
    }
)
```

## Лицензия MIT

Данный проект создан в образовательных целях.
