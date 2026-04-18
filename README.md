# Integrated Retail Analytics for Store Optimization

The Integrated Retail Analysis for Store Optimization project is an end-to-end data science solution designed to analyze retail sales data, uncover hidden patterns, and generate actionable business insights.

This project moves beyond basic analysis by combining:

- Data preprocessing
- Exploratory data analysis
- Anomaly detection
- Feature engineering   
- Machine learning
- Time-series analysis
Business strategy formulation

The final outcome is a production-ready forecasting system that helps optimize inventory, promotions, and store performance.

## Objectives
- Analyze retail sales patterns across stores and departments
- Detect anomalies and understand their causes
- Build a robust demand forecasting model
- Segment stores based on performance
- Infer product relationships for cross-selling
- Evaluate the impact of external economic factors
Generate actionable business strategies

## Project Structure

```
Integrated-Retail-Analytics-for-Store-Optimization/
├── data/
│   ├── raw/                  # Original datasets (sales, stores, features)
│   ├── interim/              # Merged / partially cleaned data
│   └── processed/            # Final cleaned + feature engineered data
│
├── notebooks/
│   └── Integrated_Retail_Analytics_for_Store_Optimization.ipynb # Original exploratory notebook
│
├── src/
│   ├── config/
│   │   └── config.py         # Paths, hyperparameters, column definitions
│   │
│   ├── data_ingestion/
│   │   └── load_data.py      # Load raw CSVs
│   │
│   ├── data_preprocessing/
│   │   ├── cleaning.py       # MarkDown imputation, data quality
│   │   ├── merging.py        # 3-way dataset merge
│   │   ├── transformation.py # Date parsing, log1p target
│   │   └── saving.py         # Interim & processed checkpoints
│   │
│   ├── eda/
│   │   ├── univariate.py     # Distributions, text summary
│   │   ├── bivariate.py      # Sales by type, holiday effect
│   │   ├── multivariate.py   # Correlation heatmap
│   │   └── visualization.py  # Shared plot helpers
│   │
│   ├── anomaly_detection/
│   │   ├── statistical.py    # IQR & Z-score outlier detection
│   │   └── time_series.py    # Rolling-baseline spike detection
│   │
│   ├── feature_engineering/
│   │   ├── feature_builder.py  # Time features, markdown aggregates
│   │   ├── encoding.py         # One-Hot Encoding (Type: A/B/C)
│   │   └── feature_selection.py# Feature + target selection, train/test split
│   │
│   ├── preprocessing/
│   │   └── scaler.py         # StandardScaler fit/transform/save/load
│   │
│   ├── model_building/
│   │   ├── train.py          # RandomForestRegressor training
│   │   ├── evaluate.py       # RMSE, MAE, R², metrics.json
│   │   └── tune.py           # CV + RandomizedSearchCV + GridSearchCV
│   │
│   ├── segmentation/
│   │   ├── clustering.py     # K-Means store clustering (k=3)
│   │   └── evaluation.py     # Silhouette score, elbow method
│   │
│   ├── market_basket/
│   │   └── association.py    # Dept co-performance correlation analysis
│   │
│   ├── external_factors/
│   │   └── analysis.py       # CPI / Unemployment / Fuel_Price impact
│   │
│   ├── personalization/
│   │   └── strategy.py       # Cluster-based store strategies
│   │
│   ├── explainability/
│   │   └── feature_importance.py # MDI + permutation importance
│   │
│   ├── utils/
│   │   ├── logger.py         # Centralised logging
│   │   └── helpers.py        # Utility functions
│   │
│   └── pipeline/
│       ├── data_pipeline.py  # Stages 1–4: ingest, EDA, preprocess
│       ├── train_pipeline.py # Stages 5–8: engineer, CV, train, eval
│       └── full_pipeline.py  # Orchestrates both pipelines
│
├── models/
│   ├── random_forest.pkl     # Trained Random Forest Regressor
│   └── scaler.pkl            # Fitted StandardScaler
│
├── reports/
│   ├── figures/              # All generated plots (.png)
│   ├── insights.txt          # EDA summary + feature importance ranking
│   ├── metrics.json          # Model evaluation metrics (RMSE/MAE/R²)
│   ├── cv_results.txt        # Cross-validation fold scores
│   └── tuning_results.txt    # Hyperparameter search results
│
├── requirements.txt
├── README.md
├── .gitignore
└── main.py
```

## Quick Start

### 1. Clone this repo

```bash
git clone https://github.com/ManuThota/Integrated-Retail-Analytics-for-Store-Optimization.git

cd Integrated-Retail-Analytics-for-Store-Optimization
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Place datasets

Download the dataset from : https://drive.google.com/drive/folders/1z9mPgev0LU23aKAAzgVVpqJ82NQDRapN?usp=sharing

```bash
Integrated-Retail-Analytics-for-Store-Optimization/
├── data/
│   ├── raw/ <- Place all the dataset files here
```

### 6. Run the full pipeline

```bash
python main.py
```

This executes all 8 stages and writes all outputs to `models/`, `reports/`, and `logs/`.

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `data_pipeline` | Load raw CSVs |
| 2 | `data_pipeline` | EDA (figures + text summary) |
| 3 | `data_pipeline` | Merge → clean → transform |
| 4 | `data_pipeline` | Save interim & processed |
| 5 | `train_pipeline` | OHE + feature select + scale |
| 6 | `train_pipeline` | 5-fold cross-validation |
| 7 | `train_pipeline` | Train Random Forest |
| 8 | `train_pipeline` | Evaluate + save reports |


## End-to-End Workflow
1. **Data Ingestion**
- Load sales, store, and external datasets
- Validate structure and schema
2. **Data Preprocessing**
- Convert date formats
- Merge datasets
- Handle missing values (MarkDown → 0)
- Remove duplicate columns
- Save:
    - interim/merged_data.csv
    - processed/processed_data.csv
3. **Anomaly Detection**
- IQR-based outlier removal
- Time-based anomaly detection
- Cleaned dataset for modeling
4. **Feature Engineering**
- One-hot encoding (Store Type)
- Time features (Year, Month, Week)
- Negative sales removal
- Log transformation of target variable
- Final datasets:
    - final_data.csv
5. **Machine Learning**
- Model: Random Forest Regressor
- Train-test split
- Evaluation metrics:
    - RMSE
    - MAE
    - R² Score
- Cross-validation
- Hyperparameter tuning (RandomizedSearchCV)
6. **Segmentation**
- K-Means clustering
- Store performance grouping
7. **Market Basket Analysis**
- Department-level correlation
- Cross-selling insights
8. **External Factors Analysis**
- CPI, Fuel Price, Unemployment impact
- Minimal influence compared to internal features
9. **Business Strategy**
- Inventory optimization
- Targeted promotions
- Store-specific strategies

## Model

**Algorithm**: Random Forest Regressor  
**Target**: `log1p(Weekly_Sales)` (log-transformed to reduce right-skew)  
**Encoding**: One-Hot Encoding for `Type` (A/B/C → Type_B, Type_C)  
**Scaler**: StandardScaler (fit on training data only)

**Baseline results** (from notebook):
| Metric | Value |
|--------|-------|
| CV R²  | ≈ 0.968 |
| Test R² | ≈ 0.97 |
| RMSE   | ≈ 0.34 (log-space) |
| MAE    | ≈ 0.18 (log-space) |

## License

This project is intended for educational and portfolio purposes.

## Author

mad_titan
Aspiring Data Scientist / ML Engineer

## Acknowledgment

This project demonstrates a complete transition from:
```
Exploratory notebook → Production-ready ML system
```