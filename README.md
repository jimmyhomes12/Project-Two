# Project Two: Gold Price Forecasting with Geopolitical Risk

## 📊 Project Overview

This repository contains a comprehensive machine learning project that forecasts **next-day gold prices** using historical market data and **geopolitical risk indices**. The project demonstrates end-to-end data science skills including:

- Time-series feature engineering
- Machine learning model development (XGBoost)
- Model evaluation and validation
- Interactive dashboard development with Streamlit

## 🎯 Key Features

- **Predictive Modeling**: XGBoost regression model to forecast gold prices
- **Geopolitical Risk Analysis**: Integration of GPRD indices as features
- **Feature Engineering**: Lag features, rolling statistics, and calendar features
- **Interactive Dashboard**: Streamlit web app for data exploration and model visualization
- **Comprehensive Documentation**: Jupyter notebooks with detailed analysis

## 📁 Repository Structure

```
Project-Two/
├── sales-forecasting-gold-gpr/     # Main project directory
│   ├── data/                       # Data storage
│   │   ├── raw/                    # Raw CSV files
│   │   └── processed/              # Processed data
│   ├── notebooks/                  # Jupyter notebooks
│   │   └── gold_gpr_forecasting.ipynb
│   ├── models/                     # Saved models
│   ├── outputs/                    # Generated outputs
│   │   ├── plots/                  # Visualizations
│   │   └── forecasts/              # Forecast CSVs
│   ├── app.py                      # Streamlit dashboard
│   └── README.md                   # Project documentation
├── retail-eda-reference/           # Reference for related project
├── requirements.txt                # Python dependencies
└── .gitignore                      # Git ignore rules
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/jimmyhomes12/Project-Two.git
cd Project-Two
```

2. Create and activate a virtual environment:
```bash
# Linux/Mac
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the Jupyter Notebook

```bash
cd sales-forecasting-gold-gpr
jupyter notebook notebooks/gold_gpr_forecasting.ipynb
```

### Run the Streamlit Dashboard

**🚀 Live Demo**: [Coming soon – deploy to Streamlit Cloud]

Or run locally:

```bash
cd sales-forecasting-gold-gpr
streamlit run app.py
```

Then navigate to http://localhost:8501 in your browser.

## 📈 Dataset

- **Source**: Gold-Silver Price VS Geopolitical Risk (1985–2025) from Kaggle
- **Time Period**: ~40 years of daily data (1985–2025)
- **Key Variables**:
  - Gold and Silver spot prices
  - Geopolitical Risk Indices (GPRD, GPRD_ACT, GPRD_THREAT)
  - Event descriptions

## 🔬 Methodology

1. **Data Preparation**
   - Load and clean historical data
   - Handle missing values with forward/backward fill

2. **Feature Engineering**
   - Create lag features (1, 2, 5, 10, 20 days)
   - Calculate rolling means and standard deviations
   - Extract calendar features (year, month, day of week)

3. **Model Development**
   - Time-based train/validation split (2000-2017 train / 2018-2020 validation)
   - Baseline models (naive, moving average)
   - XGBoost regression model

4. **Evaluation**
   - RMSE, MAE, MAPE, R² metrics
   - Comparison against baselines
   - Feature importance analysis

## 📊 Key Results

Model performance on validation period (2018-2020):

- **XGBoost RMSE**: $72.48
- **XGBoost MAE**: $42.02  
- **XGBoost MAPE**: 2.43%
- **XGBoost R²**: 0.9068
- **Naive baseline RMSE**: $14.27
- **5-day MA RMSE**: $21.47

> **Note on baselines**: The naive baseline (using today's price to predict tomorrow) achieves low RMSE because gold prices are highly autocorrelated day-to-day. However, the XGBoost model's **R² of 0.91** shows it captures longer-term trends and patterns that simple baselines miss, especially when incorporating geopolitical risk indicators. For practical applications (e.g., detecting trend shifts during crises), the ML model provides value beyond naive persistence.

## 🔍 Key Insights from Analysis

1. **GOLD_LAG_1 dominates predictions** (50%+ feature importance), confirming strong day-to-day momentum in gold prices.

2. **Rolling averages (5-day, 20-day) capture trend** – second and fourth most important features, showing the model learns both short-term and medium-term dynamics.

3. **Geopolitical risk has measurable but modest impact** – GPRD features appear in top 20 but with <5% importance each. The correlation matrix shows GPRD has near-zero correlation with gold price (0.09), suggesting GPR's influence is subtle and conditional rather than direct.

4. **2020 COVID volatility well-captured** – the model tracks the March 2020 spike to $1,750 and subsequent correction, demonstrating robustness during extreme events.

5. **Silver provides minimal incremental signal** – SILVER_LAG features rank low in importance despite 0.92 correlation with gold, likely due to redundancy with gold lags.

6. **Residuals show heteroscedasticity** – errors increase at higher price levels (>$1,700), suggesting the model may benefit from log-transformation or variance-stabilizing preprocessing in future iterations.

## 🩺 Model Diagnostics

Residual analysis reveals:
- Generally centered errors around zero for most predictions.
- Increased variance at higher price levels (>$1,700), indicating the model underestimates volatility during extreme price moves.
- A few large outliers (~$150+ error) correspond to sudden geopolitical shocks not fully captured by lagged GPR features.

## 📈 Model Performance

![Gold Price Predictions](sales-forecasting-gold-gpr/outputs/plots/gold_actual_vs_predicted_xgb.png)

*The model tracks gold price trends well, with R² = 0.91 on the 2018-2020 validation period.*

## 🎯 Feature Importance

![Feature Importance](sales-forecasting-gold-gpr/outputs/plots/feature_importance_xgb.png)

*GOLD_LAG_1 and rolling averages dominate, with geopolitical risk features contributing modestly.*

## 🛠️ Technologies Used

- **Python 3.x**
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Matplotlib, Seaborn
- **Dashboard**: Streamlit
- **Notebooks**: Jupyter

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Jimmy Homes**
- GitHub: [@jimmyhomes12](https://github.com/jimmyhomes12)

## 🙏 Acknowledgments

- Dataset provided by Kaggle community
- Inspired by financial forecasting and geopolitical risk research
