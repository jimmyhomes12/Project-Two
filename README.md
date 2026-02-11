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
   - Time-based train/validation split (pre-2020 / 2020+)
   - Baseline models (naive, moving average)
   - XGBoost regression model

4. **Evaluation**
   - RMSE, MAE, MAPE, R² metrics
   - Comparison against baselines
   - Feature importance analysis

## 📊 Results

The XGBoost model demonstrates strong predictive performance:
- Outperforms naive and moving average baselines
- Geopolitical risk features contribute significantly to predictions
- Strong correlation between predicted and actual values

*Note: Specific metrics available after running the notebook*

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
