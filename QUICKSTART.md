# Quick Start Guide - Gold Price Forecasting Project

This guide will help you get started with the Gold Price Forecasting project.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/jimmyhomes12/Project-Two.git
cd Project-Two

# Create a virtual environment (recommended)
python -m venv .venv

# Activate the virtual environment
# On Linux/Mac:
source .venv/bin/activate

# On Windows:
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Verify Data

Ensure the dataset is in place:
```bash
ls sales-forecasting-gold-gpr/data/raw/
```

You should see:
- `Gold-Silver-GeopoliticalRisk_HistoricalData.csv`
- Other related CSV files

## Step 3: Run the Jupyter Notebook

### Option A: Using Jupyter Notebook

```bash
cd sales-forecasting-gold-gpr
jupyter notebook
```

Then open `notebooks/gold_gpr_forecasting.ipynb` and run all cells.

### Option B: Using JupyterLab

```bash
cd sales-forecasting-gold-gpr
jupyter lab
```

Navigate to `notebooks/gold_gpr_forecasting.ipynb` and run all cells.

## Step 4: Run the Streamlit Dashboard

After running the notebook to train the model:

```bash
cd sales-forecasting-gold-gpr
streamlit run app.py
```

The dashboard will open in your browser at http://localhost:8501

## What the Notebook Does

The Jupyter notebook will:

1. ✅ Load and clean the historical data
2. ✅ Perform exploratory data analysis with visualizations
3. ✅ Engineer features (lags, rolling stats, calendar features)
4. ✅ Split data into training and validation sets
5. ✅ Train baseline models (naive, moving average)
6. ✅ Train XGBoost regression model
7. ✅ Evaluate model performance
8. ✅ Generate visualizations and save results
9. ✅ Save the trained model to `models/gold_xgb_model.pkl`
10. ✅ Save forecasts to `outputs/forecasts/`

## What the Dashboard Offers

The Streamlit app provides:

- 📊 **Historical Data Viewer**: Explore gold, silver, and GPRD time series
- 📈 **Interactive Charts**: Zoom, pan, and analyze trends
- 🤖 **Model Forecasts**: View actual vs predicted values
- 📉 **Performance Metrics**: RMSE, MAE, MAPE displayed
- 📅 **Date Range Selection**: Filter data by custom date ranges

## Output Files

After running the notebook, you'll find:

### Plots (in `outputs/plots/`)
- `gold_silver_timeseries.png` - Gold and silver price trends
- `gprd_timeseries.png` - Geopolitical risk index over time
- `corr_matrix.png` - Correlation heatmap
- `gold_actual_vs_predicted_xgb.png` - Model performance visualization
- `feature_importance_xgb.png` - Top feature importances
- `residual_analysis.png` - Residual plots

### Models (in `models/`)
- `gold_xgb_model.pkl` - Trained XGBoost model
- `feature_info.pkl` - Feature metadata and metrics

### Forecasts (in `outputs/forecasts/`)
- `gold_val_forecasts_xgb.csv` - Validation period predictions

## Troubleshooting

### Issue: Module not found error
```bash
# Ensure you've activated the virtual environment
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\Activate.ps1  # Windows

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: Data file not found
```bash
# Verify data location
ls sales-forecasting-gold-gpr/data/raw/

# If missing, download from Kaggle:
# https://www.kaggle.com/datasets/[dataset-link]
```

### Issue: Streamlit won't start
```bash
# Check if streamlit is installed
pip list | grep streamlit

# If not installed:
pip install streamlit

# Try running with full path
python -m streamlit run app.py
```

### Issue: Port already in use
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

## Next Steps

1. **Explore the Data**: Use the dashboard to understand trends
2. **Review the Notebook**: Study the feature engineering approach
3. **Experiment**: Try different hyperparameters in the notebook
4. **Deploy**: Consider deploying the Streamlit app to Streamlit Cloud
5. **Extend**: Add SHAP analysis, different models, or additional features

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Project README](README.md)

## Support

For issues or questions:
- Check the main README.md
- Review the notebook comments
- Check GitHub Issues
- Contact the author

---

**Happy Forecasting! 📈**
