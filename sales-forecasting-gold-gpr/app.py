"""
Gold Price Forecasting Dashboard
Interactive Streamlit app for exploring gold prices, geopolitical risk, and model forecasts
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Gold Price Forecasting",
    page_icon="📈",
    layout="wide"
)

# Set style
sns.set_style("whitegrid")

# Define paths
BASE_DIR = Path(__file__).parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_FORECASTS_DIR = BASE_DIR / "outputs" / "forecasts"

@st.cache_data
def load_data():
    """Load and prepare the dataset"""
    file_path = DATA_RAW_DIR / "Gold-Silver-GeopoliticalRisk_HistoricalData.csv"
    df = pd.read_csv(file_path)
    
    # Normalize column names
    df.columns = [c.strip().upper() for c in df.columns]
    
    # Convert date
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').set_index('DATE')
    
    # Select and clean relevant columns
    df = df[['GOLD_PRICE', 'SILVER_PRICE', 'GPRD', 'GPRD_ACT', 'GPRD_THREAT']]
    df = df.ffill().bfill()
    
    return df

@st.cache_data
def create_features(df):
    """Create all features for modeling"""
    df_feat = df.copy()
    
    # Target
    df_feat['GOLD_TARGET'] = df_feat['GOLD_PRICE'].shift(-1)
    
    # Lag features
    lags = [1, 2, 5, 10, 20]
    for lag in lags:
        df_feat[f'GOLD_LAG_{lag}'] = df_feat['GOLD_PRICE'].shift(lag)
        df_feat[f'SILVER_LAG_{lag}'] = df_feat['SILVER_PRICE'].shift(lag)
        df_feat[f'GPRD_LAG_{lag}'] = df_feat['GPRD'].shift(lag)
    
    # Rolling features
    windows = [5, 10, 20]
    for window in windows:
        df_feat[f'GOLD_ROLL_MEAN_{window}'] = df_feat['GOLD_PRICE'].rolling(window).mean()
        df_feat[f'GOLD_ROLL_STD_{window}'] = df_feat['GOLD_PRICE'].rolling(window).std()
        df_feat[f'GPRD_ROLL_MEAN_{window}'] = df_feat['GPRD'].rolling(window).mean()
    
    # Time features
    df_feat['YEAR'] = df_feat.index.year
    df_feat['MONTH'] = df_feat.index.month
    df_feat['DAYOFWEEK'] = df_feat.index.dayofweek
    
    return df_feat.dropna()

@st.cache_resource
def load_model():
    """Load the trained XGBoost model"""
    model_path = MODELS_DIR / "gold_xgb_model.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_forecasts():
    """Load pre-computed forecasts"""
    forecast_path = OUTPUTS_FORECASTS_DIR / "gold_val_forecasts_xgb.csv"
    if forecast_path.exists():
        df = pd.read_csv(forecast_path)
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df
    return None

def main():
    # Title and description
    st.title("📈 Gold Price Forecasting Dashboard")
    st.markdown("""
    Explore historical gold prices, geopolitical risk indices, and machine learning forecasts.
    This dashboard uses XGBoost to predict next-day gold prices based on historical trends and geopolitical risk data.
    """)
    
    # Load data
    with st.spinner("Loading data..."):
        df_raw = load_data()
        df_feat = create_features(df_raw)
        model = load_model()
        forecasts = load_forecasts()
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Date range selector
    min_date = df_raw.index.min().date()
    max_date = df_raw.index.max().date()
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=max_date - timedelta(days=365*2),
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Series selector
    series_option = st.sidebar.selectbox(
        "Select Series to View",
        ["Gold Price", "Silver Price", "Geopolitical Risk (GPRD)"]
    )
    
    # Filter data by date range
    mask = (df_raw.index.date >= start_date) & (df_raw.index.date <= end_date)
    df_filtered = df_raw[mask]
    
    # Main content area - two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Historical Data")
        
        # Plot selected series
        fig, ax = plt.subplots(figsize=(12, 5))
        
        if series_option == "Gold Price":
            ax.plot(df_filtered.index, df_filtered['GOLD_PRICE'], color='gold', linewidth=2)
            ax.set_ylabel('Price (USD)', fontsize=12)
            ax.set_title(f'Gold Spot Price ({start_date} to {end_date})', fontsize=14, fontweight='bold')
        elif series_option == "Silver Price":
            ax.plot(df_filtered.index, df_filtered['SILVER_PRICE'], color='silver', linewidth=2)
            ax.set_ylabel('Price (USD)', fontsize=12)
            ax.set_title(f'Silver Spot Price ({start_date} to {end_date})', fontsize=14, fontweight='bold')
        else:  # GPRD
            ax.plot(df_filtered.index, df_filtered['GPRD'], color='crimson', linewidth=2)
            ax.set_ylabel('GPRD Index', fontsize=12)
            ax.set_title(f'Geopolitical Risk Index ({start_date} to {end_date})', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Statistics
        st.subheader("📈 Summary Statistics")
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        if series_option == "Gold Price":
            series_data = df_filtered['GOLD_PRICE']
        elif series_option == "Silver Price":
            series_data = df_filtered['SILVER_PRICE']
        else:
            series_data = df_filtered['GPRD']
        
        with stats_col1:
            st.metric("Mean", f"{series_data.mean():.2f}")
        with stats_col2:
            st.metric("Std Dev", f"{series_data.std():.2f}")
        with stats_col3:
            st.metric("Min", f"{series_data.min():.2f}")
        with stats_col4:
            st.metric("Max", f"{series_data.max():.2f}")
    
    with col2:
        st.header("📋 Dataset Info")
        
        st.metric("Total Records", len(df_raw))
        st.metric("Date Range", f"{min_date} to {max_date}")
        st.metric("Total Days", (max_date - min_date).days)
        
        st.markdown("---")
        
        st.subheader("Current Values")
        latest_data = df_raw.iloc[-1]
        st.metric("Gold Price", f"${latest_data['GOLD_PRICE']:.2f}")
        st.metric("Silver Price", f"${latest_data['SILVER_PRICE']:.2f}")
        st.metric("GPRD Index", f"{latest_data['GPRD']:.2f}")
    
    # Model forecasts section
    st.markdown("---")
    st.header("🤖 Model Forecasts")
    
    if forecasts is not None and model is not None:
        # Filter forecasts by date
        forecast_mask = (forecasts['DATE'].dt.date >= start_date) & (forecasts['DATE'].dt.date <= end_date)
        forecasts_filtered = forecasts[forecast_mask]
        
        if len(forecasts_filtered) > 0:
            # Forecast plot
            fig2, ax2 = plt.subplots(figsize=(14, 5))
            ax2.plot(forecasts_filtered['DATE'], forecasts_filtered['GOLD_ACTUAL'], 
                    label='Actual', alpha=0.8, linewidth=2, color='steelblue')
            ax2.plot(forecasts_filtered['DATE'], forecasts_filtered['GOLD_PREDICTED'], 
                    label='Predicted', alpha=0.8, linewidth=2, color='orange')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Gold Price (USD)', fontsize=12)
            ax2.set_title('Gold Price: Actual vs Predicted (XGBoost Model)', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)
            
            # Model metrics
            st.subheader("📊 Model Performance Metrics")
            
            actual = forecasts_filtered['GOLD_ACTUAL'].values
            predicted = forecasts_filtered['GOLD_PREDICTED'].values
            
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mae = np.mean(np.abs(actual - predicted))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("RMSE", f"{rmse:.2f}")
            with metric_col2:
                st.metric("MAE", f"{mae:.2f}")
            with metric_col3:
                st.metric("MAPE", f"{mape:.2f}%")
            
            # Error distribution
            st.subheader("📉 Prediction Error Distribution")
            errors = actual - predicted
            
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='teal')
            ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
            ax3.set_xlabel('Prediction Error (USD)', fontsize=12)
            ax3.set_ylabel('Frequency', fontsize=12)
            ax3.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig3)
        else:
            st.info("No forecast data available for the selected date range.")
    else:
        st.warning("⚠️ Model or forecast data not found. Please run the Jupyter notebook to train the model first.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Data Source:** Gold-Silver Price VS Geopolitical Risk (1985–2025) - Kaggle  
    **Model:** XGBoost with lag features, rolling statistics, and calendar features  
    **Author:** Portfolio Project
    """)

if __name__ == "__main__":
    main()
