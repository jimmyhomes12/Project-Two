"""
Multi-Metal Price Forecasting Dashboard
Interactive Streamlit app for exploring precious/industrial metal prices,
geopolitical risk, and XGBoost model forecasts for 8 metals.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from pathlib import Path
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Multi-Metal Price Forecasting",
    layout="wide",
)

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_FORECASTS_DIR = OUTPUTS_DIR / "forecasts"

# Metal configuration
METAL_NAMES = ["Gold", "Silver", "Platinum", "Palladium",
               "Copper", "Aluminum", "Nickel", "Zinc"]

METAL_COLORS = {
    "Gold": "#FFD700",
    "Silver": "#C0C0C0",
    "Platinum": "#E5E4E2",
    "Palladium": "#CED0DD",
    "Copper": "#B87333",
    "Aluminum": "#848789",
    "Nickel": "#727472",
    "Zinc": "#7C7F82",
}

# ==============================================================================
# LOAD DATA
# ==============================================================================


@st.cache_data
def load_data() -> pd.DataFrame | None:
    """Load multi-metal + GPRD dataset."""
    engineered = DATA_DIR / "metals_features_engineered.csv"
    combined = DATA_RAW_DIR / "all_metals_gprd.csv"
    fallback = DATA_RAW_DIR / "Gold-Silver-GeopoliticalRisk_HistoricalData.csv"

    if engineered.exists():
        df = pd.read_csv(engineered, index_col=0, parse_dates=True)
    elif combined.exists():
        df = pd.read_csv(combined, index_col=0, parse_dates=True)
    elif fallback.exists():
        df = pd.read_csv(fallback)
        df.columns = [c.strip().upper() for c in df.columns]
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.sort_values("DATE").set_index("DATE")
        keep = ["GOLD_PRICE", "SILVER_PRICE", "GPRD", "GPRD_ACT", "GPRD_THREAT"]
        df = df[[c for c in keep if c in df.columns]]
    else:
        st.error("Data file not found. Run the notebook to generate data.")
        return None

    return df.ffill().bfill()


@st.cache_data
def load_model_performance() -> pd.DataFrame | None:
    """Load the saved performance summary CSV."""
    path = OUTPUTS_DIR / "model_performance_all_metals.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return None


@st.cache_resource
def load_model(metal: str):
    """Load trained model and feature columns for a specific metal."""
    model_path = MODELS_DIR / f"{metal.lower()}_xgb_model.pkl"
    feature_path = MODELS_DIR / f"{metal.lower()}_feature_cols.pkl"

    if not model_path.exists():
        return None, None

    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_path) if feature_path.exists() else None
    return model, feature_cols


@st.cache_data
def load_metal_forecasts(metal: str) -> pd.DataFrame | None:
    """Load pre-computed test-set forecasts for the given metal."""
    path = OUTPUTS_FORECASTS_DIR / f"{metal.lower()}_val_forecasts_xgb.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["DATE"] = pd.to_datetime(df["DATE"])
        return df
    return None


# ==============================================================================
# HEADER
# ==============================================================================

st.title("🏆 Multi-Metal Price Forecasting with Geopolitical Risk")
st.markdown(
    "**Predict next-day prices for 8 valuable metals using XGBoost models**  \n"
    "Data: Gold, Silver, Platinum, Palladium, Copper, Aluminum, Nickel, Zinc + GPRD indices"
)

# ==============================================================================
# SIDEBAR
# ==============================================================================

st.sidebar.header("⚙️ Controls")

df_feat = load_data()

if df_feat is not None:
    min_date = df_feat.index.min().date()
    max_date = df_feat.index.max().date()

    start_date = st.sidebar.date_input(
        "Start Date",
        value=pd.to_datetime("2020-01-01").date(),
        min_value=min_date,
        max_value=max_date,
    )

    end_date = st.sidebar.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )

    # Metal selector for individual forecasts
    selected_metal = st.sidebar.selectbox(
        "Select Metal for Forecast",
        METAL_NAMES,
    )

    # Multi-select for comparison
    comparison_metals = st.sidebar.multiselect(
        "Select Metals for Comparison",
        METAL_NAMES,
        default=["Gold", "Silver", "Platinum"],
    )

# ==============================================================================
# MAIN CONTENT
# ==============================================================================

if df_feat is None:
    st.stop()

# Filter data by date range
df_filtered = df_feat[
    (df_feat.index.date >= start_date) & (df_feat.index.date <= end_date)
]

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "📈 Historical Prices",
    "🤖 Model Forecasts",
    "📊 Metal Comparison",
])

# ==============================================================================
# TAB 1: HISTORICAL PRICES
# ==============================================================================

with tab1:
    st.subheader("Historical Metal Prices")

    fig = go.Figure()

    for metal in comparison_metals:
        price_col = f"{metal.upper()}_PRICE"
        if price_col in df_filtered.columns:
            fig.add_trace(go.Scatter(
                x=df_filtered.index,
                y=df_filtered[price_col],
                name=metal,
                line=dict(color=METAL_COLORS.get(metal, "#000000")),
            ))

    fig.update_layout(
        title="Metal Prices Over Time",
        xaxis_title="Date",
        yaxis_title="Price (USD per troy oz)",
        hovermode="x unified",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show GPRD
    st.subheader("Geopolitical Risk Index (GPRD)")

    fig_gprd = go.Figure()

    for gprd_col in ["GPRD", "GPRD_ACT", "GPRD_THREAT"]:
        if gprd_col in df_filtered.columns:
            fig_gprd.add_trace(go.Scatter(
                x=df_filtered.index,
                y=df_filtered[gprd_col],
                name=gprd_col,
                mode="lines",
            ))

    fig_gprd.update_layout(
        title="GPRD Components",
        xaxis_title="Date",
        yaxis_title="Risk Index",
        hovermode="x unified",
        height=400,
    )

    st.plotly_chart(fig_gprd, use_container_width=True)

# ==============================================================================
# TAB 2: MODEL FORECASTS
# ==============================================================================

with tab2:
    st.subheader(f"Model Forecast: {selected_metal}")

    # Try loading pre-computed forecasts first; fall back to live prediction
    forecasts = load_metal_forecasts(selected_metal)
    model, feature_cols = load_model(selected_metal)

    if forecasts is not None and not forecasts.empty:
        actual_col = f"{selected_metal.upper()}_ACTUAL"
        pred_col = f"{selected_metal.upper()}_PREDICTED"

        fm = (forecasts["DATE"].dt.date >= start_date) & (
            forecasts["DATE"].dt.date <= end_date
        )
        filt = forecasts[fm]

        if len(filt) > 0:
            actual_vals = filt[actual_col].values
            pred_vals = filt[pred_col].values

            rmse = np.sqrt(np.mean((actual_vals - pred_vals) ** 2))
            mae = np.mean(np.abs(actual_vals - pred_vals))
            ss_res = np.sum((actual_vals - pred_vals) ** 2)
            ss_tot = np.sum((actual_vals - actual_vals.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"${rmse:.2f}")
            col2.metric("MAE", f"${mae:.2f}")
            col3.metric("R² Score", f"{r2:.3f}")

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=filt["DATE"],
                y=actual_vals,
                name="Actual",
                mode="lines",
                line=dict(color="blue"),
            ))
            fig2.add_trace(go.Scatter(
                x=filt["DATE"],
                y=pred_vals,
                name="Predicted",
                mode="lines",
                line=dict(color="orange", dash="dash"),
            ))
            fig2.update_layout(
                title=f"{selected_metal} Price: Actual vs Predicted",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                hovermode="x unified",
                height=500,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No forecast data available for the selected date range.")

    elif model is not None and feature_cols is not None:
        target_col = f"{selected_metal.upper()}_TARGET"
        if target_col not in df_filtered.columns:
            st.warning(f"No target data for {selected_metal}")
        else:
            try:
                X = df_filtered[feature_cols].dropna()
                y_actual = df_filtered.loc[X.index, target_col]
                y_pred = model.predict(X)

                ss_res = np.sum((y_actual.values - y_pred) ** 2)
                ss_tot = np.sum((y_actual.values - y_actual.mean()) ** 2)
                rmse = np.sqrt(np.mean((y_actual.values - y_pred) ** 2))
                mae = np.mean(np.abs(y_actual.values - y_pred))
                r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

                col1, col2, col3 = st.columns(3)
                col1.metric("RMSE", f"${rmse:.2f}")
                col2.metric("MAE", f"${mae:.2f}")
                col3.metric("R² Score", f"{r2:.3f}")

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=X.index, y=y_actual,
                    name="Actual", mode="lines",
                    line=dict(color="blue"),
                ))
                fig2.add_trace(go.Scatter(
                    x=X.index, y=y_pred,
                    name="Predicted", mode="lines",
                    line=dict(color="orange", dash="dash"),
                ))
                fig2.update_layout(
                    title=f"{selected_metal} Price: Actual vs Predicted",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode="x unified",
                    height=500,
                )
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating forecast: {e}")
    else:
        st.error(
            f"Model for {selected_metal} not found. "
            "Run the notebook to train models."
        )

# ==============================================================================
# TAB 3: METAL COMPARISON
# ==============================================================================

with tab3:
    st.subheader("Cross-Metal Analysis")

    # Model performance table
    perf_df = load_model_performance()

    if perf_df is not None:
        st.write("### Model Performance Comparison")
        show_cols = [c for c in ["train_rmse", "test_rmse", "train_mae", "test_mae"]
                     if c in perf_df.columns]
        st.dataframe(
            perf_df[show_cols].style.highlight_min(axis=0, color="lightgreen"),
            use_container_width=True,
        )

        # Bar chart of test RMSE
        fig_rmse = go.Figure(data=[
            go.Bar(
                x=perf_df.index.tolist(),
                y=perf_df["test_rmse"],
                marker_color=[METAL_COLORS.get(m, "#000000") for m in perf_df.index],
            )
        ])
        fig_rmse.update_layout(
            title="Test RMSE by Metal",
            xaxis_title="Metal",
            yaxis_title="RMSE (USD)",
            height=400,
        )
        st.plotly_chart(fig_rmse, use_container_width=True)
    else:
        st.info("Performance data not available. Train models first.")

    # Correlation heatmap
    st.write("### Metal Price Correlations")

    price_cols = [f"{m.upper()}_PRICE" for m in METAL_NAMES
                  if f"{m.upper()}_PRICE" in df_filtered.columns]

    if len(price_cols) > 1:
        corr_matrix = df_filtered[price_cols].corr()

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[col.replace("_PRICE", "") for col in corr_matrix.columns],
            y=[col.replace("_PRICE", "") for col in corr_matrix.index],
            colorscale="RdBu",
            zmid=0,
        ))
        fig_corr.update_layout(
            title="Correlation Matrix",
            height=500,
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Price correlation data not available for the selected date range.")

# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown("---")
st.caption(
    f"Data: 8 metals (FRED + MetalPriceAPI) + GPRD • "
    f"Models: XGBoost per metal (lags, rolling, time features) • "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)
