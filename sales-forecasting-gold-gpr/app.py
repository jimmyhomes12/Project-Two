"""
Multi-Metal Price Forecasting Dashboard
Interactive Streamlit app for exploring precious/industrial metal prices,
geopolitical risk, and XGBoost model forecasts for 8 metals.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from datetime import timedelta

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Metal Price Forecasting",
    page_icon="📈",
    layout="wide",
)

sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_FORECASTS_DIR = BASE_DIR / "outputs" / "forecasts"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METAL_NAMES = ["GOLD", "SILVER", "PLATINUM", "PALLADIUM",
               "COPPER", "ALUMINUM", "NICKEL", "ZINC"]

METAL_COLORS = {
    "GOLD": "goldenrod",
    "SILVER": "slategrey",
    "PLATINUM": "steelblue",
    "PALLADIUM": "mediumpurple",
    "COPPER": "sienna",
    "ALUMINUM": "teal",
    "NICKEL": "olivedrab",
    "ZINC": "darkcyan",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load multi-metal + GPRD dataset (falls back to Gold/Silver only)."""
    combined = DATA_RAW_DIR / "all_metals_gprd.csv"
    if combined.exists():
        df = pd.read_csv(combined, index_col=0, parse_dates=True)
    else:
        base = DATA_RAW_DIR / "Gold-Silver-GeopoliticalRisk_HistoricalData.csv"
        df = pd.read_csv(base)
        df.columns = [c.strip().upper() for c in df.columns]
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.sort_values("DATE").set_index("DATE")
        keep = ["GOLD_PRICE", "SILVER_PRICE", "GPRD", "GPRD_ACT", "GPRD_THREAT"]
        df = df[[c for c in keep if c in df.columns]]
    return df.ffill().bfill()


@st.cache_data
def load_model_performance() -> pd.DataFrame | None:
    """Load the saved performance summary CSV."""
    path = BASE_DIR / "outputs" / "model_performance_all_metals.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return None


@st.cache_resource
def load_metal_model(metal: str):
    """Load a trained XGBoost model for the given metal."""
    path = MODELS_DIR / f"{metal.lower()}_xgb_model.pkl"
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_data
def load_metal_forecasts(metal: str) -> pd.DataFrame | None:
    """Load pre-computed test-set forecasts for the given metal."""
    path = OUTPUTS_FORECASTS_DIR / f"{metal.lower()}_val_forecasts_xgb.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["DATE"] = pd.to_datetime(df["DATE"])
        return df
    return None


def get_metals_with_forecasts(df: pd.DataFrame) -> list[str]:
    """Return metals that have both a PRICE column and a saved forecast file."""
    return [
        m for m in METAL_NAMES
        if f"{m}_PRICE" in df.columns
        and (OUTPUTS_FORECASTS_DIR / f"{m.lower()}_val_forecasts_xgb.csv").exists()
    ]


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("📈 Multi-Metal Price Forecasting Dashboard")
    st.markdown(
        "Explore historical prices for **8 metals**, geopolitical risk indices, "
        "and individual XGBoost model forecasts.  "
        "Models trained with lag features, rolling statistics, and calendar features."
    )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    with st.spinner("Loading data…"):
        df_raw = load_data()
        perf_df = load_model_performance()

    # ------------------------------------------------------------------
    # Sidebar – shared controls
    # ------------------------------------------------------------------
    st.sidebar.header("⚙️ Settings")

    min_date = df_raw.index.min().date()
    max_date = df_raw.index.max().date()

    start_date = st.sidebar.date_input(
        "Start Date",
        value=max_date - timedelta(days=365 * 2),
        min_value=min_date,
        max_value=max_date,
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )

    series_option = st.sidebar.selectbox(
        "Select Series to View",
        ["Gold Price", "Silver Price", "Geopolitical Risk (GPRD)"],
    )

    # ------------------------------------------------------------------
    # Historical data section
    # ------------------------------------------------------------------
    mask = (df_raw.index.date >= start_date) & (df_raw.index.date <= end_date)
    df_filtered = df_raw[mask]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📊 Historical Data")

        fig, ax = plt.subplots(figsize=(12, 5))
        if series_option == "Gold Price":
            ax.plot(df_filtered.index, df_filtered["GOLD_PRICE"],
                    color="goldenrod", linewidth=2)
            ax.set_ylabel("Price (USD)", fontsize=12)
            ax.set_title(f"Gold Spot Price ({start_date} to {end_date})",
                         fontsize=14, fontweight="bold")
            series_data = df_filtered["GOLD_PRICE"]
        elif series_option == "Silver Price":
            ax.plot(df_filtered.index, df_filtered["SILVER_PRICE"],
                    color="slategrey", linewidth=2)
            ax.set_ylabel("Price (USD)", fontsize=12)
            ax.set_title(f"Silver Spot Price ({start_date} to {end_date})",
                         fontsize=14, fontweight="bold")
            series_data = df_filtered["SILVER_PRICE"]
        else:
            ax.plot(df_filtered.index, df_filtered["GPRD"],
                    color="crimson", linewidth=2)
            ax.set_ylabel("GPRD Index", fontsize=12)
            ax.set_title(f"Geopolitical Risk Index ({start_date} to {end_date})",
                         fontsize=14, fontweight="bold")
            series_data = df_filtered["GPRD"]

        ax.set_xlabel("Date", fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("📈 Summary Statistics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Mean", f"{series_data.mean():.2f}")
        with c2:
            st.metric("Std Dev", f"{series_data.std():.2f}")
        with c3:
            st.metric("Min", f"{series_data.min():.2f}")
        with c4:
            st.metric("Max", f"{series_data.max():.2f}")

    with col2:
        st.header("📋 Dataset Info")
        st.metric("Total Records", len(df_raw))
        st.metric("Date Range", f"{min_date} to {max_date}")
        st.metric("Total Days", (max_date - min_date).days)

        st.markdown("---")
        st.subheader("Current Values")
        latest = df_raw.iloc[-1]
        st.metric("Gold Price", f"${latest['GOLD_PRICE']:.2f}")
        st.metric("Silver Price", f"${latest['SILVER_PRICE']:.2f}")
        if "GPRD" in latest.index:
            st.metric("GPRD Index", f"{latest['GPRD']:.2f}")

    # ------------------------------------------------------------------
    # Model Forecasts – 8 metals
    # ------------------------------------------------------------------
    st.markdown("---")
    st.header("🤖 Model Forecasts (8 metals)")

    metals_with_forecasts = get_metals_with_forecasts(df_raw)

    if not metals_with_forecasts:
        st.warning(
            "⚠️ No trained models found.  "
            "Run `python src/train_models.py` from the project root to train all 8 models."
        )
    else:
        selected_metal = st.selectbox(
            "Select Metal",
            options=metals_with_forecasts,
            format_func=lambda m: m.capitalize(),
        )

        forecasts = load_metal_forecasts(selected_metal)
        model = load_metal_model(selected_metal)

        if forecasts is not None and not forecasts.empty:
            actual_col = f"{selected_metal}_ACTUAL"
            pred_col = f"{selected_metal}_PREDICTED"

            # Apply date filter
            fm = (forecasts["DATE"].dt.date >= start_date) & (
                forecasts["DATE"].dt.date <= end_date
            )
            filt = forecasts[fm]

            if len(filt) > 0:
                color = METAL_COLORS.get(selected_metal, "steelblue")

                # Actual vs Predicted chart
                fig2, ax2 = plt.subplots(figsize=(14, 5))
                ax2.plot(filt["DATE"], filt[actual_col],
                         label="Actual", alpha=0.8, linewidth=2, color=color)
                ax2.plot(filt["DATE"], filt[pred_col],
                         label="Predicted", alpha=0.8, linewidth=2,
                         color="orange", linestyle="--")
                ax2.set_xlabel("Date", fontsize=12)
                ax2.set_ylabel("Price (USD)", fontsize=12)
                ax2.set_title(
                    f"{selected_metal.capitalize()} Price – Actual vs Predicted (XGBoost)",
                    fontsize=14, fontweight="bold",
                )
                ax2.legend(fontsize=11)
                ax2.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig2)

                # Metrics
                st.subheader("📊 Model Performance Metrics")
                actual_vals = filt[actual_col].values
                pred_vals = filt[pred_col].values

                rmse = np.sqrt(np.mean((actual_vals - pred_vals) ** 2))
                mae = np.mean(np.abs(actual_vals - pred_vals))
                denom = np.where(actual_vals == 0, np.nan, actual_vals)
                with np.errstate(divide="ignore", invalid="ignore"):
                    mape = np.mean(np.abs((actual_vals - pred_vals) / denom)) * 100

                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric("RMSE", f"{rmse:.4f}")
                with mc2:
                    st.metric("MAE", f"{mae:.4f}")
                with mc3:
                    st.metric("MAPE", f"{mape:.2f}%")

                # Error distribution
                st.subheader("📉 Prediction Error Distribution")
                errors = actual_vals - pred_vals
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                ax3.hist(errors, bins=50, edgecolor="black", alpha=0.7, color="teal")
                ax3.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero Error")
                ax3.set_xlabel("Prediction Error (USD)", fontsize=12)
                ax3.set_ylabel("Frequency", fontsize=12)
                ax3.set_title("Distribution of Prediction Errors", fontsize=14, fontweight="bold")
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig3)

            else:
                st.info("No forecast data available for the selected date range.")
        else:
            st.warning(f"No forecast file found for {selected_metal}.")

        # ------------------------------------------------------------------
        # Performance summary table
        # ------------------------------------------------------------------
        if perf_df is not None:
            st.markdown("---")
            st.subheader("📋 All-Metal Performance Summary")
            show_cols = [c for c in ["train_rmse", "test_rmse", "train_mae", "test_mae"] if c in perf_df.columns]
            st.dataframe(perf_df[show_cols].style.format("{:.4f}"), use_container_width=True)

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        "**Data Source:** Gold-Silver Price VS Geopolitical Risk (1985–2025) – Kaggle  \n"
        "**Model:** XGBoost with lag features, rolling statistics, and calendar features  \n"
        "**Author:** Portfolio Project"
    )


if __name__ == "__main__":
    main()
