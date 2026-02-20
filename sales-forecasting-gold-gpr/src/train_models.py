"""
Phase 4 – Train 8 XGBoost models (one per metal) and save artefacts.

Metals: GOLD, SILVER, PLATINUM, PALLADIUM, COPPER, ALUMINUM, NICKEL, ZINC

Usage:
    python src/train_models.py
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METAL_NAMES = ["GOLD", "SILVER", "PLATINUM", "PALLADIUM",
               "COPPER", "ALUMINUM", "NICKEL", "ZINC"]

PRICE_COLS = [f"{m}_PRICE" for m in METAL_NAMES]
GPRD_COLS = ["GPRD", "GPRD_ACT", "GPRD_THREAT"]
TARGET_COLS = [f"{m}_TARGET" for m in METAL_NAMES]

EXCLUDE_COLS = PRICE_COLS + GPRD_COLS + TARGET_COLS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load the combined multi-metal dataset; fall back to Gold/Silver only."""
    combined_path = DATA_RAW_DIR / "all_metals_gprd.csv"
    if combined_path.exists():
        df = pd.read_csv(combined_path, index_col=0, parse_dates=True)
        print(f"Loaded combined dataset: {df.shape}  ({combined_path.name})")
    else:
        base_path = DATA_RAW_DIR / "Gold-Silver-GeopoliticalRisk_HistoricalData.csv"
        df = pd.read_csv(base_path)
        df.columns = [c.strip().upper() for c in df.columns]
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.sort_values("DATE").set_index("DATE")
        df = df[["GOLD_PRICE", "SILVER_PRICE", "GPRD", "GPRD_ACT", "GPRD_THREAT"]]
        df = df.ffill().bfill()
        print(f"Loaded base dataset (Gold + Silver only): {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag, rolling, time, and target features for all available metals."""
    feat = df.copy()

    available_metals = [m for m in METAL_NAMES if f"{m}_PRICE" in feat.columns]
    lag_periods = [1, 2, 3, 5, 10]
    windows = [5, 10, 20, 30]

    # Lag features
    for metal in available_metals:
        col = f"{metal}_PRICE"
        for lag in lag_periods:
            feat[f"{metal}_LAG_{lag}"] = feat[col].shift(lag)

    for gprd in GPRD_COLS:
        if gprd in feat.columns:
            for lag in lag_periods:
                feat[f"{gprd}_LAG_{lag}"] = feat[gprd].shift(lag)

    # Rolling statistics
    for metal in available_metals:
        col = f"{metal}_PRICE"
        for w in windows:
            feat[f"{metal}_ROLL_MEAN_{w}"] = feat[col].rolling(w).mean()
            feat[f"{metal}_ROLL_STD_{w}"] = feat[col].rolling(w).std()

    for gprd in GPRD_COLS:
        if gprd in feat.columns:
            for w in windows:
                feat[f"{gprd}_ROLL_MEAN_{w}"] = feat[gprd].rolling(w).mean()
                feat[f"{gprd}_ROLL_STD_{w}"] = feat[gprd].rolling(w).std()

    # Time features
    feat["YEAR"] = feat.index.year
    feat["MONTH"] = feat.index.month
    feat["QUARTER"] = feat.index.quarter
    feat["DAYOFWEEK"] = feat.index.dayofweek
    feat["DAYOFYEAR"] = feat.index.dayofyear

    # Target variables (next-day price)
    for metal in available_metals:
        feat[f"{metal}_TARGET"] = feat[f"{metal}_PRICE"].shift(-1)

    feat.dropna(inplace=True)
    print(f"Feature engineering complete: {feat.shape}")
    return feat


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_all_metals(df_feat: pd.DataFrame) -> dict:
    """Train one XGBoost model per metal; return performance dict."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    results: dict = {}

    for metal in METAL_NAMES:
        target_col = f"{metal}_TARGET"

        if target_col not in df_feat.columns:
            print(f"\n✗ Skipping {metal}: no target column found")
            continue

        print(f"\n{'=' * 60}")
        print(f"TRAINING: {metal}")
        print(f"{'=' * 60}")

        feature_cols = [c for c in df_feat.columns if c not in EXCLUDE_COLS]
        X = df_feat[feature_cols]
        y = df_feat[target_col]

        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"Samples: {len(X)}, Features: {len(feature_cols)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")

        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        )

        print("Training... ", end="", flush=True)
        model.fit(X_train, y_train)
        print("✓")

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        print(f"\nPerformance:")
        print(f"  Train RMSE: ${train_rmse:.4f}")
        print(f"  Test RMSE:  ${test_rmse:.4f}")
        print(f"  Train MAE:  ${train_mae:.4f}")
        print(f"  Test MAE:   ${test_mae:.4f}")

        results[metal] = {
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "n_samples": len(X),
            "n_features": len(feature_cols),
        }

        # Save model
        model_path = MODELS_DIR / f"{metal.lower()}_xgb_model.pkl"
        joblib.dump(model, model_path)
        print(f"✓ Saved model  → {model_path}")

        # Save feature columns
        feat_path = MODELS_DIR / f"{metal.lower()}_feature_cols.pkl"
        joblib.dump(feature_cols, feat_path)
        print(f"✓ Saved features → {feat_path}")

        # Save test predictions (for dashboard)
        pred_df = pd.DataFrame(
            {
                "DATE": y_test.index,
                f"{metal}_ACTUAL": y_test.values,
                f"{metal}_PREDICTED": y_test_pred,
            }
        )
        pred_path = OUTPUTS_DIR / "forecasts" / f"{metal.lower()}_val_forecasts_xgb.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_path, index=False)
        print(f"✓ Saved forecasts → {pred_path}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("PHASE 4: TRAIN 8 XGBOOST MODELS")
    print("=" * 60)

    df_raw = load_data()
    df_feat = engineer_features(df_raw)

    results = train_all_metals(df_feat)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE – PERFORMANCE SUMMARY")
    print("=" * 60)

    if results:
        df_results = pd.DataFrame(results).T.round(4)
        print(df_results.to_string())

        perf_path = OUTPUTS_DIR / "model_performance_all_metals.csv"
        df_results.to_csv(perf_path)
        print(f"\n✓ Performance summary → {perf_path}")
        print("\n🎉 All models trained successfully!")
    else:
        print("No models were trained.")


if __name__ == "__main__":
    main()
