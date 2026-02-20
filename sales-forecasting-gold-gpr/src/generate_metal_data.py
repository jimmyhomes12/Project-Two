"""
Generate synthetic multi-metal price data for PLATINUM, PALLADIUM,
COPPER, ALUMINUM, NICKEL, ZINC using the existing Gold/Silver/GPRD
dataset as a base.  Prices are simulated via correlated Geometric
Brownian Motion (GBM) with historically realistic parameters so the
resulting data is suitable for training portfolio-demonstration models.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_PATH = DATA_RAW_DIR / "all_metals_gprd.csv"

# ---------------------------------------------------------------------------
# Historically-inspired parameters (annual drift / annual vol)
# Starting prices are approximate values around 1985-01-01
# ---------------------------------------------------------------------------
EPSILON = 1e-12  # small value to prevent division by zero

METAL_PARAMS = {
    "PLATINUM": {"start_price": 320.0,  "mu": 0.04, "sigma": 0.22},
    "PALLADIUM": {"start_price": 70.0,   "mu": 0.06, "sigma": 0.32},
    "COPPER":    {"start_price": 0.65,   "mu": 0.03, "sigma": 0.20},
    "ALUMINUM":  {"start_price": 0.55,   "mu": 0.01, "sigma": 0.15},
    "NICKEL":    {"start_price": 2.40,   "mu": 0.03, "sigma": 0.25},
    "ZINC":      {"start_price": 0.40,   "mu": 0.02, "sigma": 0.18},
}

# Correlation with daily Gold log-returns for each synthetic metal
GOLD_CORRELATIONS = {
    "PLATINUM":  0.70,
    "PALLADIUM": 0.55,
    "COPPER":    0.35,
    "ALUMINUM":  0.25,
    "NICKEL":    0.30,
    "ZINC":      0.28,
}


def simulate_correlated_gbm(
    gold_log_returns: np.ndarray,
    start_price: float,
    mu: float,
    sigma: float,
    rho: float,
    seed: int,
) -> np.ndarray:
    """
    Simulate daily prices via GBM correlated with gold log-returns.

    Parameters
    ----------
    gold_log_returns : daily log-returns of gold (length N-1 for N dates)
    start_price      : initial price on the first trading day
    mu               : annualised drift (used only for the idiosyncratic part)
    sigma            : annualised volatility
    rho              : Pearson correlation with gold log-returns
    seed             : random seed for reproducibility

    Returns
    -------
    prices : np.ndarray of length == len(gold_log_returns) + 1
    """
    rng = np.random.default_rng(seed)
    n = len(gold_log_returns)

    dt = 1 / 252  # one trading day

    # Idiosyncratic noise (independent of gold)
    eps_idio = rng.standard_normal(n)

    # Correlated noise = rho * gold_shock + sqrt(1-rho²) * idiosyncratic
    gold_shocks = gold_log_returns / (gold_log_returns.std() + EPSILON)
    corr_noise = rho * gold_shocks + np.sqrt(1 - rho ** 2) * eps_idio

    # Daily log-returns via GBM
    daily_log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * corr_noise

    prices = np.empty(n + 1)
    prices[0] = start_price
    prices[1:] = start_price * np.exp(np.cumsum(daily_log_returns))
    return prices


def main() -> None:
    print("=" * 60)
    print("GENERATING MULTI-METAL DATA")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load base dataset (Gold, Silver, GPRD)
    # ------------------------------------------------------------------
    base_path = DATA_RAW_DIR / "Gold-Silver-GeopoliticalRisk_HistoricalData.csv"
    df = pd.read_csv(base_path)
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").set_index("DATE")
    df = df[["GOLD_PRICE", "SILVER_PRICE", "GPRD", "GPRD_ACT", "GPRD_THREAT"]]
    df = df.ffill().bfill()

    print(f"Base dataset: {len(df)} rows  ({df.index.min().date()} → {df.index.max().date()})")

    # ------------------------------------------------------------------
    # Compute daily gold log-returns for correlation
    # ------------------------------------------------------------------
    gold_log_ret = np.log(df["GOLD_PRICE"] / df["GOLD_PRICE"].shift(1)).dropna().values

    # ------------------------------------------------------------------
    # Simulate each synthetic metal
    # ------------------------------------------------------------------
    df_out = df.copy()

    for idx, (metal, params) in enumerate(METAL_PARAMS.items()):
        prices = simulate_correlated_gbm(
            gold_log_returns=gold_log_ret,
            start_price=params["start_price"],
            mu=params["mu"],
            sigma=params["sigma"],
            rho=GOLD_CORRELATIONS[metal],
            seed=42 + idx,
        )
        # prices has length == len(df); first value aligns with df.index[0]
        df_out[f"{metal}_PRICE"] = prices[: len(df)]
        print(f"  ✓ {metal}: start=${params['start_price']:.2f}  "
              f"end=${prices[len(df) - 1]:.2f}")

    # ------------------------------------------------------------------
    # Save combined dataset
    # ------------------------------------------------------------------
    df_out.to_csv(OUTPUT_PATH)
    print(f"\n✓ Combined dataset saved → {OUTPUT_PATH}")
    print(f"  Shape : {df_out.shape}")
    print(f"  Columns: {list(df_out.columns)}")


if __name__ == "__main__":
    main()
