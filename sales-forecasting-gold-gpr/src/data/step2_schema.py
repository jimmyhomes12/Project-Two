from pathlib import Path
import pandas as pd
import re

RAW_DIR = Path("data/raw")

def pick_main_csv(raw_dir: Path) -> Path:
    csvs = sorted(raw_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in: {raw_dir.resolve()}")

    preferred = []
    for p in csvs:
        name = p.name.lower()
        score = 0
        if "combined" in name or "merge" in name or "final" in name: score += 5
        if "gold" in name: score += 2
        if "silver" in name: score += 2
        if "gpr" in name or "geopolitical" in name or "risk" in name: score += 2
        preferred.append((score, p))

    preferred.sort(reverse=True, key=lambda x: x[0])
    return preferred[0][1]

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # normalize column names: lowercase + underscores
    df.columns = [re.sub(r"[^a-z0-9]+", "_", c.strip().lower()) for c in df.columns]


    # date column
    date_candidates = [c for c in df.columns if c in ("date", "datetime", "time", "timestamp")]
    if not date_candidates:
        date_candidates = [c for c in df.columns if "date" in c]
    if not date_candidates:
        raise ValueError(f"Could not find a date column. Columns: {df.columns.tolist()}")
    date_col = date_candidates[0]

    # gold column
    gold_candidates = [c for c in df.columns if "gold" in c and ("price" in c or "usd" in c or c == "gold")]
    if not gold_candidates:
        gold_candidates = [c for c in df.columns if "gold" in c]
    if not gold_candidates:
        raise ValueError("Could not find a gold column.")
    gold_col = gold_candidates[0]

    # silver column
    silver_candidates = [c for c in df.columns if "silver" in c and ("price" in c or "usd" in c or c == "silver")]
    if not silver_candidates:
        silver_candidates = [c for c in df.columns if "silver" in c]
    if not silver_candidates:
        raise ValueError("Could not find a silver column.")
    silver_col = silver_candidates[0]

    # gpr column
    gpr_candidates = [c for c in df.columns if c in ("gpr", "geopolitical_risk", "geopolitical_risk_index")]
    if not gpr_candidates:
        gpr_candidates = [c for c in df.columns if "gpr" in c or ("geopolitical" in c and "risk" in c) or c == "risk"]
    if not gpr_candidates:
        raise ValueError("Could not find a geopolitical risk (GPR) column.")
    gpr_col = gpr_candidates[0]

    # rename to standard schema
    df = df.rename(columns={
        date_col: "date",
        gold_col: "gold_price",
        silver_col: "silver_price",
        gpr_col: "gpr",
    })

    # enforce dtypes
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["gold_price", "silver_price", "gpr"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # sort by date + clean
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    return df

def main():
    main_csv = pick_main_csv(RAW_DIR)
    print(f"\nUsing CSV: {main_csv}\n")

    df_raw = pd.read_csv(main_csv)
    print("Raw columns:", list(df_raw.columns))

    df = standardize_columns(df_raw)

    print("\nStandardized columns:", list(df.columns))
    print("\nHead:\n", df.head())
    print("\nInfo:\n")
    print(df.info())

    print("\nDate range:", df["date"].min(), "→", df["date"].max())
    print("\nMissing (%):")
    print((df[["gold_price", "silver_price", "gpr"]].isna().mean() * 100).round(2))

if __name__ == "__main__":
    main()
