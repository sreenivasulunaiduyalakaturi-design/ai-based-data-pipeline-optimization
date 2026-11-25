"""Generate a synthetic, slightly dirty dataset (e.g. sales transactions)."""
import numpy as np
import pandas as pd

import config


def generate_raw_data(n_rows: int = config.N_ROWS, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "transaction_id": np.arange(n_rows),
        "customer_id": rng.integers(1, 5_000, n_rows),
        "product": rng.choice(["A", "B", "C", "D", "E"], n_rows),
        "quantity": rng.integers(1, 20, n_rows).astype(float),
        "unit_price": rng.normal(50, 15, n_rows).round(2),
        "country": rng.choice(["IN", "US", "UK", "DE", "FR"], n_rows),
    })
    # Inject dirt: missing values, negative prices, duplicates
    df.loc[rng.choice(n_rows, n_rows // 50, replace=False), "quantity"] = np.nan
    df.loc[rng.choice(n_rows, n_rows // 100, replace=False), "unit_price"] = -1
    dupes = df.sample(n_rows // 50, random_state=seed)
    return pd.concat([df, dupes], ignore_index=True)


def main():
    config.DATA_DIR.mkdir(exist_ok=True)
    df = generate_raw_data()
    df.to_csv(config.RAW_DATA_PATH, index=False)
    print(f"Generated {len(df):,} rows -> {config.RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
