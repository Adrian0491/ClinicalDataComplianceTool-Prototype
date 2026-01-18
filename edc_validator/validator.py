import os
import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest
from edc_validator.sdtm_rules import validate_dm
from datetime import datetime

OUTPUT_DIR = "output"
DEFAULT_INPUT_FILE = os.path.join("mock_data", "mock_data.csv")

REQUIRED_COLUMNS = ["age", "systolic_bp", "treatment_dose", "visit_date"]
NUMERIC_COLS = ["age", "systolic_bp", "treatment_dose"]


def _ensure_columns(df: pl.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {df.columns}")


def load_data(input_file: str) -> pl.DataFrame:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # infer schema; if your data has commas in text fields, you can add separator/quote settings
    df = pl.read_csv(input_file, infer_schema_length=1000)
    _ensure_columns(df, REQUIRED_COLUMNS)
    return df


def apply_rules(df: pl.DataFrame) -> pl.DataFrame:
    # Cast numeric fields safely; invalid parses become null
    df2 = df.with_columns([
        pl.col("age").cast(pl.Int64, strict=False).alias("age_num"),
        pl.col("systolic_bp").cast(pl.Float64, strict=False).alias("systolic_bp_num"),
        pl.col("treatment_dose").cast(pl.Float64, strict=False).alias("treatment_dose_num"),
    ])

    # Basic rule flags (1 = valid, 0 = invalid)
    df2 = df2.with_columns([
        pl.when(pl.col("age_num").is_between(18, 100)).then(1).otherwise(0).alias("age_valid"),
        pl.when(pl.col("systolic_bp_num").is_between(90, 180)).then(1).otherwise(0).alias("bp_valid"),
        pl.when(pl.col("treatment_dose_num").is_not_null() & (pl.col("treatment_dose_num") > 0)).then(1).otherwise(0).alias("dose_valid"),
        pl.when(pl.col("visit_date").is_not_null()).then(1).otherwise(0).alias("date_valid"),
    ])

    return df2


def detect_anomalies(df: pl.DataFrame) -> pl.DataFrame:
    # If too few rows, mark all normal
    if df.height < 10:
        return df.with_columns(pl.lit(0).alias("anomaly"))

    # Extract numeric matrix; impute missing with column median
    # (Polars median ignores nulls)
    df_num = df.select([pl.col(c) for c in ["age_num", "systolic_bp_num", "treatment_dose_num"]])

    medians = df_num.select([pl.col(c).median().alias(c) for c in df_num.columns]).row(0)
    median_map = dict(zip(df_num.columns, medians))

    df_num_imputed = df_num.with_columns([
        pl.col(c).fill_null(median_map[c]).alias(c) for c in df_num.columns
    ])

    X = df_num_imputed.to_numpy()

    clf = IsolationForest(contamination=0.1, random_state=42)
    preds = clf.fit_predict(X)  # -1 anomaly, 1 normal
    anomaly = (preds == -1).astype(np.int32)

    return df.with_columns(pl.Series("anomaly", anomaly))


def generate_report(df: pl.DataFrame) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(OUTPUT_DIR, f"validation_report_{timestamp}.csv")

    flagged = df.filter(
        (pl.col("age_valid") == 0)
        | (pl.col("bp_valid") == 0)
        | (pl.col("dose_valid") == 0)
        | (pl.col("date_valid") == 0)
        | (pl.col("anomaly") == 1)
    )

    print("\nEDC Compliance Validation Report")
    print(f"Generated: {timestamp}")
    print(f"Total records: {df.height}")
    print(f"Flagged records: {flagged.height}")

    if flagged.height > 0:
        print("\nFlagged Issues (preview):")
        # show first rows without flooding terminal
        print(flagged.head(25))

    flagged.write_csv(output_path)
    print(f"\nReport saved to: {output_path}")
    return output_path


def main(input_file: str = DEFAULT_INPUT_FILE) -> None:
    print("Starting EDC Data Validator (Polars + scikit-learn)...")
    df = load_data(input_file)
    df = apply_rules(df)
    df = detect_anomalies(df)
    generate_report(df)


if __name__ == "__main__":
    main()
