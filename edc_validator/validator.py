import polars as pl
from sklearn.ensemble import IsolationForest
import numpy as np
from datetime import datetime
import os

"""
edc_validator.py

MVP script for real-time clinical data validation and compliance checking.
Initiated January 2026 as part of ClinicalDataComplianceTool-Prototype.

Usage:
    python edc_validator.py
"""

# === CONFIG ===
INPUT_FILE = "mock_clinical_data.csv"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === VALIDATION RULES (easy to extend) ===
def apply_rules(df: pl.DataFrame) -> pl.DataFrame:
    """Apply business rules for clinical data compliance."""
    return df.with_columns([
        # Age: 18–100
        pl.col("age").is_between(18, 100).alias("age_valid"),
        # Systolic BP: 90–180, not null
        (pl.col("systolic_bp").is_between(90, 180) & pl.col("systolic_bp").is_not_null()).alias("bp_valid"),
        # Treatment dose: > 0, not null
        (pl.col("treatment_dose").gt(0) & pl.col("treatment_dose").is_not_null()).alias("dose_valid"),
        # Visit date: not null and reasonable (after 2000)
        pl.col("visit_date").str.strptime(pl.Date, "%Y-%m-%d").is_not_null().alias("date_valid")
    ])

# === ANOMALY DETECTION ===
def detect_anomalies(df: pl.DataFrame) -> pl.DataFrame:
    """Basic statistical anomaly detection on numeric columns."""
    numeric_cols = ["age", "systolic_bp", "treatment_dose"]
    # Fill nulls temporarily for model
    data = df.select(numeric_cols).fill_null(0).to_numpy()
    
    if len(data) < 3:  # Isolation Forest needs some data
        return df.with_columns(pl.lit(1).alias("anomaly"))  # 1 = normal

    model = IsolationForest(contamination=0.1, random_state=42)
    anomalies = model.fit_predict(data)
    return df.with_columns(pl.Series("anomaly", anomalies))

# === REPORTING ===
def generate_report(df: pl.DataFrame, flagged: pl.DataFrame):
    """Generate human-readable report and save to CSV."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"{OUTPUT_DIR}/validation_report_{timestamp}.csv"
    
    summary = f"""
    EDC Compliance Validation Report
    Generated: {timestamp}
    Total records: {len(df)}
    Flagged records: {len(flagged)}
    """
    print(summary)
    
    if len(flagged) > 0:
        print("\nFlagged Issues:")
        print(flagged)
    else:
        print("\nAll data passed validation.")
    
    # Save full flagged data
    flagged.write_csv(output_path)
    print(f"\nReport saved to: {output_path}")

# === MAIN EXECUTION ===
def main():
    print("Starting EDC Data Validator...")
    
    # 1. Load data
    df = pl.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} records from {INPUT_FILE}")
    
    # 2. Apply rules
    df = apply_rules(df)
    
    # 3. Detect anomalies
    df = detect_anomalies(df)
    
    # 4. Identify flagged records
    flagged = df.filter(
        (~pl.col("age_valid")) |
        (~pl.col("bp_valid")) |
        (~pl.col("dose_valid")) |
        (~pl.col("date_valid")) |
        (pl.col("anomaly") == -1)
    )
    
    # 5. Generate & save report
    generate_report(df, flagged)

if __name__ == "__main__":
    main()
