import duckdb
import pyod.models.iforest as iforest  # Isolation Forest from PyOD
import numpy as np
from datetime import datetime
import os

"""
edc_validator.py

MVP script using DuckDB for data handling/validation and PyOD for anomaly detection.
Validates mock clinical trial data against predefined rules and detects anomalies.
"""

# === CONFIG ===
INPUT_FILE = f"/mock_clinical_data.csv"
OUTPUT_DIR = f"output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load & Validate with DuckDB (SQL rules) ===
def load_and_validate():
    con = duckdb.connect()
    
    # Register CSV as table
    con.execute(f"CREATE TABLE clinical_data AS SELECT * FROM read_csv_auto('{INPUT_FILE}')")
    
    # Apply simple rules via SQL (easy to read/audit)
    con.execute("""
        SELECT *,
               CASE WHEN age BETWEEN 18 AND 100 THEN 1 ELSE 0 END AS age_valid,
               CASE WHEN systolic_bp BETWEEN 90 AND 180 AND systolic_bp IS NOT NULL THEN 1 ELSE 0 END AS bp_valid,
               CASE WHEN treatment_dose > 0 AND treatment_dose IS NOT NULL THEN 1 ELSE 0 END AS dose_valid,
               CASE WHEN visit_date IS NOT NULL THEN 1 ELSE 0 END AS date_valid
        FROM clinical_data
    """)
    
    result = con.fetchdf()  # Get as pandas-like DataFrame (DuckDB returns pandas df)
    return result

# === Anomaly Detection with PyOD ===
def detect_anomalies(df):
    numeric_cols = ['age', 'systolic_bp', 'treatment_dose']
    data = df[numeric_cols].fillna(0).to_numpy()  # Fill nulls
    
    if len(data) < 3:
        df['anomaly'] = 1  # Normal
        return df
    
    clf = iforest.IForest(contamination=0.1, random_state=42)
    clf.fit(data)
    df['anomaly'] = clf.predict(data)  # 0 = normal, 1 = anomaly (PyOD convention)
    return df

# === Reporting ===
def generate_report(df):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"{OUTPUT_DIR}/validation_report_{timestamp}.csv"
    
    flagged = df[(df['age_valid'] == 0) | (df['bp_valid'] == 0) | 
                 (df['dose_valid'] == 0) | (df['date_valid'] == 0) | 
                 (df['anomaly'] == 1)]
    
    summary = f"""
EDC Compliance Validation Report
Generated: {timestamp}
Total records: {len(df)}
Flagged records: {len(flagged)}
"""
    print(summary)
    
    if not flagged.empty:
        print("\nFlagged Issues:")
        print(flagged)
    else:
        print("\nAll data passed validation.")
    
    flagged.to_csv(output_path, index=False)
    print(f"\nReport saved to: {output_path}")

# === MAIN ===
def main():
    print("Starting EDC Data Validator (DuckDB + PyOD)...")
    
    df = load_and_validate()
    print(f"Loaded and validated {len(df)} records")
    
    df = detect_anomalies(df)
    
    generate_report(df)

if __name__ == "__main__":
    main()