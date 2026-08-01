from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from bk.schemas import FINDINGS_COLUMNS, FINDINGS_DTYPES, concat_findings, empty_findings

# SDTM-aligned column names used by the generic validation pipeline.
# AGE  → Demographics (DM.AGE)
# SYSBP → Vital Signs result where VSTESTCD = 'SYSBP'
# DOSE  → Treatment dose (study-specific, typically in EX domain)
NUMERIC_COLS  = ["AGE", "SYSBP", "DOSE"]
REQUIRED_COLS = ["AGE", "SYSBP", "DOSE", "VSDTC"]

# (inclusive_min, inclusive_max) — use None for no bound.
# AGE bounds are intentionally consistent with SDTM_DM_004 in domain.py.
RULES: dict[str, tuple[float | None, float | None]] = {
    "AGE":   (18.0, 120.0),
    "SYSBP": (90.0, 180.0),
    "DOSE":  (0.001, None),   # must be > 0
}


def load_generic(path: str) -> pd.DataFrame:
    """Load the generic clinical CSV and validate required columns exist."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    return df


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add numeric cast columns and validity flag columns (1=valid, 0=invalid)
    for each rule in RULES, plus a date non-null check.
    """
    df = df.copy()

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[f"{col}_num"] = pd.to_numeric(df[col], errors="coerce")

    for col, (lo, hi) in RULES.items():
        num_col = f"{col}_num"
        if num_col not in df.columns:
            continue
        valid = pd.Series(True, index=df.index)
        if lo is not None:
            valid &= df[num_col] >= lo
        if hi is not None:
            valid &= df[num_col] <= hi
        df[f"{col}_valid"] = valid.where(df[num_col].notna(), other=False).astype(int)

    if "VSDTC" in df.columns:
        df["date_valid"] = df["VSDTC"].notna().astype(int)

    return df


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Run IsolationForest on the numeric feature columns.
    Adds an `anomaly` column: 1 = anomalous, 0 = normal.
    Falls back to all-zero if fewer than 10 rows.

    contamination default is 0.05, not sklearn's 0.1. Benchmarked against a
    synthetic 300-subject cohort (AGE/SYSBP/DOSE, N(45,15)/N(125,12)/{0,50,
    100,150}) with a ~5% implanted outlier rate (impossible ages, implausible
    BP, dosing errors): 0.1 flags 2x too many rows (~50% precision, alert
    fatigue), while 0.05 holds ~90% precision/recall and stays robust
    (recall >= 0.6) when the true rate drifts between 2-8%.
    """
    df = df.copy()
    num_cols = [f"{c}_num" for c in NUMERIC_COLS if f"{c}_num" in df.columns]

    if len(df) < 10 or not num_cols:
        df["anomaly"] = 0
        return df

    X_df = df[num_cols].copy()
    for col in num_cols:
        median = X_df[col].median()
        X_df[col] = X_df[col].fillna(median if pd.notna(median) else 0)

    X = X_df.to_numpy()
    clf = IsolationForest(contamination=contamination, random_state=42)
    preds = clf.fit_predict(X)            # -1 = anomaly, 1 = normal
    df["anomaly"] = (preds == -1).astype(int)
    return df


def _usubjid_of(row: pd.Series) -> str:
    val = row.get("USUBJID")
    return "" if val is None or pd.isna(val) else str(val)


def to_findings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert flagged rows from the generic validator into standard findings."""
    df_i = df.copy()
    df_i["row_index"] = range(len(df_i))
    parts: list[pd.DataFrame] = []

    # Rule-based flags
    flag_cols = [c for c in df_i.columns if c.endswith("_valid")]
    for col in flag_cols:
        field = col.replace("_valid", "")
        bad = df_i[df_i[col] == 0]
        if len(bad) == 0:
            continue
        rows = []
        for _, row in bad.iterrows():
            ev = str(row[field]) if field in bad.columns and pd.notna(row.get(field)) else ""
            rows.append({
                "finding_type": "SDTM_RULE",
                "rule_id":      f"GENERIC_{field.upper()}_001",
                "severity":     "MED",
                "domain":       "GENERAL",
                "field":        field,
                "message":      f"{field} failed validation rule.",
                "row_index":    int(row["row_index"]),
                "usubjid":      _usubjid_of(row),
                "evidence":     ev,
            })
        if rows:
            parts.append(pd.DataFrame(rows, columns=FINDINGS_COLUMNS))

    # Anomaly flags
    if "anomaly" in df_i.columns:
        bad_anom = df_i[df_i["anomaly"] == 1]
        if len(bad_anom):
            rows = []
            for _, row in bad_anom.iterrows():
                evidence = ", ".join(
                    f"{c}={row[c]}" for c in NUMERIC_COLS
                    if c in bad_anom.columns and pd.notna(row.get(c))
                )
                rows.append({
                    "finding_type": "ANOMALY",
                    "rule_id":      "ANOMALY_001",
                    "severity":     "LOW",
                    "domain":       "GENERAL",
                    "field":        "multivariate",
                    "message":      "Statistical outlier detected by IsolationForest.",
                    "row_index":    int(row["row_index"]),
                    "usubjid":      _usubjid_of(row),
                    "evidence":     evidence,
                })
            parts.append(pd.DataFrame(rows, columns=FINDINGS_COLUMNS))

    return concat_findings(parts) if parts else empty_findings()


def build_frame_from_domains(
    dm: pd.DataFrame, vs: pd.DataFrame, ex: pd.DataFrame
) -> pd.DataFrame:
    """
    Assemble the per-subject AGE/SYSBP/DOSE/VSDTC frame that apply_rules() and
    detect_anomalies() expect, from the SDTM DM/VS/EX domain frames used by
    the validation pipeline (rather than the flat generic CSV upload).

    One row per DM subject: AGE from DM, mean SYSBP from VS (VSTESTCD ==
    'SYSBP'), mean EXDOSE from EX, earliest VSDTC from VS.
    """
    if dm.empty or "USUBJID" not in dm.columns:
        return pd.DataFrame(columns=["USUBJID", *REQUIRED_COLS])

    out = dm[["USUBJID"]].copy()
    out["AGE"] = pd.to_numeric(dm.get("AGE"), errors="coerce").values

    if not vs.empty and {"USUBJID", "VSTESTCD", "VSORRES"}.issubset(vs.columns):
        sysbp = vs[vs["VSTESTCD"] == "SYSBP"].copy()
        sysbp["VSORRES"] = pd.to_numeric(sysbp["VSORRES"], errors="coerce")
        sysbp_by_subj = sysbp.groupby("USUBJID")["VSORRES"].mean().rename("SYSBP")
        out = out.merge(sysbp_by_subj, on="USUBJID", how="left")

        if "VSDTC" in vs.columns:
            date_by_subj = vs.groupby("USUBJID")["VSDTC"].min().rename("VSDTC")
            out = out.merge(date_by_subj, on="USUBJID", how="left")

    if not ex.empty and {"USUBJID", "EXDOSE"}.issubset(ex.columns):
        dose = ex.copy()
        dose["EXDOSE"] = pd.to_numeric(dose["EXDOSE"], errors="coerce")
        dose_by_subj = dose.groupby("USUBJID")["EXDOSE"].mean().rename("DOSE")
        out = out.merge(dose_by_subj, on="USUBJID", how="left")

    for col in REQUIRED_COLS:
        if col not in out.columns:
            out[col] = pd.NA

    return out
