from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest

from app.validators.schemas import (
    FINDINGS_COLUMNS,
    concat_findings,
    empty_findings,
)

# SDTM-aligned column names used by the generic validation pipeline.
# AGE      → Demographics (DM.AGE)
# SYSBP    → Vital Signs result where VSTESTCD = 'SYSBP'
# DOSE     → Treatment dose (study-specific, typically in EX domain)
# ALT      → Laboratory result where LBTESTCD = 'ALT' (liver safety signal)
# QTCF     → ECG result where EGTESTCD in ('QTCF', 'QT') (cardiac safety signal)
# QS_SCORE → mean Questionnaire numeric result (QSSTRESN), patient-reported burden
# PR_COUNT → count of Procedures rows per subject, procedure burden
#
# ALT, QTCF, QS_SCORE, and PR_COUNT are anomaly-detection-only features (see
# NUMERIC_COLS below) — none get a RULES threshold. AGE/SYSBP/DOSE are
# required per-subject attributes in the flat generic CSV, so a missing
# value there is itself a data-quality issue (apply_rules() flags NaN as
# invalid). ALT and QTCF are the opposite: sourced from LB/EG, a missing
# value just means that subject wasn't tested for that specific lab/ECG
# measure this visit — normal, not an error — so giving them a RULES entry
# would flag nearly every untested subject as "invalid" (verified: it did,
# on the mock cohort). QS_SCORE/PR_COUNT are excluded for a different
# reason — different eCOA instruments use incompatible scales and there is
# no universal "valid range" for a procedure count.
NUMERIC_COLS  = ["AGE", "SYSBP", "DOSE", "ALT", "QTCF", "QS_SCORE", "PR_COUNT"]
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

    Re-validated after extending NUMERIC_COLS to 7 features (adding ALT,
    QTCF, QS_SCORE, PR_COUNT): the extra low-signal dimensions dilute
    precision somewhat (~73% at 0.05 vs ~93% with 3 features) but 0.05 is
    still clearly the best balance — 0.10 drops to ~46% precision on the
    same cohort, 0.03 drops recall to ~56%.
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
    dm: pd.DataFrame,
    vs: pd.DataFrame,
    ex: pd.DataFrame,
    lb: pd.DataFrame | None = None,
    eg: pd.DataFrame | None = None,
    qs: pd.DataFrame | None = None,
    pr: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Assemble the per-subject anomaly-detection feature frame from the SDTM
    domain frames used by the validation pipeline (rather than the flat
    generic CSV upload). One row per DM subject.

    Core features (present since the DM/VS/EX pipeline was first wired up):
      AGE from DM, mean SYSBP from VS (VSTESTCD == 'SYSBP'), mean EXDOSE
      from EX, earliest VSDTC from VS.

    lb/eg/qs/pr are optional — a job may not have every domain uploaded —
    and contribute additional features when supplied:
      ALT      mean LBSTRESN (falling back to LBORRES) where LBTESTCD == 'ALT'
      QTCF     mean EGORRES where EGTESTCD in ('QTCF', 'QT')
      QS_SCORE mean QSSTRESN across all questionnaire rows
      PR_COUNT count of PR rows per subject (0, not missing, when PR is
               supplied but the subject has no procedures)
    """
    if dm.empty or "USUBJID" not in dm.columns:
        return pd.DataFrame(columns=["USUBJID", *REQUIRED_COLS])

    lb = lb if lb is not None else pd.DataFrame()
    eg = eg if eg is not None else pd.DataFrame()
    qs = qs if qs is not None else pd.DataFrame()
    pr = pr if pr is not None else pd.DataFrame()

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

    if not lb.empty and {"USUBJID", "LBTESTCD"}.issubset(lb.columns):
        alt = lb[lb["LBTESTCD"] == "ALT"].copy()
        src_col = "LBSTRESN" if "LBSTRESN" in alt.columns else "LBORRES"
        alt["_ALT_N"] = pd.to_numeric(alt[src_col], errors="coerce")
        alt_by_subj = alt.groupby("USUBJID")["_ALT_N"].mean().rename("ALT")
        out = out.merge(alt_by_subj, on="USUBJID", how="left")

    if not eg.empty and {"USUBJID", "EGTESTCD", "EGORRES"}.issubset(eg.columns):
        qtcf = eg[eg["EGTESTCD"].isin(["QTCF", "QT"])].copy()
        qtcf["_QTCF_N"] = pd.to_numeric(qtcf["EGORRES"], errors="coerce")
        qtcf_by_subj = qtcf.groupby("USUBJID")["_QTCF_N"].mean().rename("QTCF")
        out = out.merge(qtcf_by_subj, on="USUBJID", how="left")

    if not qs.empty and {"USUBJID", "QSSTRESN"}.issubset(qs.columns):
        qs_num = qs.copy()
        qs_num["_QSSTRESN_N"] = pd.to_numeric(qs_num["QSSTRESN"], errors="coerce")
        qs_by_subj = qs_num.groupby("USUBJID")["_QSSTRESN_N"].mean().rename("QS_SCORE")
        out = out.merge(qs_by_subj, on="USUBJID", how="left")

    if not pr.empty and "USUBJID" in pr.columns:
        pr_count = pr.groupby("USUBJID").size().rename("PR_COUNT")
        out = out.merge(pr_count, on="USUBJID", how="left")
        out["PR_COUNT"] = out["PR_COUNT"].fillna(0)

    for col in REQUIRED_COLS:
        if col not in out.columns:
            out[col] = pd.NA

    return out
