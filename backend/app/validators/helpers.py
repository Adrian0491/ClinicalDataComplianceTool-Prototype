from __future__ import annotations

import pandas as pd

from app.validators.schemas import (
    FINDINGS_COLUMNS,
    FINDINGS_DTYPES,
    dataset_finding,
    empty_findings,
)


def ensure_row_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with a 'row_index' column equal to the original integer index."""
    df = df.copy()
    df["row_index"] = range(len(df))
    return df


def parse_iso_date(series: pd.Series) -> pd.Series:
    """
    Parse a string Series as ISO-8601 YYYY-MM-DD dates.
    Invalid / unparseable values become NaT.
    """
    return pd.to_datetime(series, format="%Y-%m-%d", errors="coerce").dt.date


def mk_findings(
    df: pd.DataFrame,
    mask: pd.Series,
    *,
    finding_type: str,
    rule_id: str,
    severity: str,
    domain: str,
    field: str,
    message: str,
    evidence_col: str | None = None,
    evidence_fn=None,
) -> pd.DataFrame:
    """
    Return one findings row per df row where mask is True.

    Parameters
    ----------
    df           Source DataFrame (row_index will be added if missing).
    mask         Boolean Series selecting violating rows.
    evidence_col Column name to use as evidence (falls back to field, then "").
    evidence_fn  Callable(sub_df) -> Series to compute evidence. Overrides evidence_col.
    """
    df_i = ensure_row_index(df)
    viol = df_i[mask].copy()

    if len(viol) == 0:
        return empty_findings()

    rows = []
    for _, row in viol.iterrows():
        if evidence_fn is not None:
            ev = str(evidence_fn(row))
        elif evidence_col and evidence_col in viol.columns:
            ev = "" if pd.isna(row[evidence_col]) else str(row[evidence_col])
        elif field in viol.columns:
            ev = "" if pd.isna(row[field]) else str(row[field])
        else:
            ev = ""

        usubjid = str(row["USUBJID"]) if "USUBJID" in viol.columns and not pd.isna(row.get("USUBJID")) else ""
        row_message = f"{message} (found: '{ev}')" if ev else message

        rows.append({
            "finding_type": finding_type,
            "rule_id":      rule_id,
            "severity":     severity,
            "domain":       domain,
            "field":        field,
            "message":      row_message,
            "row_index":    int(row["row_index"]),
            "usubjid":      usubjid,
            "evidence":     ev,
        })

    result = pd.DataFrame(rows, columns=FINDINGS_COLUMNS)
    return result.astype(FINDINGS_DTYPES)


def require_columns(
    df: pd.DataFrame,
    cols: list[str],
    domain: str,
    rule_prefix: str,
) -> list[pd.DataFrame]:
    """
    Return a CRIT finding list if any column in `cols` is missing from df.
    Returns empty list if all columns are present.
    """
    missing = [c for c in cols if c not in df.columns]
    if not missing:
        return []
    return [dataset_finding(
        rule_id=f"{rule_prefix}_000",
        severity="CRIT",
        domain=domain,
        field=",".join(missing),
        message=f"Missing required columns for {domain}: {missing}",
    )]
