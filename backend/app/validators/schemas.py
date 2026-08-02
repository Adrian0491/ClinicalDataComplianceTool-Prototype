from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Findings schema
# ---------------------------------------------------------------------------

FINDINGS_COLUMNS = [
    "finding_type",   # SDTM_RULE | CROSS_DOMAIN | DATASET_JSON | ANOMALY
    "rule_id",
    "severity",       # CRIT | HIGH | MED | LOW
    "domain",         # DM | VS | AE | CM | CROSS | GENERAL
    "field",
    "message",
    "row_index",      # -1 = dataset-level finding
    "usubjid",
    "evidence",
]

FINDINGS_DTYPES = {
    "finding_type": "string",
    "rule_id":      "string",
    "severity":     "string",
    "domain":       "string",
    "field":        "string",
    "message":      "string",
    "row_index":    "int64",
    "usubjid":      "string",
    "evidence":     "string",
}


def empty_findings() -> pd.DataFrame:
    """Return an empty DataFrame that conforms to the findings schema."""
    return pd.DataFrame(columns=FINDINGS_COLUMNS).astype(FINDINGS_DTYPES)


def _row_to_df(**kwargs) -> pd.DataFrame:
    """Build a single-row findings DataFrame from keyword args."""
    row = {col: kwargs.get(col, "" if col != "row_index" else -1)
           for col in FINDINGS_COLUMNS}
    return pd.DataFrame([row]).astype(FINDINGS_DTYPES)


def dataset_finding(
    *,
    rule_id: str,
    severity: str,
    domain: str,
    field: str,
    message: str,
    finding_type: str = "SDTM_RULE",
    evidence: str = "",
) -> pd.DataFrame:
    """Create a single dataset-level finding (row_index = -1)."""
    return _row_to_df(
        finding_type=finding_type,
        rule_id=rule_id,
        severity=severity,
        domain=domain,
        field=field,
        message=message,
        row_index=-1,
        usubjid="",
        evidence=evidence,
    )


def concat_findings(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate a list of findings DataFrames, ignoring empties."""
    non_empty = [f for f in parts if isinstance(f, pd.DataFrame) and len(f) > 0]
    if not non_empty:
        return empty_findings()
    result = pd.concat(non_empty, ignore_index=True)
    return result.astype({k: v for k, v in FINDINGS_DTYPES.items() if k in result.columns})


# ---------------------------------------------------------------------------
# SDTM controlled vocabulary
# ---------------------------------------------------------------------------

VS_ALLOWED_TESTCD: list[str] = [
    "SYSBP", "DIABP", "PULSE", "TEMP", "WEIGHT", "HEIGHT", "RESP",
]

VS_UNITS_BY_TESTCD: dict[str, list[str]] = {
    "SYSBP":  ["mmHg"],
    "DIABP":  ["mmHg"],
    "PULSE":  ["beats/min", "bpm"],
    "RESP":   ["breaths/min", "bpm"],
    "TEMP":   ["C", "F"],
    "WEIGHT": ["kg", "g", "lb"],
    "HEIGHT": ["cm", "m", "in"],
}

AE_ALLOWED_SER: list[str] = ["Y", "N"]
AE_ALLOWED_SEV: list[str] = ["MILD", "MODERATE", "SEVERE"]

DM_ALLOWED_SEX:  list[str] = ["M", "F", "U"]
DM_ALLOWED_AGEU: list[str] = ["YEARS", "MONTHS", "DAYS"]

EX_ALLOWED_ROUTE: list[str] = [
    "ORAL", "INTRAVENOUS", "SUBCUTANEOUS", "INTRAMUSCULAR",
    "TOPICAL", "INHALATION", "TRANSDERMAL",
]
EX_ALLOWED_DOSU: list[str] = ["mg", "g", "mcg", "mL", "IU"]

LB_ALLOWED_TESTCD: list[str] = [
    "ALT", "AST", "BILI", "CREAT", "BUN", "HGB", "HCT",
    "WBC", "PLAT", "NA", "K", "CL", "GLUC", "ALB",
]
LB_ALLOWED_NRIND: list[str] = ["NORMAL", "HIGH", "LOW", "ABNORMAL"]

DS_ALLOWED_DECOD: list[str] = [
    "COMPLETED", "DEATH", "ADVERSE EVENT", "LOST TO FOLLOW-UP",
    "PHYSICIAN DECISION", "PROTOCOL DEVIATION", "WITHDRAWAL BY SUBJECT",
    "STUDY TERMINATED BY SPONSOR", "LACK OF EFFICACY",
]
DS_ALLOWED_CAT: list[str] = ["DISPOSITION EVENT", "PROTOCOL MILESTONE"]

EG_ALLOWED_TESTCD: list[str] = ["QT", "QTCF", "QTCB", "HR", "PR", "QRS", "RR"]

EG_UNITS_BY_TESTCD: dict[str, list[str]] = {
    "QT":   ["msec", "ms"],
    "QTCF": ["msec", "ms"],
    "QTCB": ["msec", "ms"],
    "PR":   ["msec", "ms"],
    "QRS":  ["msec", "ms"],
    "RR":   ["msec", "ms"],
    "HR":   ["beats/min", "bpm"],
}

# QSSTAT is CDISC-controlled: the only permitted value is "NOT DONE" (used to
# flag an eCOA assessment that could not be completed); it is otherwise null.
QS_ALLOWED_STAT: list[str] = ["NOT DONE"]

# RECIST 1.1 response categories.
RS_ALLOWED_ORRES: list[str] = ["CR", "PR", "SD", "PD", "NE"]
RS_ALLOWED_EVAL: list[str] = [
    "INVESTIGATOR", "INDEPENDENT ASSESSOR", "INDEPENDENT REVIEW COMMITTEE",
]
# Same convention as QSSTAT: "NOT DONE" is the only permitted value.
RS_ALLOWED_STAT: list[str] = ["NOT DONE"]

PR_ALLOWED_CAT: list[str] = ["SURGICAL", "DIAGNOSTIC", "THERAPEUTIC", "OTHER"]
