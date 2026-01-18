"""
SDTM-like Domain Validation (MVP subset)

This module implements a practical, auditable subset of CDISC/SDTM-inspired checks for:
  - VS (Vital Signs)
  - AE (Adverse Events)
  - CM (Concomitant Medications)
and cross-domain consistency checks between:
  - VS <-> AE
  - VS <-> CM

Design goals:
  1) Be useful on real-world clinical exports (CSV) without requiring full SDTMIG/Define-XML parsing.
  2) Produce "Findings" (one row per issue) suitable for compliance reporting and audit trails.
  3) Keep logic deterministic and explainable (good for NIW/RFE evidence).

Notes:
  - These are "SDTM-like" rules: aligned in spirit with SDTM conventions (USUBJID, --DTC, controlled terms),
    but not a complete replacement for formal SDTMIG validation tools.
  - Date parsing uses ISO 8601 "YYYY-MM-DD" (subset) for MVP stability.
"""

from __future__ import annotations

import polars as pl


# -----------------------------
# Findings helpers
# -----------------------------

FINDINGS_SCHEMA = {
    "finding_type": pl.Utf8,  # SDTM_RULE or CROSS_DOMAIN
    "rule_id": pl.Utf8,
    "severity": pl.Utf8,      # LOW / MED / HIGH / CRIT
    "domain": pl.Utf8,        # VS / AE / CM / CROSS
    "field": pl.Utf8,
    "message": pl.Utf8,
    "row_index": pl.Int64,    # -1 means dataset-level / join-level finding
    "usubjid": pl.Utf8,       # best-effort
    "evidence": pl.Utf8,      # best-effort string
}


def _empty_findings() -> pl.DataFrame:
    return pl.DataFrame(schema=FINDINGS_SCHEMA)


def _ensure_row_index(df: pl.DataFrame) -> pl.DataFrame:
    return df if "row_index" in df.columns else df.with_row_index("row_index")


def _mk_findings(
    df: pl.DataFrame,
    mask: pl.Expr,
    *,
    finding_type: str,
    rule_id: str,
    severity: str,
    domain: str,
    field: str,
    message: str,
    evidence_expr: pl.Expr | None = None,
) -> pl.DataFrame:
    df_i = _ensure_row_index(df)

    viol = df_i.filter(mask)
    if viol.height == 0:
        return _empty_findings()

    if evidence_expr is None:
        evidence_expr = (
            pl.col(field).cast(pl.Utf8, strict=False)
            if field in viol.columns
            else pl.lit("")
        )

    usubjid_expr = (
        pl.col("USUBJID").cast(pl.Utf8, strict=False)
        if "USUBJID" in viol.columns
        else pl.lit("")
    )

    out = viol.select([
        pl.lit(finding_type).alias("finding_type"),
        pl.lit(rule_id).alias("rule_id"),
        pl.lit(severity).alias("severity"),
        pl.lit(domain).alias("domain"),
        pl.lit(field).alias("field"),
        pl.lit(message).alias("message"),
        pl.col("row_index").alias("row_index"),
        usubjid_expr.alias("usubjid"),
        evidence_expr.alias("evidence"),
    ])

    # Ensure schema consistency
    return out.select([pl.col(c).cast(t, strict=False).alias(c) for c, t in FINDINGS_SCHEMA.items()])


def _dataset_level_finding(*, rule_id: str, severity: str, domain: str, field: str, message: str) -> pl.DataFrame:
    return pl.DataFrame({
        "finding_type": ["SDTM_RULE" if domain in ("VS", "AE", "CM") else "CROSS_DOMAIN"],
        "rule_id": [rule_id],
        "severity": [severity],
        "domain": [domain],
        "field": [field],
        "message": [message],
        "row_index": [-1],
        "usubjid": [""],
        "evidence": [""],
    }).select([pl.col(c).cast(t, strict=False).alias(c) for c, t in FINDINGS_SCHEMA.items()])


def _require_columns(df: pl.DataFrame, cols: list[str], domain: str) -> list[pl.DataFrame]:
    missing = [c for c in cols if c not in df.columns]
    if not missing:
        return []
    return [_dataset_level_finding(
        rule_id=f"SDTM_{domain}_000",
        severity="CRIT",
        domain=domain,
        field=",".join(missing),
        message=f"Missing required columns for {domain}: {missing}"
    )]


def _parse_iso_date(expr: pl.Expr) -> pl.Expr:
    # Parse ISO date "YYYY-MM-DD" (subset for MVP). Invalid parses become null.
    return expr.cast(pl.Utf8, strict=False).str.strptime(pl.Date, "%Y-%m-%d", strict=False)


# -----------------------------
# VS validator
# -----------------------------

VS_ALLOWED_TESTCD = ["SYSBP", "DIABP", "HR", "TEMP", "WEIGHT", "HEIGHT", "RESP"]
VS_UNITS_BY_TESTCD = {
    "SYSBP": ["mmHg"],
    "DIABP": ["mmHg"],
    "HR": ["bpm"],
    "RESP": ["breaths/min", "bpm"],  # some exports misuse bpm; tolerate for MVP
    "TEMP": ["C", "F"],
    "WEIGHT": ["kg", "g", "lb"],
    "HEIGHT": ["cm", "m", "in"],
}


def validate_vs(vs: pl.DataFrame) -> pl.DataFrame:
    findings: list[pl.DataFrame] = []

    findings += _require_columns(vs, ["USUBJID", "VSTESTCD", "VSORRES", "VSDTC"], "VS")
    if findings and findings[0]["severity"][0] == "CRIT":
        return pl.concat(findings, how="vertical_relaxed")

    vs_i = _ensure_row_index(vs).with_columns([
        pl.col("USUBJID").cast(pl.Utf8, strict=False).alias("USUBJID_S"),
        pl.col("VSTESTCD").cast(pl.Utf8, strict=False).alias("VSTESTCD_S"),
        pl.col("VSORRES").cast(pl.Utf8, strict=False).alias("VSORRES_S"),
        _parse_iso_date(pl.col("VSDTC")).alias("VSDTC_D"),
    ])

    # VS_001: USUBJID required non-empty
    findings.append(_mk_findings(
        vs_i,
        pl.col("USUBJID_S").is_null() | (pl.col("USUBJID_S").str.strip_chars() == ""),
        finding_type="SDTM_RULE", rule_id="SDTM_VS_001", severity="HIGH", domain="VS",
        field="USUBJID", message="USUBJID is required and must be non-empty.",
        evidence_expr=pl.col("USUBJID_S"),
    ))

    # VS_002: VSTESTCD allowed set
    findings.append(_mk_findings(
        vs_i,
        pl.col("VSTESTCD_S").is_not_null() & ~pl.col("VSTESTCD_S").is_in(VS_ALLOWED_TESTCD),
        finding_type="SDTM_RULE", rule_id="SDTM_VS_002", severity="MED", domain="VS",
        field="VSTESTCD", message=f"VSTESTCD should be one of: {VS_ALLOWED_TESTCD}.",
        evidence_expr=pl.col("VSTESTCD_S"),
    ))

    # VS_003: VSDTC ISO date parseable (subset check)
    findings.append(_mk_findings(
        vs_i,
        pl.col("VSDTC").is_not_null() & pl.col("VSDTC_D").is_null(),
        finding_type="SDTM_RULE", rule_id="SDTM_VS_003", severity="LOW", domain="VS",
        field="VSDTC", message="VSDTC should be ISO date YYYY-MM-DD (subset check).",
        evidence_expr=pl.col("VSDTC").cast(pl.Utf8, strict=False),
    ))

    # VS_004: Numeric result when test implies numeric (here: all allowed VSTESTCD imply numeric)
    vsorres_num = pl.col("VSORRES_S").str.replace_all(",", ".").cast(pl.Float64, strict=False)
    findings.append(_mk_findings(
        vs_i,
        pl.col("VSTESTCD_S").is_in(VS_ALLOWED_TESTCD) & pl.col("VSORRES").is_not_null() & vsorres_num.is_null(),
        finding_type="SDTM_RULE", rule_id="SDTM_VS_004", severity="HIGH", domain="VS",
        field="VSORRES", message="VSORRES should be numeric for this VSTESTCD.",
        evidence_expr=pl.col("VSORRES_S"),
    ))

    # VS_005: Units consistency (if VSORRESU present)
    if "VSORRESU" in vs_i.columns:
        vs_u = vs_i.with_columns(pl.col("VSORRESU").cast(pl.Utf8, strict=False).alias("VSORRESU_S"))

        # Build a mask for invalid units by testcd
        invalid_units_mask = pl.lit(False)
        for testcd, allowed_units in VS_UNITS_BY_TESTCD.items():
            invalid_units_mask = invalid_units_mask | (
                (pl.col("VSTESTCD_S") == testcd)
                & pl.col("VSORRESU_S").is_not_null()
                & ~pl.col("VSORRESU_S").is_in(allowed_units)
            )

        findings.append(_mk_findings(
            vs_u,
            invalid_units_mask,
            finding_type="SDTM_RULE", rule_id="SDTM_VS_005", severity="MED", domain="VS",
            field="VSORRESU", message="VSORRESU unit is not consistent with VSTESTCD (MVP unit map).",
            evidence_expr=(pl.col("VSTESTCD_S") + pl.lit(" / ") + pl.col("VSORRESU_S")),
        ))
    else:
        findings.append(_dataset_level_finding(
            rule_id="SDTM_VS_005",
            severity="LOW",
            domain="VS",
            field="VSORRESU",
            message="Column VSORRESU not present; unit consistency checks skipped."
        ))

    return pl.concat([f for f in findings if f.height > 0], how="vertical_relaxed") if any(f.height > 0 for f in findings) else _empty_findings()


# -----------------------------
# AE validator
# -----------------------------

AE_ALLOWED_SER = ["Y", "N"]
AE_ALLOWED_SEV = ["MILD", "MODERATE", "SEVERE"]


def validate_ae(ae: pl.DataFrame) -> pl.DataFrame:
    findings: list[pl.DataFrame] = []

    findings += _require_columns(ae, ["USUBJID", "AETERM", "AESTDTC"], "AE")
    if findings and findings[0]["severity"][0] == "CRIT":
        return pl.concat(findings, how="vertical_relaxed")

    ae_i = _ensure_row_index(ae).with_columns([
        pl.col("USUBJID").cast(pl.Utf8, strict=False).alias("USUBJID_S"),
        pl.col("AETERM").cast(pl.Utf8, strict=False).alias("AETERM_S"),
        _parse_iso_date(pl.col("AESTDTC")).alias("AESTDTC_D"),
        _parse_iso_date(pl.col("AEENDTC")).alias("AEENDTC_D") if "AEENDTC" in ae.columns else pl.lit(None).alias("AEENDTC_D"),
    ])

    # AE_001: AETERM required
    findings.append(_mk_findings(
        ae_i,
        pl.col("AETERM_S").is_null() | (pl.col("AETERM_S").str.strip_chars() == ""),
        finding_type="SDTM_RULE", rule_id="SDTM_AE_001", severity="HIGH", domain="AE",
        field="AETERM", message="AETERM is required and must be non-empty.",
        evidence_expr=pl.col("AETERM_S"),
    ))

    # AE_002: AESTDTC parseable
    findings.append(_mk_findings(
        ae_i,
        pl.col("AESTDTC").is_not_null() & pl.col("AESTDTC_D").is_null(),
        finding_type="SDTM_RULE", rule_id="SDTM_AE_002", severity="MED", domain="AE",
        field="AESTDTC", message="AESTDTC should be ISO date YYYY-MM-DD (subset check).",
        evidence_expr=pl.col("AESTDTC").cast(pl.Utf8, strict=False),
    ))

    # AE_003: AEENDTC parseable (if present)
    if "AEENDTC" in ae_i.columns:
        findings.append(_mk_findings(
            ae_i,
            pl.col("AEENDTC").is_not_null() & pl.col("AEENDTC_D").is_null(),
            finding_type="SDTM_RULE", rule_id="SDTM_AE_003", severity="LOW", domain="AE",
            field="AEENDTC", message="AEENDTC should be ISO date YYYY-MM-DD (subset check).",
            evidence_expr=pl.col("AEENDTC").cast(pl.Utf8, strict=False),
        ))

    # AE_004: AESTDTC <= AEENDTC when both present/parseable
    if "AEENDTC" in ae.columns:
        findings.append(_mk_findings(
            ae_i,
            pl.col("AESTDTC_D").is_not_null() & pl.col("AEENDTC_D").is_not_null() & (pl.col("AESTDTC_D") > pl.col("AEENDTC_D")),
            finding_type="SDTM_RULE", rule_id="SDTM_AE_004", severity="HIGH", domain="AE",
            field="AESTDTC/AEENDTC", message="AESTDTC must be on/before AEENDTC.",
            evidence_expr=pl.col("AESTDTC").cast(pl.Utf8, strict=False) + pl.lit(" > ") + pl.col("AEENDTC").cast(pl.Utf8, strict=False),
        ))

    # AE_005: AESER controlled terminology (if present)
    if "AESER" in ae.columns:
        findings.append(_mk_findings(
            ae_i,
            pl.col("AESER").is_not_null() & ~pl.col("AESER").cast(pl.Utf8, strict=False).is_in(AE_ALLOWED_SER),
            finding_type="SDTM_RULE", rule_id="SDTM_AE_005", severity="MED", domain="AE",
            field="AESER", message=f"AESER should be one of: {AE_ALLOWED_SER}.",
            evidence_expr=pl.col("AESER").cast(pl.Utf8, strict=False),
        ))

    # AE_006: AESEV controlled terminology (if present)
    if "AESEV" in ae.columns:
        findings.append(_mk_findings(
            ae_i,
            pl.col("AESEV").is_not_null() & ~pl.col("AESEV").cast(pl.Utf8, strict=False).is_in(AE_ALLOWED_SEV),
            finding_type="SDTM_RULE", rule_id="SDTM_AE_006", severity="LOW", domain="AE",
            field="AESEV", message=f"AESEV should be one of: {AE_ALLOWED_SEV}.",
            evidence_expr=pl.col("AESEV").cast(pl.Utf8, strict=False),
        ))

    return pl.concat([f for f in findings if f.height > 0], how="vertical_relaxed") if any(f.height > 0 for f in findings) else _empty_findings()


# -----------------------------
# CM validator
# -----------------------------

def validate_cm(cm: pl.DataFrame) -> pl.DataFrame:
    findings: list[pl.DataFrame] = []

    findings += _require_columns(cm, ["USUBJID", "CMTRT", "CMSTDTC"], "CM")
    if findings and findings[0]["severity"][0] == "CRIT":
        return pl.concat(findings, how="vertical_relaxed")

    cm_i = _ensure_row_index(cm).with_columns([
        pl.col("USUBJID").cast(pl.Utf8, strict=False).alias("USUBJID_S"),
        pl.col("CMTRT").cast(pl.Utf8, strict=False).alias("CMTRT_S"),
        _parse_iso_date(pl.col("CMSTDTC")).alias("CMSTDTC_D"),
        _parse_iso_date(pl.col("CMENDTC")).alias("CMENDTC_D") if "CMENDTC" in cm.columns else pl.lit(None).alias("CMENDTC_D"),
    ])

    # CM_001: CMTRT required
    findings.append(_mk_findings(
        cm_i,
        pl.col("CMTRT_S").is_null() | (pl.col("CMTRT_S").str.strip_chars() == ""),
        finding_type="SDTM_RULE", rule_id="SDTM_CM_001", severity="HIGH", domain="CM",
        field="CMTRT", message="CMTRT is required and must be non-empty.",
        evidence_expr=pl.col("CMTRT_S"),
    ))

    # CM_002: CMSTDTC parseable
    findings.append(_mk_findings(
        cm_i,
        pl.col("CMSTDTC").is_not_null() & pl.col("CMSTDTC_D").is_null(),
        finding_type="SDTM_RULE", rule_id="SDTM_CM_002", severity="MED", domain="CM",
        field="CMSTDTC", message="CMSTDTC should be ISO date YYYY-MM-DD (subset check).",
        evidence_expr=pl.col("CMSTDTC").cast(pl.Utf8, strict=False),
    ))

    # CM_003: CMENDTC parseable (if present)
    if "CMENDTC" in cm.columns:
        findings.append(_mk_findings(
            cm_i,
            pl.col("CMENDTC").is_not_null() & pl.col("CMENDTC_D").is_null(),
            finding_type="SDTM_RULE", rule_id="SDTM_CM_003", severity="LOW", domain="CM",
            field="CMENDTC", message="CMENDTC should be ISO date YYYY-MM-DD (subset check).",
            evidence_expr=pl.col("CMENDTC").cast(pl.Utf8, strict=False),
        ))

        # CM_004: CMSTDTC <= CMENDTC when both present/parseable
        findings.append(_mk_findings(
            cm_i,
            pl.col("CMSTDTC_D").is_not_null() & pl.col("CMENDTC_D").is_not_null() & (pl.col("CMSTDTC_D") > pl.col("CMENDTC_D")),
            finding_type="SDTM_RULE", rule_id="SDTM_CM_004", severity="HIGH", domain="CM",
            field="CMSTDTC/CMENDTC", message="CMSTDTC must be on/before CMENDTC.",
            evidence_expr=pl.col("CMSTDTC").cast(pl.Utf8, strict=False) + pl.lit(" > ") + pl.col("CMENDTC").cast(pl.Utf8, strict=False),
        ))

    return pl.concat([f for f in findings if f.height > 0], how="vertical_relaxed") if any(f.height > 0 for f in findings) else _empty_findings()

# -----------------------------
# DM validator (Demographics anchor)
# -----------------------------

DM_ALLOWED_SEX = ["M", "F", "U"]
DM_ALLOWED_AGEU = ["YEARS", "MONTHS", "DAYS"]


def validate_dm(dm: pl.DataFrame) -> pl.DataFrame:
    findings: list[pl.DataFrame] = []

    findings += _require_columns(dm, ["USUBJID", "STUDYID"], "DM")
    if findings and findings[0]["severity"][0] == "CRIT":
        return pl.concat(findings, how="vertical_relaxed")

    dm_i = _ensure_row_index(dm).with_columns([
        pl.col("USUBJID").cast(pl.Utf8, strict=False).alias("USUBJID_S"),
        pl.col("STUDYID").cast(pl.Utf8, strict=False).alias("STUDYID_S"),
        pl.col("SEX").cast(pl.Utf8, strict=False).alias("SEX_S") if "SEX" in dm.columns else pl.lit(None).alias("SEX_S"),
        pl.col("AGE").cast(pl.Float64, strict=False).alias("AGE_N") if "AGE" in dm.columns else pl.lit(None).alias("AGE_N"),
        pl.col("AGEU").cast(pl.Utf8, strict=False).alias("AGEU_S") if "AGEU" in dm.columns else pl.lit(None).alias("AGEU_S"),
        _parse_iso_date(pl.col("RFSTDTC")).alias("RFSTDTC_D") if "RFSTDTC" in dm.columns else pl.lit(None).alias("RFSTDTC_D"),
        _parse_iso_date(pl.col("RFENDTC")).alias("RFENDTC_D") if "RFENDTC" in dm.columns else pl.lit(None).alias("RFENDTC_D"),
    ])

    # DM_001: USUBJID required non-empty
    findings.append(_mk_findings(
        dm_i,
        pl.col("USUBJID_S").is_null() | (pl.col("USUBJID_S").str.strip_chars() == ""),
        finding_type="SDTM_RULE", rule_id="SDTM_DM_001", severity="HIGH", domain="DM",
        field="USUBJID", message="USUBJID is required and must be non-empty.",
        evidence_expr=pl.col("USUBJID_S"),
    ))

    # DM_002: USUBJID must be unique in DM
    dupes = (
        dm_i.filter(pl.col("USUBJID_S").is_not_null() & (pl.col("USUBJID_S").str.strip_chars() != ""))
            .with_columns(pl.col("USUBJID_S").is_duplicated().alias("IS_DUP"))
            .filter(pl.col("IS_DUP"))
    )
    if dupes.height > 0:
        findings.append(dupes.select([
            pl.lit("SDTM_RULE").alias("finding_type"),
            pl.lit("SDTM_DM_002").alias("rule_id"),
            pl.lit("HIGH").alias("severity"),
            pl.lit("DM").alias("domain"),
            pl.lit("USUBJID").alias("field"),
            pl.lit("USUBJID must be unique in DM.").alias("message"),
            pl.col("row_index").alias("row_index"),
            pl.col("USUBJID_S").alias("usubjid"),
            pl.col("USUBJID_S").alias("evidence"),
        ]).select([pl.col(c).cast(t, strict=False).alias(c) for c, t in FINDINGS_SCHEMA.items()]))

    # DM_003: SEX controlled terminology (if present)
    if "SEX" in dm.columns:
        findings.append(_mk_findings(
            dm_i,
            pl.col("SEX_S").is_not_null() & ~pl.col("SEX_S").is_in(DM_ALLOWED_SEX),
            finding_type="SDTM_RULE", rule_id="SDTM_DM_003", severity="MED", domain="DM",
            field="SEX", message=f"SEX should be one of: {DM_ALLOWED_SEX}.",
            evidence_expr=pl.col("SEX_S"),
        ))

    # DM_004: AGE reasonable bounds (if present)
    if "AGE" in dm.columns:
        findings.append(_mk_findings(
            dm_i,
            pl.col("AGE_N").is_not_null() & ((pl.col("AGE_N") < 0) | (pl.col("AGE_N") > 120)),
            finding_type="SDTM_RULE", rule_id="SDTM_DM_004", severity="MED", domain="DM",
            field="AGE", message="AGE should be between 0 and 120 (MVP heuristic).",
            evidence_expr=pl.col("AGE").cast(pl.Utf8, strict=False),
        ))

    # DM_005: AGEU valid when AGE present (if present)
    if "AGE" in dm.columns and "AGEU" in dm.columns:
        findings.append(_mk_findings(
            dm_i,
            pl.col("AGE_N").is_not_null() & (pl.col("AGEU_S").is_null() | ~pl.col("AGEU_S").is_in(DM_ALLOWED_AGEU)),
            finding_type="SDTM_RULE", rule_id="SDTM_DM_005", severity="LOW", domain="DM",
            field="AGEU", message=f"AGEU should be one of: {DM_ALLOWED_AGEU} when AGE is present.",
            evidence_expr=pl.col("AGEU_S"),
        ))

    # DM_006/007: RFSTDTC/RFENDTC parseable and ordered (if present)
    if "RFSTDTC" in dm.columns:
        findings.append(_mk_findings(
            dm_i,
            pl.col("RFSTDTC").is_not_null() & pl.col("RFSTDTC_D").is_null(),
            finding_type="SDTM_RULE", rule_id="SDTM_DM_006", severity="LOW", domain="DM",
            field="RFSTDTC", message="RFSTDTC should be ISO date YYYY-MM-DD (subset check).",
            evidence_expr=pl.col("RFSTDTC").cast(pl.Utf8, strict=False),
        ))
    if "RFENDTC" in dm.columns:
        findings.append(_mk_findings(
            dm_i,
            pl.col("RFENDTC").is_not_null() & pl.col("RFENDTC_D").is_null(),
            finding_type="SDTM_RULE", rule_id="SDTM_DM_007", severity="LOW", domain="DM",
            field="RFENDTC", message="RFENDTC should be ISO date YYYY-MM-DD (subset check).",
            evidence_expr=pl.col("RFENDTC").cast(pl.Utf8, strict=False),
        ))
    if "RFSTDTC" in dm.columns and "RFENDTC" in dm.columns:
        findings.append(_mk_findings(
            dm_i,
            pl.col("RFSTDTC_D").is_not_null() & pl.col("RFENDTC_D").is_not_null() & (pl.col("RFSTDTC_D") > pl.col("RFENDTC_D")),
            finding_type="SDTM_RULE", rule_id="SDTM_DM_008", severity="HIGH", domain="DM",
            field="RFSTDTC/RFENDTC", message="RFSTDTC must be on/before RFENDTC.",
            evidence_expr=pl.col("RFSTDTC").cast(pl.Utf8, strict=False) + pl.lit(" > ") + pl.col("RFENDTC").cast(pl.Utf8, strict=False),
        ))

    return pl.concat([f for f in findings if f.height > 0], how="vertical_relaxed") if any(f.height > 0 for f in findings) else _empty_findings()


# -----------------------------
# Cross-domain rules: VS/AE and VS/CM
# -----------------------------

def validate_vs_ae(vs: pl.DataFrame, ae: pl.DataFrame) -> pl.DataFrame:
    """
    Cross-domain rules between VS and AE.

    Implemented (MVP):
      - X_VSAE_001 (HIGH): AE.USUBJID must exist in VS.USUBJID (simple orphan check)
      - X_VSAE_002 (MED): If VS has dates for a subject, AE start date should not be outside VS date range by more than a tolerance window (optional)
    """
    findings: list[pl.DataFrame] = []

    # Require columns
    if any([
        c not in vs.columns for c in ["USUBJID", "VSDTC"]
    ]) or any([
        c not in ae.columns for c in ["USUBJID", "AESTDTC"]
    ]):
        return _dataset_level_finding(
            rule_id="X_VSAE_000", severity="CRIT", domain="CROSS", field="USUBJID/--DTC",
            message="Missing required columns for VS/AE cross checks (need VS.USUBJID,VSDTC and AE.USUBJID,AESTDTC)."
        )

    vs_i = _ensure_row_index(vs).with_columns([
        pl.col("USUBJID").cast(pl.Utf8, strict=False).alias("USUBJID_S"),
        _parse_iso_date(pl.col("VSDTC")).alias("VSDTC_D"),
    ])

    ae_i = _ensure_row_index(ae).with_columns([
        pl.col("USUBJID").cast(pl.Utf8, strict=False).alias("USUBJID_S"),
        _parse_iso_date(pl.col("AESTDTC")).alias("AESTDTC_D"),
    ])

    # X_VSAE_001: Orphan AE subjects not present in VS
    vs_subjects = vs_i.select(pl.col("USUBJID_S")).unique()
    ae_orphans = (
        ae_i.join(vs_subjects, on="USUBJID_S", how="anti")
            .select([
                pl.lit("CROSS_DOMAIN").alias("finding_type"),
                pl.lit("X_VSAE_001").alias("rule_id"),
                pl.lit("HIGH").alias("severity"),
                pl.lit("CROSS").alias("domain"),
                pl.lit("USUBJID").alias("field"),
                pl.lit("AE subject not found in VS (orphan USUBJID).").alias("message"),
                pl.col("row_index").alias("row_index"),
                pl.col("USUBJID_S").alias("usubjid"),
                pl.col("USUBJID_S").alias("evidence"),
            ])
    )
    if ae_orphans.height > 0:
        findings.append(ae_orphans.select([pl.col(c).cast(t, strict=False).alias(c) for c, t in FINDINGS_SCHEMA.items()]))

    # X_VSAE_002: AE start date outside VS date range (if dates parseable)
    # Build VS min/max per subject
    vs_range = (
        vs_i.filter(pl.col("VSDTC_D").is_not_null())
            .group_by("USUBJID_S")
            .agg([
                pl.col("VSDTC_D").min().alias("VS_MIN"),
                pl.col("VSDTC_D").max().alias("VS_MAX"),
            ])
    )

    ae_with_range = ae_i.join(vs_range, on="USUBJID_S", how="inner")
    # If AE date is parseable and outside [VS_MIN, VS_MAX], flag
    out_of_range = ae_with_range.filter(
        pl.col("AESTDTC_D").is_not_null()
        & (
            (pl.col("AESTDTC_D") < pl.col("VS_MIN"))
            | (pl.col("AESTDTC_D") > pl.col("VS_MAX"))
        )
    )

    if out_of_range.height > 0:
        findings.append(out_of_range.select([
            pl.lit("CROSS_DOMAIN").alias("finding_type"),
            pl.lit("X_VSAE_002").alias("rule_id"),
            pl.lit("MED").alias("severity"),
            pl.lit("CROSS").alias("domain"),
            pl.lit("AESTDTC").alias("field"),
            pl.lit("AE start date is outside the subject's VS date range (MVP heuristic).").alias("message"),
            pl.col("row_index").alias("row_index"),
            pl.col("USUBJID_S").alias("usubjid"),
            (pl.col("AESTDTC").cast(pl.Utf8, strict=False)
             + pl.lit(" vs [")
             + pl.col("VS_MIN").cast(pl.Utf8, strict=False)
             + pl.lit(", ")
             + pl.col("VS_MAX").cast(pl.Utf8, strict=False)
             + pl.lit("]")).alias("evidence"),
        ]).select([pl.col(c).cast(t, strict=False).alias(c) for c, t in FINDINGS_SCHEMA.items()]))

    return pl.concat(findings, how="vertical_relaxed") if findings else _empty_findings()


def validate_vs_cm(vs: pl.DataFrame, cm: pl.DataFrame) -> pl.DataFrame:
    """
    Cross-domain rules between VS and CM.

    Implemented (MVP):
      - X_VSCM_001 (HIGH): CM.USUBJID must exist in VS.USUBJID (orphan check)
      - X_VSCM_002 (LOW/MED): VS dates outside medication window (if CM start/end dates exist)
    """
    findings: list[pl.DataFrame] = []

    if any([
        c not in vs.columns for c in ["USUBJID", "VSDTC"]
    ]) or any([
        c not in cm.columns for c in ["USUBJID", "CMSTDTC"]
    ]):
        return _dataset_level_finding(
            rule_id="X_VSCM_000", severity="CRIT", domain="CROSS", field="USUBJID/--DTC",
            message="Missing required columns for VS/CM cross checks (need VS.USUBJID,VSDTC and CM.USUBJID,CMSTDTC)."
        )

    vs_i = _ensure_row_index(vs).with_columns([
        pl.col("USUBJID").cast(pl.Utf8, strict=False).alias("USUBJID_S"),
        _parse_iso_date(pl.col("VSDTC")).alias("VSDTC_D"),
    ])

    cm_i = _ensure_row_index(cm).with_columns([
        pl.col("USUBJID").cast(pl.Utf8, strict=False).alias("USUBJID_S"),
        _parse_iso_date(pl.col("CMSTDTC")).alias("CMSTDTC_D"),
        _parse_iso_date(pl.col("CMENDTC")).alias("CMENDTC_D") if "CMENDTC" in cm.columns else pl.lit(None).alias("CMENDTC_D"),
    ])

    # X_VSCM_001: Orphan CM subjects not present in VS
    vs_subjects = vs_i.select(pl.col("USUBJID_S")).unique()
    cm_orphans = (
        cm_i.join(vs_subjects, on="USUBJID_S", how="anti")
            .select([
                pl.lit("CROSS_DOMAIN").alias("finding_type"),
                pl.lit("X_VSCM_001").alias("rule_id"),
                pl.lit("HIGH").alias("severity"),
                pl.lit("CROSS").alias("domain"),
                pl.lit("USUBJID").alias("field"),
                pl.lit("CM subject not found in VS (orphan USUBJID).").alias("message"),
                pl.col("row_index").alias("row_index"),
                pl.col("USUBJID_S").alias("usubjid"),
                pl.col("USUBJID_S").alias("evidence"),
            ])
    )
    if cm_orphans.height > 0:
        findings.append(cm_orphans.select([pl.col(c).cast(t, strict=False).alias(c) for c, t in FINDINGS_SCHEMA.items()]))

    # X_VSCM_002: VS dates outside CM window (if CMENDTC exists & parseable)
    if "CMENDTC" in cm.columns:
        cm_windows = (
            cm_i.filter(pl.col("CMSTDTC_D").is_not_null())
                .group_by("USUBJID_S")
                .agg([
                    pl.col("CMSTDTC_D").min().alias("CM_MIN"),
                    pl.col("CMENDTC_D").max().alias("CM_MAX"),
                ])
        )

        vs_with_cm = vs_i.join(cm_windows, on="USUBJID_S", how="inner")
        out = vs_with_cm.filter(
            pl.col("VSDTC_D").is_not_null()
            & pl.col("CM_MAX").is_not_null()
            & (
                (pl.col("VSDTC_D") < pl.col("CM_MIN"))
                | (pl.col("VSDTC_D") > pl.col("CM_MAX"))
            )
        )

        if out.height > 0:
            findings.append(out.select([
                pl.lit("CROSS_DOMAIN").alias("finding_type"),
                pl.lit("X_VSCM_002").alias("rule_id"),
                pl.lit("LOW").alias("severity"),
                pl.lit("CROSS").alias("domain"),
                pl.lit("VSDTC").alias("field"),
                pl.lit("VS date is outside the subject's CM medication window (MVP heuristic).").alias("message"),
                pl.col("row_index").alias("row_index"),
                pl.col("USUBJID_S").alias("usubjid"),
                (pl.col("VSDTC").cast(pl.Utf8, strict=False)
                 + pl.lit(" vs [")
                 + pl.col("CM_MIN").cast(pl.Utf8, strict=False)
                 + pl.lit(", ")
                 + pl.col("CM_MAX").cast(pl.Utf8, strict=False)
                 + pl.lit("]")).alias("evidence"),
            ]).select([pl.col(c).cast(t, strict=False).alias(c) for c, t in FINDINGS_SCHEMA.items()]))

    return pl.concat(findings, how="vertical_relaxed") if findings else _empty_findings()

def validate_dm_link(dm: pl.DataFrame, other: pl.DataFrame, other_domain: str) -> pl.DataFrame:
    """
    X_DMLINK_001 (HIGH): other.USUBJID must exist in DM.USUBJID
    """
    if "USUBJID" not in dm.columns or "USUBJID" not in other.columns:
        return _dataset_level_finding(
            rule_id=f"X_DMLINK_{other_domain}_000",
            severity="CRIT",
            domain="CROSS",
            field="USUBJID",
            message=f"Missing USUBJID for DM/{other_domain} link check."
        )

    dm_u = _ensure_row_index(dm).with_columns(pl.col("USUBJID").cast(pl.Utf8, strict=False).alias("USUBJID_S")).select("USUBJID_S").unique()
    ot = _ensure_row_index(other).with_columns(pl.col("USUBJID").cast(pl.Utf8, strict=False).alias("USUBJID_S"))

    orphans = (
        ot.join(dm_u, on="USUBJID_S", how="anti")
          .select([
              pl.lit("CROSS_DOMAIN").alias("finding_type"),
              pl.lit(f"X_DMLINK_{other_domain}_001").alias("rule_id"),
              pl.lit("HIGH").alias("severity"),
              pl.lit("CROSS").alias("domain"),
              pl.lit("USUBJID").alias("field"),
              pl.lit(f"{other_domain} subject not found in DM (orphan USUBJID).").alias("message"),
              pl.col("row_index").alias("row_index"),
              pl.col("USUBJID_S").alias("usubjid"),
              pl.col("USUBJID_S").alias("evidence"),
          ])
    )

    return orphans.select([pl.col(c).cast(t, strict=False).alias(c) for c, t in FINDINGS_SCHEMA.items()]) if orphans.height > 0 else _empty_findings()
