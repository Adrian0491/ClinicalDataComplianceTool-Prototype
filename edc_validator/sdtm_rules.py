import re
import polars as pl

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")  # simple ISO date (YYYY-MM-DD)

def _mk_findings(df: pl.DataFrame, mask: pl.Expr, rule_id: str, severity: str, field: str, message: str) -> pl.DataFrame:
    """
    Return a findings DataFrame for rows where mask == True (i.e., rule violated).
    """
    # add a row index so we can point to specific records
    df_i = df.with_row_index("row_index")
    viol = df_i.filter(mask)
    if viol.height == 0:
        return pl.DataFrame(schema={
            "finding_type": pl.Utf8,
            "rule_id": pl.Utf8,
            "severity": pl.Utf8,
            "field": pl.Utf8,
            "message": pl.Utf8,
            "row_index": pl.Int64,
            "evidence": pl.Utf8,
        })

    # evidence is best-effort string
    evidence_expr = pl.col(field).cast(pl.Utf8, strict=False) if field in df.columns else pl.lit("")

    return viol.select([
        pl.lit("SDTM_RULE").alias("finding_type"),
        pl.lit(rule_id).alias("rule_id"),
        pl.lit(severity).alias("severity"),
        pl.lit(field).alias("field"),
        pl.lit(message).alias("message"),
        pl.col("row_index").alias("row_index"),
        evidence_expr.alias("evidence"),
    ])


def validate_dm(df: pl.DataFrame) -> pl.DataFrame:
    """
    SDTM DM-like checks (subset).
    Returns findings table (one row per issue).
    """
    findings = []

    # DM_001: USUBJID required
    if "USUBJID" in df.columns:
        findings.append(_mk_findings(
            df,
            pl.col("USUBJID").is_null() | (pl.col("USUBJID").cast(pl.Utf8, strict=False).str.strip_chars() == ""),
            "SDTM_DM_001",
            "HIGH",
            "USUBJID",
            "USUBJID is required and must be non-empty."
        ))
        # DM_002: USUBJID unique
        dupes = (
            df.with_row_index("row_index")
              .filter(pl.col("USUBJID").is_not_null())
              .with_columns(pl.col("USUBJID").cast(pl.Utf8, strict=False).alias("USUBJID_S"))
              .filter(pl.col("USUBJID_S").is_duplicated())
        )
        if dupes.height > 0:
            findings.append(dupes.select([
                pl.lit("SDTM_RULE").alias("finding_type"),
                pl.lit("SDTM_DM_002").alias("rule_id"),
                pl.lit("HIGH").alias("severity"),
                pl.lit("USUBJID").alias("field"),
                pl.lit("USUBJID must be unique in DM.").alias("message"),
                pl.col("row_index").alias("row_index"),
                pl.col("USUBJID_S").alias("evidence"),
            ]))
    else:
        # If column missing entirely, return a dataset-level finding
        findings.append(pl.DataFrame({
            "finding_type": ["SDTM_RULE"],
            "rule_id": ["SDTM_DM_001"],
            "severity": ["HIGH"],
            "field": ["USUBJID"],
            "message": ["Missing required column USUBJID."],
            "row_index": [-1],
            "evidence": [""],
        }))

    # DM_003: STUDYID required
    if "STUDYID" in df.columns:
        findings.append(_mk_findings(
            df,
            pl.col("STUDYID").is_null() | (pl.col("STUDYID").cast(pl.Utf8, strict=False).str.strip_chars() == ""),
            "SDTM_DM_003",
            "HIGH",
            "STUDYID",
            "STUDYID is required and must be non-empty."
        ))

    # DM_004: SEX controlled terminology (simple set)
    if "SEX" in df.columns:
        findings.append(_mk_findings(
            df,
            pl.col("SEX").is_not_null() & ~pl.col("SEX").cast(pl.Utf8, strict=False).is_in(["M", "F", "U"]),
            "SDTM_DM_004",
            "MED",
            "SEX",
            "SEX must be one of: M, F, U."
        ))

    # DM_005: AGE non-negative (and reasonable upper bound)
    if "AGE" in df.columns:
        age_num = pl.col("AGE").cast(pl.Float64, strict=False)
        findings.append(_mk_findings(
            df,
            age_num.is_not_null() & ((age_num < 0) | (age_num > 120)),
            "SDTM_DM_005",
            "MED",
            "AGE",
            "AGE should be between 0 and 120 for typical adult/human studies."
        ))

    # DM_006: RFSTDTC ISO date format (YYYY-MM-DD) if present
    if "RFSTDTC" in df.columns:
        findings.append(_mk_findings(
            df,
            pl.col("RFSTDTC").is_not_null() & ~pl.col("RFSTDTC").cast(pl.Utf8, strict=False).str.contains(r"^\d{4}-\d{2}-\d{2}$"),
            "SDTM_DM_006",
            "LOW",
            "RFSTDTC",
            "RFSTDTC should be ISO 8601 date format YYYY-MM-DD (subset check)."
        ))

    if not findings:
        return pl.DataFrame()

    # Concatenate all findings
    return pl.concat(findings, how="vertical_relaxed")
