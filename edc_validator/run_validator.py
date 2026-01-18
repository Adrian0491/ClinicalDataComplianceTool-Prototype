import os
import polars as pl

from edc_validator.domain_validation import (
    validate_vs, validate_ae, validate_cm,
    validate_vs_ae, validate_vs_cm
)
from edc_validator.sdtm_rules import validate_dm

# --- DM anchoring (SDTM-correct) ---
def validate_dm_link(dm: pl.DataFrame, other: pl.DataFrame, other_domain: str) -> pl.DataFrame:
    """
    X_DMLINK_<DOMAIN>_001 (HIGH): other.USUBJID must exist in DM.USUBJID
    """
    # Reuse findings schema from domain_validation
    from edc_validator.domain_validation import FINDINGS_SCHEMA, _empty_findings, _ensure_row_index

    if "USUBJID" not in dm.columns or "USUBJID" not in other.columns:
        return pl.DataFrame({
            "finding_type": ["CROSS_DOMAIN"],
            "rule_id": [f"X_DMLINK_{other_domain}_000"],
            "severity": ["CRIT"],
            "domain": ["CROSS"],
            "field": ["USUBJID"],
            "message": [f"Missing USUBJID for DM/{other_domain} link check."],
            "row_index": [-1],
            "usubjid": [""],
            "evidence": [""],
        }).select([pl.col(c).cast(t, strict=False).alias(c) for c, t in FINDINGS_SCHEMA.items()])

    dm_u = _ensure_row_index(dm).with_columns(
        pl.col("USUBJID").cast(pl.Utf8, strict=False).alias("USUBJID_S")
    ).select("USUBJID_S").unique()

    ot = _ensure_row_index(other).with_columns(
        pl.col("USUBJID").cast(pl.Utf8, strict=False).alias("USUBJID_S")
    )

    orphans = ot.join(dm_u, on="USUBJID_S", how="anti")
    if orphans.height == 0:
        return _empty_findings()

    out = orphans.select([
        pl.lit("CROSS_DOMAIN").alias("finding_type"),
        pl.lit(f"X_DMLINK_{other_domain}_001").alias("rule_id"),
        pl.lit("HIGH").alias("severity"),
        pl.lit("CROSS").alias("domain"),
        pl.lit("USUBJID").alias("field"),
        pl.lit(f"{other_domain} subject not found in DM (orphan USUBJID).").alias("message"),
        pl.col("row_index").alias("row_index"),
        pl.col("USUBJID_S").alias("usubjid"),
        pl.col("USUBJID_S").alias("evidence"),
    ]).select([pl.col(c).cast(t, strict=False).alias(c) for c, t in FINDINGS_SCHEMA.items()])

    return out


def main():
    os.makedirs("output", exist_ok=True)

    # IMPORTANT: dm.csv must exist in mock_data/
    dm = pl.read_csv("mock_data/dm.csv")
    vs = pl.read_csv("mock_data/vs.csv")
    ae = pl.read_csv("mock_data/ae.csv")
    cm = pl.read_csv("mock_data/cm.csv")

    f_dm = validate_dm(dm)
    f_vs = validate_vs(vs)
    f_ae = validate_ae(ae)
    f_cm = validate_cm(cm)

    # SDTM-correct anchoring
    f_dm_vs = validate_dm_link(dm, vs, "VS")
    f_dm_ae = validate_dm_link(dm, ae, "AE")
    f_dm_cm = validate_dm_link(dm, cm, "CM")

    # Optional heuristics you already have
    f_vsae = validate_vs_ae(vs, ae)
    f_vscm = validate_vs_cm(vs, cm)

    findings = pl.concat(
        [f_dm, f_vs, f_ae, f_cm, f_dm_vs, f_dm_ae, f_dm_cm, f_vsae, f_vscm],
        how="vertical_relaxed",
    )

    findings.write_csv("output/findings.csv")
    print(findings)
    print("\nSaved: output/findings.csv")


if __name__ == "__main__":
    main()
