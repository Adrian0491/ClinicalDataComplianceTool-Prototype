from __future__ import annotations

import pandas as pd

from bk.schemas import (
    VS_ALLOWED_TESTCD,
    VS_UNITS_BY_TESTCD,
    AE_ALLOWED_SER,
    AE_ALLOWED_SEV,
    DM_ALLOWED_SEX,
    DM_ALLOWED_AGEU,
    EX_ALLOWED_ROUTE,
    EX_ALLOWED_DOSU,
    LB_ALLOWED_TESTCD,
    LB_ALLOWED_NRIND,
    DS_ALLOWED_DECOD,
    DS_ALLOWED_CAT,
    EG_ALLOWED_TESTCD,
    EG_UNITS_BY_TESTCD,
    FINDINGS_COLUMNS,
    FINDINGS_DTYPES,
    concat_findings,
    dataset_finding,
    empty_findings,
)

from bk.validator.helpers import (
    require_columns,
    ensure_row_index,
    parse_iso_date,
    mk_findings,
)



# ============================================================================
# Demographics (DM)
# ============================================================================

def validate_dm(dm: pd.DataFrame) -> pd.DataFrame:
    findings: list[pd.DataFrame] = []

    crit = require_columns(dm, ["USUBJID", "STUDYID"], "DM", "SDTM_DM")
    if crit:
        return concat_findings(crit)

    dm_i = ensure_row_index(dm)
    dm_i["_USUBJID_S"] = dm_i["USUBJID"].fillna("").astype(str).str.strip()

    if "SEX"     in dm_i.columns: dm_i["_SEX_S"]    = dm_i["SEX"].fillna("").astype(str).str.strip()
    if "AGE"     in dm_i.columns: dm_i["_AGE_N"]    = pd.to_numeric(dm_i["AGE"], errors="coerce")
    if "AGEU"    in dm_i.columns: dm_i["_AGEU_S"]   = dm_i["AGEU"].fillna("").astype(str).str.strip()
    if "RFSTDTC" in dm_i.columns: dm_i["_RFSTDTC_D"] = parse_iso_date(dm_i["RFSTDTC"])
    if "RFENDTC" in dm_i.columns: dm_i["_RFENDTC_D"] = parse_iso_date(dm_i["RFENDTC"])

    # DM_001: USUBJID required non-empty
    findings.append(mk_findings(
        dm_i,
        dm_i["_USUBJID_S"].isin(["", "nan"]) | dm_i["USUBJID"].isna(),
        finding_type="SDTM_RULE", rule_id="SDTM_DM_001", severity="HIGH", domain="DM",
        field="USUBJID", message="USUBJID is required and must be non-empty.",
        evidence_col="USUBJID",
    ))

    # DM_002: USUBJID must be unique in DM
    valid = dm_i[~(dm_i["_USUBJID_S"].isin(["", "nan"]) | dm_i["USUBJID"].isna())]
    dupes = valid[valid["_USUBJID_S"].duplicated(keep=False)]
    if len(dupes):
        findings.append(mk_findings(
            dupes, pd.Series(True, index=dupes.index),
            finding_type="SDTM_RULE", rule_id="SDTM_DM_002", severity="HIGH", domain="DM",
            field="USUBJID", message="USUBJID must be unique in DM.",
            evidence_col="_USUBJID_S",
        ))

    # DM_003: SEX controlled terminology
    if "SEX" in dm_i.columns:
        findings.append(mk_findings(
            dm_i,
            dm_i["SEX"].notna() & ~dm_i["_SEX_S"].isin(DM_ALLOWED_SEX),
            finding_type="SDTM_RULE", rule_id="SDTM_DM_003", severity="MED", domain="DM",
            field="SEX", message=f"SEX must be one of: {DM_ALLOWED_SEX}.",
            evidence_col="_SEX_S",
        ))

    # DM_004: AGE reasonable bounds
    if "AGE" in dm_i.columns:
        findings.append(mk_findings(
            dm_i,
            dm_i["_AGE_N"].notna() & ((dm_i["_AGE_N"] < 0) | (dm_i["_AGE_N"] > 120)),
            finding_type="SDTM_RULE", rule_id="SDTM_DM_004", severity="MED", domain="DM",
            field="AGE", message="AGE should be between 0 and 120.",
            evidence_col="AGE",
        ))

    # DM_005: AGEU valid when AGE present
    if "AGE" in dm_i.columns and "AGEU" in dm_i.columns:
        findings.append(mk_findings(
            dm_i,
            dm_i["_AGE_N"].notna() & ~dm_i["_AGEU_S"].isin(DM_ALLOWED_AGEU),
            finding_type="SDTM_RULE", rule_id="SDTM_DM_005", severity="LOW", domain="DM",
            field="AGEU", message=f"AGEU should be one of: {DM_ALLOWED_AGEU} when AGE is present.",
            evidence_col="_AGEU_S",
        ))

    # DM_006 / DM_007: date fields parseable
    if "RFSTDTC" in dm_i.columns:
        findings.append(mk_findings(
            dm_i, dm_i["RFSTDTC"].notna() & dm_i["_RFSTDTC_D"].isna(),
            finding_type="SDTM_RULE", rule_id="SDTM_DM_006", severity="LOW", domain="DM",
            field="RFSTDTC", message="RFSTDTC should be ISO date YYYY-MM-DD.",
            evidence_col="RFSTDTC",
        ))
    if "RFENDTC" in dm_i.columns:
        findings.append(mk_findings(
            dm_i, dm_i["RFENDTC"].notna() & dm_i["_RFENDTC_D"].isna(),
            finding_type="SDTM_RULE", rule_id="SDTM_DM_007", severity="LOW", domain="DM",
            field="RFENDTC", message="RFENDTC should be ISO date YYYY-MM-DD.",
            evidence_col="RFENDTC",
        ))

    # DM_008: RFSTDTC <= RFENDTC
    if "RFSTDTC" in dm_i.columns and "RFENDTC" in dm_i.columns:
        both = dm_i["_RFSTDTC_D"].notna() & dm_i["_RFENDTC_D"].notna()
        findings.append(mk_findings(
            dm_i,
            both & (dm_i["_RFSTDTC_D"] > dm_i["_RFENDTC_D"]),
            finding_type="SDTM_RULE", rule_id="SDTM_DM_008", severity="HIGH", domain="DM",
            field="RFSTDTC/RFENDTC", message="RFSTDTC must be on or before RFENDTC.",
            evidence_fn=lambda r: f"{r.get('RFSTDTC','')} > {r.get('RFENDTC','')}",
        ))

    return concat_findings(findings)


# ============================================================================
# Vital Signs (VS)
# ============================================================================

def validate_vs(vs: pd.DataFrame) -> pd.DataFrame:
    findings: list[pd.DataFrame] = []

    crit = require_columns(vs, ["USUBJID", "VSTESTCD", "VSORRES", "VSDTC"], "VS", "SDTM_VS")
    if crit:
        return concat_findings(crit)

    vs_i = ensure_row_index(vs)
    vs_i["_USUBJID_S"]  = vs_i["USUBJID"].fillna("").astype(str).str.strip()
    vs_i["_VSTESTCD_S"] = vs_i["VSTESTCD"].fillna("").astype(str).str.strip()
    vs_i["_VSORRES_S"]  = vs_i["VSORRES"].fillna("").astype(str)
    vs_i["_VSDTC_D"]   = parse_iso_date(vs_i["VSDTC"])
    vs_i["_VSORRES_N"]  = pd.to_numeric(
        vs_i["_VSORRES_S"].str.replace(",", ".", regex=False), errors="coerce"
    )

    # VS_001: USUBJID required
    findings.append(mk_findings(
        vs_i,
        vs_i["_USUBJID_S"].isin(["", "nan"]) | vs_i["USUBJID"].isna(),
        finding_type="SDTM_RULE", rule_id="SDTM_VS_001", severity="HIGH", domain="VS",
        field="USUBJID", message="USUBJID is required and must be non-empty.",
        evidence_col="USUBJID",
    ))

    # VS_002: VSTESTCD allowed set
    findings.append(mk_findings(
        vs_i,
        vs_i["VSTESTCD"].notna() & ~vs_i["_VSTESTCD_S"].isin(VS_ALLOWED_TESTCD),
        finding_type="SDTM_RULE", rule_id="SDTM_VS_002", severity="MED", domain="VS",
        field="VSTESTCD", message=f"VSTESTCD should be one of: {VS_ALLOWED_TESTCD}.",
        evidence_col="_VSTESTCD_S",
    ))

    # VS_003: VSDTC parseable
    findings.append(mk_findings(
        vs_i,
        vs_i["VSDTC"].notna() & vs_i["_VSDTC_D"].isna(),
        finding_type="SDTM_RULE", rule_id="SDTM_VS_003", severity="LOW", domain="VS",
        field="VSDTC", message="VSDTC should be ISO date YYYY-MM-DD.",
        evidence_col="VSDTC",
    ))

    # VS_004: Numeric result for numeric tests
    findings.append(mk_findings(
        vs_i,
        vs_i["_VSTESTCD_S"].isin(VS_ALLOWED_TESTCD)
        & vs_i["VSORRES"].notna()
        & vs_i["_VSORRES_N"].isna(),
        finding_type="SDTM_RULE", rule_id="SDTM_VS_004", severity="HIGH", domain="VS",
        field="VSORRES", message="VSORRES should be numeric for this VSTESTCD.",
        evidence_col="_VSORRES_S",
    ))

    # VS_005: Units consistency
    if "VSORRESU" in vs_i.columns:
        vs_i["_VSORRESU_S"] = vs_i["VSORRESU"].fillna("").astype(str).str.strip()
        valid_pairs: set[tuple] = {
            (tc, u) for tc, units in VS_UNITS_BY_TESTCD.items() for u in units
        }
        known_testcds = set(VS_UNITS_BY_TESTCD.keys())
        bad_mask = (
            vs_i["_VSTESTCD_S"].isin(known_testcds)
            & vs_i["VSORRESU"].notna()
            & vs_i.apply(
                lambda r: (r["_VSTESTCD_S"], r["_VSORRESU_S"]) not in valid_pairs, axis=1
            )
        )
        findings.append(mk_findings(
            vs_i, bad_mask,
            finding_type="SDTM_RULE", rule_id="SDTM_VS_005", severity="MED", domain="VS",
            field="VSORRESU", message="VSORRESU unit is not consistent with VSTESTCD.",
            evidence_fn=lambda r: f"{r['_VSTESTCD_S']} / {r['_VSORRESU_S']}",
        ))
    else:
        findings.append(dataset_finding(
            rule_id="SDTM_VS_005", severity="LOW", domain="VS", field="VSORRESU",
            message="Column VSORRESU not present; unit consistency checks skipped.",
        ))

    return concat_findings(findings)


# ============================================================================
# Adverse Events (AE)
# ============================================================================

def validate_ae(ae: pd.DataFrame) -> pd.DataFrame:
    findings: list[pd.DataFrame] = []

    crit = require_columns(ae, ["USUBJID", "AETERM", "AESTDTC"], "AE", "SDTM_AE")
    if crit:
        return concat_findings(crit)

    ae_i = ensure_row_index(ae)
    ae_i["_USUBJID_S"] = ae_i["USUBJID"].fillna("").astype(str).str.strip()
    ae_i["_AETERM_S"]  = ae_i["AETERM"].fillna("").astype(str).str.strip()
    ae_i["_AESTDTC_D"] = parse_iso_date(ae_i["AESTDTC"])
    if "AEENDTC" in ae_i.columns:
        ae_i["_AEENDTC_D"] = parse_iso_date(ae_i["AEENDTC"])

    # AE_001: AETERM required
    findings.append(mk_findings(
        ae_i,
        ae_i["_AETERM_S"].isin(["", "nan"]) | ae_i["AETERM"].isna(),
        finding_type="SDTM_RULE", rule_id="SDTM_AE_001", severity="HIGH", domain="AE",
        field="AETERM", message="AETERM is required and must be non-empty.",
        evidence_col="AETERM",
    ))

    # AE_002: AESTDTC parseable
    findings.append(mk_findings(
        ae_i,
        ae_i["AESTDTC"].notna() & ae_i["_AESTDTC_D"].isna(),
        finding_type="SDTM_RULE", rule_id="SDTM_AE_002", severity="MED", domain="AE",
        field="AESTDTC", message="AESTDTC should be ISO date YYYY-MM-DD.",
        evidence_col="AESTDTC",
    ))

    if "AEENDTC" in ae_i.columns:
        # AE_003: AEENDTC parseable
        findings.append(mk_findings(
            ae_i,
            ae_i["AEENDTC"].notna() & ae_i["_AEENDTC_D"].isna(),
            finding_type="SDTM_RULE", rule_id="SDTM_AE_003", severity="LOW", domain="AE",
            field="AEENDTC", message="AEENDTC should be ISO date YYYY-MM-DD.",
            evidence_col="AEENDTC",
        ))

        # AE_004: AESTDTC <= AEENDTC
        both = ae_i["_AESTDTC_D"].notna() & ae_i["_AEENDTC_D"].notna()
        findings.append(mk_findings(
            ae_i,
            both & (ae_i["_AESTDTC_D"] > ae_i["_AEENDTC_D"]),
            finding_type="SDTM_RULE", rule_id="SDTM_AE_004", severity="HIGH", domain="AE",
            field="AESTDTC/AEENDTC", message="AESTDTC must be on or before AEENDTC.",
            evidence_fn=lambda r: f"{r.get('AESTDTC','')} > {r.get('AEENDTC','')}",
        ))

    # AE_005: AESER controlled terminology
    if "AESER" in ae_i.columns:
        findings.append(mk_findings(
            ae_i,
            ae_i["AESER"].notna() & ~ae_i["AESER"].astype(str).isin(AE_ALLOWED_SER),
            finding_type="SDTM_RULE", rule_id="SDTM_AE_005", severity="MED", domain="AE",
            field="AESER", message=f"AESER should be one of: {AE_ALLOWED_SER}.",
            evidence_col="AESER",
        ))

    # AE_006: AESEV controlled terminology
    if "AESEV" in ae_i.columns:
        findings.append(mk_findings(
            ae_i,
            ae_i["AESEV"].notna() & ~ae_i["AESEV"].astype(str).isin(AE_ALLOWED_SEV),
            finding_type="SDTM_RULE", rule_id="SDTM_AE_006", severity="MED", domain="AE",
            field="AESEV", message=f"AESEV should be one of: {AE_ALLOWED_SEV}.",
            evidence_col="AESEV",
        ))

    return concat_findings(findings)


# ============================================================================
# Concomitant Medications (CM)
# ============================================================================

def validate_cm(cm: pd.DataFrame) -> pd.DataFrame:
    findings: list[pd.DataFrame] = []

    crit = require_columns(cm, ["USUBJID", "CMTRT", "CMSTDTC"], "CM", "SDTM_CM")
    if crit:
        return concat_findings(crit)

    cm_i = ensure_row_index(cm)
    cm_i["_USUBJID_S"] = cm_i["USUBJID"].fillna("").astype(str).str.strip()
    cm_i["_CMTRT_S"]   = cm_i["CMTRT"].fillna("").astype(str).str.strip()
    cm_i["_CMSTDTC_D"] = parse_iso_date(cm_i["CMSTDTC"])
    if "CMENDTC" in cm_i.columns:
        cm_i["_CMENDTC_D"] = parse_iso_date(cm_i["CMENDTC"])

    # CM_001: CMTRT required
    findings.append(mk_findings(
        cm_i,
        cm_i["_CMTRT_S"].isin(["", "nan"]) | cm_i["CMTRT"].isna(),
        finding_type="SDTM_RULE", rule_id="SDTM_CM_001", severity="HIGH", domain="CM",
        field="CMTRT", message="CMTRT is required and must be non-empty.",
        evidence_col="CMTRT",
    ))

    # CM_002: CMSTDTC parseable
    findings.append(mk_findings(
        cm_i,
        cm_i["CMSTDTC"].notna() & cm_i["_CMSTDTC_D"].isna(),
        finding_type="SDTM_RULE", rule_id="SDTM_CM_002", severity="MED", domain="CM",
        field="CMSTDTC", message="CMSTDTC should be ISO date YYYY-MM-DD.",
        evidence_col="CMSTDTC",
    ))

    if "CMENDTC" in cm_i.columns:
        # CM_003: CMENDTC parseable
        findings.append(mk_findings(
            cm_i,
            cm_i["CMENDTC"].notna() & cm_i["_CMENDTC_D"].isna(),
            finding_type="SDTM_RULE", rule_id="SDTM_CM_003", severity="LOW", domain="CM",
            field="CMENDTC", message="CMENDTC should be ISO date YYYY-MM-DD.",
            evidence_col="CMENDTC",
        ))

        # CM_004: CMSTDTC <= CMENDTC
        both = cm_i["_CMSTDTC_D"].notna() & cm_i["_CMENDTC_D"].notna()
        findings.append(mk_findings(
            cm_i,
            both & (cm_i["_CMSTDTC_D"] > cm_i["_CMENDTC_D"]),
            finding_type="SDTM_RULE", rule_id="SDTM_CM_004", severity="HIGH", domain="CM",
            field="CMSTDTC/CMENDTC", message="CMSTDTC must be on or before CMENDTC.",
            evidence_fn=lambda r: f"{r.get('CMSTDTC','')} > {r.get('CMENDTC','')}",
        ))

    return concat_findings(findings)

# ============================================================================
# Exposure (EX)
# ============================================================================

def validate_ex(ex: pd.DataFrame) -> pd.DataFrame:
    findings: list[pd.DataFrame] = []

    crit = require_columns(ex, ["USUBJID", "EXTRT", "EXSTDTC"], "EX", "SDTM_EX")
    if crit:
        return concat_findings(crit)

    ex_i = ensure_row_index(ex)
    ex_i["_USUBJID_S"] = ex_i["USUBJID"].fillna("").astype(str).str.strip()
    ex_i["_EXTRT_S"]   = ex_i["EXTRT"].fillna("").astype(str).str.strip()
    ex_i["_EXSTDTC_D"] = parse_iso_date(ex_i["EXSTDTC"])
    if "EXENDTC" in ex_i.columns:
        ex_i["_EXENDTC_D"] = parse_iso_date(ex_i["EXENDTC"])
    if "EXDOSE" in ex_i.columns:
        ex_i["_EXDOSE_N"] = pd.to_numeric(ex_i["EXDOSE"], errors="coerce")

    # EX_001: USUBJID required
    findings.append(mk_findings(
        ex_i,
        ex_i["_USUBJID_S"].isin(["", "nan"]) | ex_i["USUBJID"].isna(),
        finding_type="SDTM_RULE", rule_id="SDTM_EX_001", severity="HIGH", domain="EX",
        field="USUBJID", message="USUBJID is required and must be non-empty.",
        evidence_col="USUBJID",
    ))

    # EX_002: EXTRT required
    findings.append(mk_findings(
        ex_i,
        ex_i["_EXTRT_S"].isin(["", "nan"]) | ex_i["EXTRT"].isna(),
        finding_type="SDTM_RULE", rule_id="SDTM_EX_002", severity="HIGH", domain="EX",
        field="EXTRT", message="EXTRT is required and must be non-empty.",
        evidence_col="EXTRT",
    ))

    if "EXDOSE" in ex_i.columns:
        # EX_003: EXDOSE numeric
        findings.append(mk_findings(
            ex_i,
            ex_i["EXDOSE"].notna() & ex_i["_EXDOSE_N"].isna(),
            finding_type="SDTM_RULE", rule_id="SDTM_EX_003", severity="HIGH", domain="EX",
            field="EXDOSE", message="EXDOSE should be numeric.",
            evidence_col="EXDOSE",
        ))

        # EX_004: EXDOSE non-negative
        findings.append(mk_findings(
            ex_i,
            ex_i["_EXDOSE_N"].notna() & (ex_i["_EXDOSE_N"] < 0),
            finding_type="SDTM_RULE", rule_id="SDTM_EX_004", severity="MED", domain="EX",
            field="EXDOSE", message="EXDOSE should not be negative.",
            evidence_col="EXDOSE",
        ))

    # EX_005: EXSTDTC parseable
    findings.append(mk_findings(
        ex_i,
        ex_i["EXSTDTC"].notna() & ex_i["_EXSTDTC_D"].isna(),
        finding_type="SDTM_RULE", rule_id="SDTM_EX_005", severity="LOW", domain="EX",
        field="EXSTDTC", message="EXSTDTC should be ISO date YYYY-MM-DD.",
        evidence_col="EXSTDTC",
    ))

    if "EXENDTC" in ex_i.columns:
        # EX_006: EXENDTC parseable
        findings.append(mk_findings(
            ex_i,
            ex_i["EXENDTC"].notna() & ex_i["_EXENDTC_D"].isna(),
            finding_type="SDTM_RULE", rule_id="SDTM_EX_006", severity="LOW", domain="EX",
            field="EXENDTC", message="EXENDTC should be ISO date YYYY-MM-DD.",
            evidence_col="EXENDTC",
        ))

        # EX_007: EXSTDTC <= EXENDTC
        both = ex_i["_EXSTDTC_D"].notna() & ex_i["_EXENDTC_D"].notna()
        findings.append(mk_findings(
            ex_i,
            both & (ex_i["_EXSTDTC_D"] > ex_i["_EXENDTC_D"]),
            finding_type="SDTM_RULE", rule_id="SDTM_EX_007", severity="HIGH", domain="EX",
            field="EXSTDTC/EXENDTC", message="EXSTDTC must be on or before EXENDTC.",
            evidence_fn=lambda r: f"{r.get('EXSTDTC','')} > {r.get('EXENDTC','')}",
        ))

    # EX_008: EXROUTE controlled terminology
    if "EXROUTE" in ex_i.columns:
        findings.append(mk_findings(
            ex_i,
            ex_i["EXROUTE"].notna() & ~ex_i["EXROUTE"].astype(str).str.strip().isin(EX_ALLOWED_ROUTE),
            finding_type="SDTM_RULE", rule_id="SDTM_EX_008", severity="MED", domain="EX",
            field="EXROUTE", message=f"EXROUTE should be one of: {EX_ALLOWED_ROUTE}.",
            evidence_col="EXROUTE",
        ))

    # EX_009: EXDOSU controlled terminology
    if "EXDOSU" in ex_i.columns:
        findings.append(mk_findings(
            ex_i,
            ex_i["EXDOSU"].notna() & ~ex_i["EXDOSU"].astype(str).str.strip().isin(EX_ALLOWED_DOSU),
            finding_type="SDTM_RULE", rule_id="SDTM_EX_009", severity="MED", domain="EX",
            field="EXDOSU", message=f"EXDOSU should be one of: {EX_ALLOWED_DOSU}.",
            evidence_col="EXDOSU",
        ))

    return concat_findings(findings)


# ============================================================================
# Laboratory Test Results (LB)
# ============================================================================

def validate_lb(lb: pd.DataFrame) -> pd.DataFrame:
    findings: list[pd.DataFrame] = []

    crit = require_columns(lb, ["USUBJID", "LBTESTCD", "LBORRES", "LBDTC"], "LB", "SDTM_LB")
    if crit:
        return concat_findings(crit)

    lb_i = ensure_row_index(lb)
    lb_i["_USUBJID_S"]  = lb_i["USUBJID"].fillna("").astype(str).str.strip()
    lb_i["_LBTESTCD_S"] = lb_i["LBTESTCD"].fillna("").astype(str).str.strip()
    lb_i["_LBORRES_S"]  = lb_i["LBORRES"].fillna("").astype(str).str.strip()
    lb_i["_LBDTC_D"]    = parse_iso_date(lb_i["LBDTC"])

    # LB_001: USUBJID required
    findings.append(mk_findings(
        lb_i,
        lb_i["_USUBJID_S"].isin(["", "nan"]) | lb_i["USUBJID"].isna(),
        finding_type="SDTM_RULE", rule_id="SDTM_LB_001", severity="HIGH", domain="LB",
        field="USUBJID", message="USUBJID is required and must be non-empty.",
        evidence_col="USUBJID",
    ))

    # LB_002: LBORRES required
    findings.append(mk_findings(
        lb_i,
        lb_i["_LBORRES_S"].isin(["", "nan"]) | lb_i["LBORRES"].isna(),
        finding_type="SDTM_RULE", rule_id="SDTM_LB_002", severity="HIGH", domain="LB",
        field="LBORRES", message="LBORRES is required and must be non-empty.",
        evidence_col="LBORRES",
    ))

    # LB_003: LBTESTCD controlled terminology
    findings.append(mk_findings(
        lb_i,
        lb_i["LBTESTCD"].notna() & ~lb_i["_LBTESTCD_S"].isin(LB_ALLOWED_TESTCD),
        finding_type="SDTM_RULE", rule_id="SDTM_LB_003", severity="MED", domain="LB",
        field="LBTESTCD", message=f"LBTESTCD should be one of: {LB_ALLOWED_TESTCD}.",
        evidence_col="_LBTESTCD_S",
    ))

    # LB_004: LBDTC parseable
    findings.append(mk_findings(
        lb_i,
        lb_i["LBDTC"].notna() & lb_i["_LBDTC_D"].isna(),
        finding_type="SDTM_RULE", rule_id="SDTM_LB_004", severity="LOW", domain="LB",
        field="LBDTC", message="LBDTC should be ISO date YYYY-MM-DD.",
        evidence_col="LBDTC",
    ))

    # LB_005: LBSTRESN numeric when present
    if "LBSTRESN" in lb_i.columns:
        lb_i["_LBSTRESN_N"] = pd.to_numeric(lb_i["LBSTRESN"], errors="coerce")
        findings.append(mk_findings(
            lb_i,
            lb_i["LBSTRESN"].notna() & lb_i["_LBSTRESN_N"].isna(),
            finding_type="SDTM_RULE", rule_id="SDTM_LB_005", severity="MED", domain="LB",
            field="LBSTRESN", message="LBSTRESN should be numeric.",
            evidence_col="LBSTRESN",
        ))

    # LB_006: LBNRIND controlled terminology
    if "LBNRIND" in lb_i.columns:
        findings.append(mk_findings(
            lb_i,
            lb_i["LBNRIND"].notna() & ~lb_i["LBNRIND"].astype(str).str.strip().isin(LB_ALLOWED_NRIND),
            finding_type="SDTM_RULE", rule_id="SDTM_LB_006", severity="LOW", domain="LB",
            field="LBNRIND", message=f"LBNRIND should be one of: {LB_ALLOWED_NRIND}.",
            evidence_col="LBNRIND",
        ))

    return concat_findings(findings)


# ============================================================================
# Disposition (DS)
# ============================================================================

def validate_ds(ds: pd.DataFrame) -> pd.DataFrame:
    findings: list[pd.DataFrame] = []

    crit = require_columns(ds, ["USUBJID", "DSDECOD", "DSSTDTC"], "DS", "SDTM_DS")
    if crit:
        return concat_findings(crit)

    ds_i = ensure_row_index(ds)
    ds_i["_USUBJID_S"] = ds_i["USUBJID"].fillna("").astype(str).str.strip()
    ds_i["_DSDECOD_S"] = ds_i["DSDECOD"].fillna("").astype(str).str.strip()
    ds_i["_DSSTDTC_D"] = parse_iso_date(ds_i["DSSTDTC"])

    # DS_001: USUBJID required
    findings.append(mk_findings(
        ds_i,
        ds_i["_USUBJID_S"].isin(["", "nan"]) | ds_i["USUBJID"].isna(),
        finding_type="SDTM_RULE", rule_id="SDTM_DS_001", severity="HIGH", domain="DS",
        field="USUBJID", message="USUBJID is required and must be non-empty.",
        evidence_col="USUBJID",
    ))

    # DS_002: DSDECOD required
    findings.append(mk_findings(
        ds_i,
        ds_i["_DSDECOD_S"].isin(["", "nan"]) | ds_i["DSDECOD"].isna(),
        finding_type="SDTM_RULE", rule_id="SDTM_DS_002", severity="HIGH", domain="DS",
        field="DSDECOD", message="DSDECOD is required and must be non-empty.",
        evidence_col="DSDECOD",
    ))

    # DS_003: DSDECOD controlled terminology
    findings.append(mk_findings(
        ds_i,
        ds_i["DSDECOD"].notna() & ~ds_i["_DSDECOD_S"].isin(DS_ALLOWED_DECOD),
        finding_type="SDTM_RULE", rule_id="SDTM_DS_003", severity="MED", domain="DS",
        field="DSDECOD", message=f"DSDECOD should be one of: {DS_ALLOWED_DECOD}.",
        evidence_col="_DSDECOD_S",
    ))

    # DS_004: DSSTDTC parseable
    findings.append(mk_findings(
        ds_i,
        ds_i["DSSTDTC"].notna() & ds_i["_DSSTDTC_D"].isna(),
        finding_type="SDTM_RULE", rule_id="SDTM_DS_004", severity="LOW", domain="DS",
        field="DSSTDTC", message="DSSTDTC should be ISO date YYYY-MM-DD.",
        evidence_col="DSSTDTC",
    ))

    # DS_005: DSTERM required when present
    if "DSTERM" in ds_i.columns:
        ds_i["_DSTERM_S"] = ds_i["DSTERM"].fillna("").astype(str).str.strip()
        findings.append(mk_findings(
            ds_i,
            ds_i["_DSTERM_S"].isin(["", "nan"]) | ds_i["DSTERM"].isna(),
            finding_type="SDTM_RULE", rule_id="SDTM_DS_005", severity="MED", domain="DS",
            field="DSTERM", message="DSTERM is required and must be non-empty.",
            evidence_col="DSTERM",
        ))

    # DS_006: DSCAT controlled terminology
    if "DSCAT" in ds_i.columns:
        findings.append(mk_findings(
            ds_i,
            ds_i["DSCAT"].notna() & ~ds_i["DSCAT"].astype(str).str.strip().isin(DS_ALLOWED_CAT),
            finding_type="SDTM_RULE", rule_id="SDTM_DS_006", severity="LOW", domain="DS",
            field="DSCAT", message=f"DSCAT should be one of: {DS_ALLOWED_CAT}.",
            evidence_col="DSCAT",
        ))

    return concat_findings(findings)


# ============================================================================
# ECG Test Results (EG)
# ============================================================================

def validate_eg(eg: pd.DataFrame) -> pd.DataFrame:
    findings: list[pd.DataFrame] = []

    crit = require_columns(eg, ["USUBJID", "EGTESTCD", "EGORRES", "EGDTC"], "EG", "SDTM_EG")
    if crit:
        return concat_findings(crit)

    eg_i = ensure_row_index(eg)
    eg_i["_USUBJID_S"]  = eg_i["USUBJID"].fillna("").astype(str).str.strip()
    eg_i["_EGTESTCD_S"] = eg_i["EGTESTCD"].fillna("").astype(str).str.strip()
    eg_i["_EGORRES_S"]  = eg_i["EGORRES"].fillna("").astype(str)
    eg_i["_EGDTC_D"]    = parse_iso_date(eg_i["EGDTC"])
    eg_i["_EGORRES_N"]  = pd.to_numeric(
        eg_i["_EGORRES_S"].str.replace(",", ".", regex=False), errors="coerce"
    )

    # EG_001: USUBJID required
    findings.append(mk_findings(
        eg_i,
        eg_i["_USUBJID_S"].isin(["", "nan"]) | eg_i["USUBJID"].isna(),
        finding_type="SDTM_RULE", rule_id="SDTM_EG_001", severity="HIGH", domain="EG",
        field="USUBJID", message="USUBJID is required and must be non-empty.",
        evidence_col="USUBJID",
    ))

    # EG_002: EGTESTCD allowed set
    findings.append(mk_findings(
        eg_i,
        eg_i["EGTESTCD"].notna() & ~eg_i["_EGTESTCD_S"].isin(EG_ALLOWED_TESTCD),
        finding_type="SDTM_RULE", rule_id="SDTM_EG_002", severity="MED", domain="EG",
        field="EGTESTCD", message=f"EGTESTCD should be one of: {EG_ALLOWED_TESTCD}.",
        evidence_col="_EGTESTCD_S",
    ))

    # EG_003: EGDTC parseable
    findings.append(mk_findings(
        eg_i,
        eg_i["EGDTC"].notna() & eg_i["_EGDTC_D"].isna(),
        finding_type="SDTM_RULE", rule_id="SDTM_EG_003", severity="LOW", domain="EG",
        field="EGDTC", message="EGDTC should be ISO date YYYY-MM-DD.",
        evidence_col="EGDTC",
    ))

    # EG_004: Numeric result for numeric tests
    findings.append(mk_findings(
        eg_i,
        eg_i["_EGTESTCD_S"].isin(EG_ALLOWED_TESTCD)
        & eg_i["EGORRES"].notna()
        & eg_i["_EGORRES_N"].isna(),
        finding_type="SDTM_RULE", rule_id="SDTM_EG_004", severity="HIGH", domain="EG",
        field="EGORRES", message="EGORRES should be numeric for this EGTESTCD.",
        evidence_col="_EGORRES_S",
    ))

    # EG_005: Units consistency
    if "EGORRESU" in eg_i.columns:
        eg_i["_EGORRESU_S"] = eg_i["EGORRESU"].fillna("").astype(str).str.strip()
        valid_pairs: set[tuple] = {
            (tc, u) for tc, units in EG_UNITS_BY_TESTCD.items() for u in units
        }
        known_testcds = set(EG_UNITS_BY_TESTCD.keys())
        bad_mask = (
            eg_i["_EGTESTCD_S"].isin(known_testcds)
            & eg_i["EGORRESU"].notna()
            & eg_i.apply(
                lambda r: (r["_EGTESTCD_S"], r["_EGORRESU_S"]) not in valid_pairs, axis=1
            )
        )
        findings.append(mk_findings(
            eg_i, bad_mask,
            finding_type="SDTM_RULE", rule_id="SDTM_EG_005", severity="MED", domain="EG",
            field="EGORRESU", message="EGORRESU unit is not consistent with EGTESTCD.",
            evidence_fn=lambda r: f"{r['_EGTESTCD_S']} / {r['_EGORRESU_S']}",
        ))
    else:
        findings.append(dataset_finding(
            rule_id="SDTM_EG_005", severity="LOW", domain="EG", field="EGORRESU",
            message="Column EGORRESU not present; unit consistency checks skipped.",
        ))

    return concat_findings(findings)


# ============================================================================
# Cross-domain rules
# ============================================================================

def validate_dm_link(dm: pd.DataFrame, other: pd.DataFrame, other_domain: str) -> pd.DataFrame:
    """
    X_DMLINK_<DOMAIN>_001 (HIGH): every USUBJID in `other` must exist in DM.
    """
    if "USUBJID" not in dm.columns or "USUBJID" not in other.columns:
        return dataset_finding(
            rule_id=f"X_DMLINK_{other_domain}_000", severity="CRIT",
            domain="CROSS", field="USUBJID", finding_type="CROSS_DOMAIN",
            message=f"Missing USUBJID for DM/{other_domain} link check.",
        )

    dm_subjects = set(dm["USUBJID"].dropna().astype(str).str.strip())
    other_i = ensure_row_index(other)
    other_i["_USUBJID_S"] = other_i["USUBJID"].fillna("").astype(str).str.strip()
    orphans = other_i[~other_i["_USUBJID_S"].isin(dm_subjects) & (other_i["_USUBJID_S"] != "")]

    if len(orphans) == 0:
        return empty_findings()

    return mk_findings(
        orphans, pd.Series(True, index=orphans.index),
        finding_type="CROSS_DOMAIN", rule_id=f"X_DMLINK_{other_domain}_001",
        severity="HIGH", domain="CROSS", field="USUBJID",
        message=f"{other_domain} subject not found in DM (orphan USUBJID).",
        evidence_col="_USUBJID_S",
    )


def validate_vs_ae(vs: pd.DataFrame, ae: pd.DataFrame) -> pd.DataFrame:
    """
    VS ↔ AE cross-domain heuristics:
      X_VSAE_001: AE subjects must exist in VS
      X_VSAE_002: AE start date should not be outside the subject's VS date range
    """
    findings: list[pd.DataFrame] = []

    if any(c not in vs.columns for c in ["USUBJID", "VSDTC"]) or \
       any(c not in ae.columns for c in ["USUBJID", "AESTDTC"]):
        return dataset_finding(
            rule_id="X_VSAE_000", severity="CRIT", domain="CROSS",
            field="USUBJID/--DTC", finding_type="CROSS_DOMAIN",
            message="Missing required columns for VS/AE cross checks.",
        )

    vs_i = ensure_row_index(vs)
    vs_i["_USUBJID_S"] = vs_i["USUBJID"].fillna("").astype(str).str.strip()
    vs_i["_VSDTC_D"]   = parse_iso_date(vs_i["VSDTC"])

    ae_i = ensure_row_index(ae)
    ae_i["_USUBJID_S"]  = ae_i["USUBJID"].fillna("").astype(str).str.strip()
    ae_i["_AESTDTC_D"]  = parse_iso_date(ae_i["AESTDTC"])

    # X_VSAE_001: orphan AE subjects
    vs_subjects = set(vs_i["_USUBJID_S"].unique())
    orphans = ae_i[~ae_i["_USUBJID_S"].isin(vs_subjects) & (ae_i["_USUBJID_S"] != "")]
    if len(orphans):
        findings.append(mk_findings(
            orphans, pd.Series(True, index=orphans.index),
            finding_type="CROSS_DOMAIN", rule_id="X_VSAE_001",
            severity="HIGH", domain="CROSS", field="USUBJID",
            message="AE subject not found in VS (orphan USUBJID).",
            evidence_col="_USUBJID_S",
        ))

    # X_VSAE_002: AE start date outside VS date range per subject
    vs_range = (
        vs_i[vs_i["_VSDTC_D"].notna()]
        .groupby("_USUBJID_S")["_VSDTC_D"]
        .agg(VS_MIN="min", VS_MAX="max")
        .reset_index()
    )
    ae_dated = ae_i[ae_i["_AESTDTC_D"].notna()].merge(vs_range, on="_USUBJID_S", how="inner")
    out_of_range = ae_dated[
        (ae_dated["_AESTDTC_D"] < ae_dated["VS_MIN"]) |
        (ae_dated["_AESTDTC_D"] > ae_dated["VS_MAX"])
    ]
    if len(out_of_range):
        findings.append(mk_findings(
            out_of_range, pd.Series(True, index=out_of_range.index),
            finding_type="CROSS_DOMAIN", rule_id="X_VSAE_002",
            severity="MED", domain="CROSS", field="AESTDTC",
            message="AE start date is outside the subject's VS date range.",
            evidence_fn=lambda r: f"{r.get('AESTDTC','')} vs [{r['VS_MIN']}, {r['VS_MAX']}]",
        ))

    return concat_findings(findings)


def validate_vs_cm(vs: pd.DataFrame, cm: pd.DataFrame) -> pd.DataFrame:
    """
    VS ↔ CM cross-domain heuristics:
      X_VSCM_001: CM subjects must exist in VS
      X_VSCM_002: VS dates outside the subject's CM medication window
    """
    findings: list[pd.DataFrame] = []

    if any(c not in vs.columns for c in ["USUBJID", "VSDTC"]) or \
       any(c not in cm.columns for c in ["USUBJID", "CMSTDTC"]):
        return dataset_finding(
            rule_id="X_VSCM_000", severity="CRIT", domain="CROSS",
            field="USUBJID/--DTC", finding_type="CROSS_DOMAIN",
            message="Missing required columns for VS/CM cross checks.",
        )

    vs_i = ensure_row_index(vs)
    vs_i["_USUBJID_S"] = vs_i["USUBJID"].fillna("").astype(str).str.strip()
    vs_i["_VSDTC_D"]   = parse_iso_date(vs_i["VSDTC"])

    cm_i = ensure_row_index(cm)
    cm_i["_USUBJID_S"]  = cm_i["USUBJID"].fillna("").astype(str).str.strip()
    cm_i["_CMSTDTC_D"]  = parse_iso_date(cm_i["CMSTDTC"])
    if "CMENDTC" in cm_i.columns:
        cm_i["_CMENDTC_D"] = parse_iso_date(cm_i["CMENDTC"])

    # X_VSCM_001: orphan CM subjects
    vs_subjects = set(vs_i["_USUBJID_S"].unique())
    orphans = cm_i[~cm_i["_USUBJID_S"].isin(vs_subjects) & (cm_i["_USUBJID_S"] != "")]
    if len(orphans):
        findings.append(mk_findings(
            orphans, pd.Series(True, index=orphans.index),
            finding_type="CROSS_DOMAIN", rule_id="X_VSCM_001",
            severity="HIGH", domain="CROSS", field="USUBJID",
            message="CM subject not found in VS (orphan USUBJID).",
            evidence_col="_USUBJID_S",
        ))

    # X_VSCM_002: VS dates outside CM window
    if "CMENDTC" in cm_i.columns:
        cm_windows = (
            cm_i[cm_i["_CMSTDTC_D"].notna()]
            .groupby("_USUBJID_S")
            .agg(CM_MIN=("_CMSTDTC_D", "min"), CM_MAX=("_CMENDTC_D", "max"))
            .reset_index()
        )
        vs_dated = vs_i[vs_i["_VSDTC_D"].notna()].merge(cm_windows, on="_USUBJID_S", how="inner")
        out = vs_dated[
            vs_dated["CM_MAX"].notna() & (
                (vs_dated["_VSDTC_D"] < vs_dated["CM_MIN"]) |
                (vs_dated["_VSDTC_D"] > vs_dated["CM_MAX"])
            )
        ]
        if len(out):
            findings.append(mk_findings(
                out, pd.Series(True, index=out.index),
                finding_type="CROSS_DOMAIN", rule_id="X_VSCM_002",
                severity="LOW", domain="CROSS", field="VSDTC",
                message="VS date is outside the subject's CM medication window.",
                evidence_fn=lambda r: f"{r.get('VSDTC','')} vs [{r['CM_MIN']}, {r['CM_MAX']}]",
            ))

    return concat_findings(findings)
