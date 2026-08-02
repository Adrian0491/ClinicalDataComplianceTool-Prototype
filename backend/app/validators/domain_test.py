from __future__ import annotations

import unittest

import pandas as pd

from app.validators.domain import (
    validate_ds,
    validate_eg,
    validate_ex,
    validate_irt_consistency,
    validate_lb,
    validate_pr,
    validate_qs,
    validate_rs,
)


def _rule_ids(findings: pd.DataFrame) -> set[str]:
    return set(findings["rule_id"]) if len(findings) else set()


class TestValidateEx(unittest.TestCase):

    def test_missing_required_columns_returns_crit(self):
        df = pd.DataFrame({"USUBJID": ["S1"]})
        findings = validate_ex(df)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings.iloc[0]["severity"], "CRIT")
        self.assertEqual(findings.iloc[0]["rule_id"], "SDTM_EX_000")

    def test_valid_rows_produce_no_findings(self):
        df = pd.DataFrame({
            "USUBJID": ["S1", "S2"],
            "EXTRT":   ["STUDY DRUG A", "PLACEBO"],
            "EXDOSE":  [100, 0],
            "EXDOSU":  ["mg", "mg"],
            "EXROUTE": ["ORAL", "ORAL"],
            "EXSTDTC": ["2024-01-01", "2024-01-01"],
            "EXENDTC": ["2024-02-01", "2024-02-01"],
        })
        findings = validate_ex(df)
        self.assertEqual(len(findings), 0)

    def test_missing_usubjid_flagged(self):
        df = pd.DataFrame({
            "USUBJID": ["", "S2"],
            "EXTRT":   ["STUDY DRUG A", "STUDY DRUG A"],
            "EXSTDTC": ["2024-01-01", "2024-01-01"],
        })
        findings = validate_ex(df)
        self.assertIn("SDTM_EX_001", _rule_ids(findings))

    def test_non_numeric_dose_flagged(self):
        df = pd.DataFrame({
            "USUBJID": ["S1"],
            "EXTRT":   ["STUDY DRUG A"],
            "EXSTDTC": ["2024-01-01"],
            "EXDOSE":  ["not-a-number"],
        })
        findings = validate_ex(df)
        self.assertIn("SDTM_EX_003", _rule_ids(findings))

    def test_negative_dose_flagged(self):
        df = pd.DataFrame({
            "USUBJID": ["S1"],
            "EXTRT":   ["STUDY DRUG A"],
            "EXSTDTC": ["2024-01-01"],
            "EXDOSE":  [-5],
        })
        findings = validate_ex(df)
        self.assertIn("SDTM_EX_004", _rule_ids(findings))

    def test_start_after_end_flagged(self):
        df = pd.DataFrame({
            "USUBJID": ["S1"],
            "EXTRT":   ["STUDY DRUG A"],
            "EXSTDTC": ["2024-05-01"],
            "EXENDTC": ["2024-01-01"],
        })
        findings = validate_ex(df)
        self.assertIn("SDTM_EX_007", _rule_ids(findings))

    def test_invalid_route_flagged(self):
        df = pd.DataFrame({
            "USUBJID": ["S1"],
            "EXTRT":   ["STUDY DRUG A"],
            "EXSTDTC": ["2024-01-01"],
            "EXROUTE": ["TELEPATHIC"],
        })
        findings = validate_ex(df)
        self.assertIn("SDTM_EX_008", _rule_ids(findings))

    def test_invalid_dosu_flagged(self):
        df = pd.DataFrame({
            "USUBJID": ["S1"],
            "EXTRT":   ["STUDY DRUG A"],
            "EXSTDTC": ["2024-01-01"],
            "EXDOSU":  ["MG"],
        })
        findings = validate_ex(df)
        self.assertIn("SDTM_EX_009", _rule_ids(findings))


class TestValidateLb(unittest.TestCase):

    def test_missing_required_columns_returns_crit(self):
        df = pd.DataFrame({"USUBJID": ["S1"]})
        findings = validate_lb(df)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings.iloc[0]["severity"], "CRIT")
        self.assertEqual(findings.iloc[0]["rule_id"], "SDTM_LB_000")

    def test_valid_rows_produce_no_findings(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1", "S2"],
            "LBTESTCD": ["ALT", "CREAT"],
            "LBORRES":  ["22", "0.9"],
            "LBSTRESN": [22, 0.9],
            "LBNRIND":  ["NORMAL", "NORMAL"],
            "LBDTC":    ["2024-01-01", "2024-01-01"],
        })
        findings = validate_lb(df)
        self.assertEqual(len(findings), 0)

    def test_missing_lborres_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "LBTESTCD": ["ALT"],
            "LBORRES":  [""],
            "LBDTC":    ["2024-01-01"],
        })
        findings = validate_lb(df)
        self.assertIn("SDTM_LB_002", _rule_ids(findings))

    def test_unknown_testcd_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "LBTESTCD": ["GLUCOSE"],
            "LBORRES":  ["110"],
            "LBDTC":    ["2024-01-01"],
        })
        findings = validate_lb(df)
        self.assertIn("SDTM_LB_003", _rule_ids(findings))

    def test_bad_date_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "LBTESTCD": ["ALT"],
            "LBORRES":  ["22"],
            "LBDTC":    ["05/05/2024"],
        })
        findings = validate_lb(df)
        self.assertIn("SDTM_LB_004", _rule_ids(findings))

    def test_non_numeric_stresn_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "LBTESTCD": ["ALT"],
            "LBORRES":  ["ND"],
            "LBSTRESN": ["ND"],
            "LBDTC":    ["2024-01-01"],
        })
        findings = validate_lb(df)
        self.assertIn("SDTM_LB_005", _rule_ids(findings))

    def test_invalid_nrind_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "LBTESTCD": ["ALT"],
            "LBORRES":  ["22"],
            "LBNRIND":  ["ELEVATED"],
            "LBDTC":    ["2024-01-01"],
        })
        findings = validate_lb(df)
        self.assertIn("SDTM_LB_006", _rule_ids(findings))


class TestValidateDs(unittest.TestCase):

    def test_missing_required_columns_returns_crit(self):
        df = pd.DataFrame({"USUBJID": ["S1"]})
        findings = validate_ds(df)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings.iloc[0]["severity"], "CRIT")
        self.assertEqual(findings.iloc[0]["rule_id"], "SDTM_DS_000")

    def test_valid_rows_produce_no_findings(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "DSTERM":   ["Completed Study"],
            "DSDECOD":  ["COMPLETED"],
            "DSCAT":    ["DISPOSITION EVENT"],
            "DSSTDTC":  ["2024-01-01"],
        })
        findings = validate_ds(df)
        self.assertEqual(len(findings), 0)

    def test_missing_dsdecod_flagged(self):
        df = pd.DataFrame({
            "USUBJID": ["S1"],
            "DSDECOD": [""],
            "DSSTDTC": ["2024-01-01"],
        })
        findings = validate_ds(df)
        self.assertIn("SDTM_DS_002", _rule_ids(findings))

    def test_invalid_dsdecod_flagged(self):
        df = pd.DataFrame({
            "USUBJID": ["S1"],
            "DSDECOD": ["SUBJECT DECISION"],
            "DSSTDTC": ["2024-01-01"],
        })
        findings = validate_ds(df)
        self.assertIn("SDTM_DS_003", _rule_ids(findings))

    def test_bad_date_flagged(self):
        df = pd.DataFrame({
            "USUBJID": ["S1"],
            "DSDECOD": ["COMPLETED"],
            "DSSTDTC": ["not-a-date"],
        })
        findings = validate_ds(df)
        self.assertIn("SDTM_DS_004", _rule_ids(findings))

    def test_missing_dsterm_flagged(self):
        df = pd.DataFrame({
            "USUBJID": ["S1"],
            "DSTERM":  [""],
            "DSDECOD": ["COMPLETED"],
            "DSSTDTC": ["2024-01-01"],
        })
        findings = validate_ds(df)
        self.assertIn("SDTM_DS_005", _rule_ids(findings))

    def test_invalid_dscat_flagged(self):
        df = pd.DataFrame({
            "USUBJID": ["S1"],
            "DSDECOD": ["COMPLETED"],
            "DSCAT":   ["ENROLLMENT"],
            "DSSTDTC": ["2024-01-01"],
        })
        findings = validate_ds(df)
        self.assertIn("SDTM_DS_006", _rule_ids(findings))


class TestValidateEg(unittest.TestCase):

    def test_missing_required_columns_returns_crit(self):
        df = pd.DataFrame({"USUBJID": ["S1"]})
        findings = validate_eg(df)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings.iloc[0]["severity"], "CRIT")
        self.assertEqual(findings.iloc[0]["rule_id"], "SDTM_EG_000")

    def test_valid_rows_produce_no_findings(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1", "S2"],
            "EGTESTCD": ["HR", "QT"],
            "EGORRES":  ["72", "398"],
            "EGORRESU": ["beats/min", "msec"],
            "EGDTC":    ["2024-01-01", "2024-01-01"],
        })
        findings = validate_eg(df)
        self.assertEqual(len(findings), 0)

    def test_unknown_testcd_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "EGTESTCD": ["PACE"],
            "EGORRES":  ["Y"],
            "EGDTC":    ["2024-01-01"],
        })
        findings = validate_eg(df)
        self.assertIn("SDTM_EG_002", _rule_ids(findings))

    def test_bad_date_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "EGTESTCD": ["HR"],
            "EGORRES":  ["72"],
            "EGDTC":    ["05-23-2024"],
        })
        findings = validate_eg(df)
        self.assertIn("SDTM_EG_003", _rule_ids(findings))

    def test_non_numeric_result_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "EGTESTCD": ["QT"],
            "EGORRES":  ["ABNORMAL"],
            "EGDTC":    ["2024-01-01"],
        })
        findings = validate_eg(df)
        self.assertIn("SDTM_EG_004", _rule_ids(findings))

    def test_unit_mismatch_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "EGTESTCD": ["QT"],
            "EGORRES":  ["392"],
            "EGORRESU": ["sec"],
            "EGDTC":    ["2024-01-01"],
        })
        findings = validate_eg(df)
        self.assertIn("SDTM_EG_005", _rule_ids(findings))

    def test_missing_egorresu_column_reports_dataset_level_finding(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "EGTESTCD": ["HR"],
            "EGORRES":  ["72"],
            "EGDTC":    ["2024-01-01"],
        })
        findings = validate_eg(df)
        row = findings[findings["rule_id"] == "SDTM_EG_005"].iloc[0]
        self.assertEqual(row["row_index"], -1)
        self.assertEqual(row["severity"], "LOW")


class TestValidateQs(unittest.TestCase):

    def test_missing_required_columns_returns_crit(self):
        df = pd.DataFrame({"USUBJID": ["S1"]})
        findings = validate_qs(df)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings.iloc[0]["severity"], "CRIT")
        self.assertEqual(findings.iloc[0]["rule_id"], "SDTM_QS_000")

    def test_valid_rows_produce_no_findings(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1", "S2"],
            "QSCAT":    ["EQ-5D-5L", "EQ-5D-5L"],
            "QSTESTCD": ["MOBILITY", "PAIN"],
            "QSORRES":  ["2", "1"],
            "QSSTRESN": [2, 1],
            "QSDTC":    ["2024-01-01", "2024-01-01"],
        })
        findings = validate_qs(df)
        self.assertEqual(len(findings), 0)

    def test_missing_qscat_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "QSCAT":    [""],
            "QSTESTCD": ["MOBILITY"],
            "QSORRES":  ["2"],
            "QSDTC":    ["2024-01-01"],
        })
        findings = validate_qs(df)
        self.assertIn("SDTM_QS_002", _rule_ids(findings))

    def test_missing_qstestcd_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "QSCAT":    ["EQ-5D-5L"],
            "QSTESTCD": [""],
            "QSORRES":  ["2"],
            "QSDTC":    ["2024-01-01"],
        })
        findings = validate_qs(df)
        self.assertIn("SDTM_QS_003", _rule_ids(findings))

    def test_bad_date_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "QSCAT":    ["EQ-5D-5L"],
            "QSTESTCD": ["MOBILITY"],
            "QSORRES":  ["2"],
            "QSDTC":    ["04/25/2024"],
        })
        findings = validate_qs(df)
        self.assertIn("SDTM_QS_004", _rule_ids(findings))

    def test_non_numeric_stresn_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "QSCAT":    ["EQ-5D-5L"],
            "QSTESTCD": ["MOBILITY"],
            "QSORRES":  ["ND"],
            "QSSTRESN": ["ND"],
            "QSDTC":    ["2024-01-01"],
        })
        findings = validate_qs(df)
        self.assertIn("SDTM_QS_005", _rule_ids(findings))

    def test_invalid_qsstat_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "QSCAT":    ["EQ-5D-5L"],
            "QSTESTCD": ["MOBILITY"],
            "QSORRES":  ["2"],
            "QSSTAT":   ["MAYBE"],
            "QSDTC":    ["2024-01-01"],
        })
        findings = validate_qs(df)
        self.assertIn("SDTM_QS_006", _rule_ids(findings))

    def test_missing_qsorres_without_not_done_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "QSCAT":    ["EQ-5D-5L"],
            "QSTESTCD": ["MOBILITY"],
            "QSORRES":  [None],
            "QSDTC":    ["2024-01-01"],
        })
        findings = validate_qs(df)
        self.assertIn("SDTM_QS_007", _rule_ids(findings))

    def test_missing_qsorres_exempt_when_not_done(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "QSCAT":    ["EQ-5D-5L"],
            "QSTESTCD": ["MOBILITY"],
            "QSORRES":  [None],
            "QSSTAT":   ["NOT DONE"],
            "QSREASND": ["Subject declined"],
            "QSDTC":    ["2024-01-01"],
        })
        findings = validate_qs(df)
        self.assertNotIn("SDTM_QS_007", _rule_ids(findings))

    def test_missing_qsreasnd_when_not_done_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "QSCAT":    ["EQ-5D-5L"],
            "QSTESTCD": ["MOBILITY"],
            "QSORRES":  [None],
            "QSSTAT":   ["NOT DONE"],
            "QSREASND": [""],
            "QSDTC":    ["2024-01-01"],
        })
        findings = validate_qs(df)
        self.assertIn("SDTM_QS_008", _rule_ids(findings))

    def test_missing_qsreasnd_column_when_not_done_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "QSCAT":    ["EQ-5D-5L"],
            "QSTESTCD": ["MOBILITY"],
            "QSORRES":  [None],
            "QSSTAT":   ["NOT DONE"],
            "QSDTC":    ["2024-01-01"],
        })
        findings = validate_qs(df)
        self.assertIn("SDTM_QS_008", _rule_ids(findings))


class TestValidateRs(unittest.TestCase):

    def test_missing_required_columns_returns_crit(self):
        df = pd.DataFrame({"USUBJID": ["S1"]})
        findings = validate_rs(df)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings.iloc[0]["severity"], "CRIT")
        self.assertEqual(findings.iloc[0]["rule_id"], "SDTM_RS_000")

    def test_valid_rows_produce_no_findings(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1", "S2"],
            "RSTESTCD": ["OVRLRESP", "OVRLRESP"],
            "RSORRES":  ["PR", "SD"],
            "RSEVAL":   ["INVESTIGATOR", "INDEPENDENT ASSESSOR"],
            "RSDTC":    ["2024-01-01", "2024-01-01"],
        })
        findings = validate_rs(df)
        self.assertEqual(len(findings), 0)

    def test_missing_rstestcd_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "RSTESTCD": [""],
            "RSORRES":  ["PR"],
            "RSDTC":    ["2024-01-01"],
        })
        findings = validate_rs(df)
        self.assertIn("SDTM_RS_002", _rule_ids(findings))

    def test_invalid_rsorres_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "RSTESTCD": ["OVRLRESP"],
            "RSORRES":  ["MIXED"],
            "RSDTC":    ["2024-01-01"],
        })
        findings = validate_rs(df)
        self.assertIn("SDTM_RS_003", _rule_ids(findings))

    def test_bad_date_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "RSTESTCD": ["OVRLRESP"],
            "RSORRES":  ["PR"],
            "RSDTC":    ["15/13/2024"],
        })
        findings = validate_rs(df)
        self.assertIn("SDTM_RS_004", _rule_ids(findings))

    def test_invalid_rseval_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "RSTESTCD": ["OVRLRESP"],
            "RSORRES":  ["PR"],
            "RSEVAL":   ["SITE RADIOLOGIST"],
            "RSDTC":    ["2024-01-01"],
        })
        findings = validate_rs(df)
        self.assertIn("SDTM_RS_005", _rule_ids(findings))

    def test_invalid_rsstat_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "RSTESTCD": ["OVRLRESP"],
            "RSORRES":  ["PR"],
            "RSSTAT":   ["MAYBE"],
            "RSDTC":    ["2024-01-01"],
        })
        findings = validate_rs(df)
        self.assertIn("SDTM_RS_006", _rule_ids(findings))

    def test_missing_rsorres_without_not_done_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "RSTESTCD": ["OVRLRESP"],
            "RSORRES":  [None],
            "RSDTC":    ["2024-01-01"],
        })
        findings = validate_rs(df)
        self.assertIn("SDTM_RS_007", _rule_ids(findings))

    def test_missing_rsorres_exempt_when_not_done(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "RSTESTCD": ["OVRLRESP"],
            "RSORRES":  [None],
            "RSSTAT":   ["NOT DONE"],
            "RSREASND": ["Scan not performed"],
            "RSDTC":    ["2024-01-01"],
        })
        findings = validate_rs(df)
        self.assertNotIn("SDTM_RS_007", _rule_ids(findings))

    def test_missing_rsreasnd_when_not_done_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "RSTESTCD": ["OVRLRESP"],
            "RSORRES":  [None],
            "RSSTAT":   ["NOT DONE"],
            "RSREASND": [""],
            "RSDTC":    ["2024-01-01"],
        })
        findings = validate_rs(df)
        self.assertIn("SDTM_RS_008", _rule_ids(findings))

    def test_missing_rsreasnd_column_when_not_done_flagged(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1"],
            "RSTESTCD": ["OVRLRESP"],
            "RSORRES":  [None],
            "RSSTAT":   ["NOT DONE"],
            "RSDTC":    ["2024-01-01"],
        })
        findings = validate_rs(df)
        self.assertIn("SDTM_RS_008", _rule_ids(findings))


class TestValidatePr(unittest.TestCase):

    def test_missing_required_columns_returns_crit(self):
        df = pd.DataFrame({"USUBJID": ["S1"]})
        findings = validate_pr(df)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings.iloc[0]["severity"], "CRIT")
        self.assertEqual(findings.iloc[0]["rule_id"], "SDTM_PR_000")

    def test_valid_rows_produce_no_findings(self):
        df = pd.DataFrame({
            "USUBJID":  ["S1", "S2"],
            "PRTRT":    ["Biopsy", "CT Scan"],
            "PRCAT":    ["DIAGNOSTIC", "DIAGNOSTIC"],
            "PRSTDTC":  ["2024-01-01", "2024-01-01"],
            "PRENDTC":  ["2024-01-01", "2024-01-01"],
        })
        findings = validate_pr(df)
        self.assertEqual(len(findings), 0)

    def test_missing_prtrt_flagged(self):
        df = pd.DataFrame({
            "USUBJID": ["S1"],
            "PRTRT":   [""],
            "PRSTDTC": ["2024-01-01"],
        })
        findings = validate_pr(df)
        self.assertIn("SDTM_PR_002", _rule_ids(findings))

    def test_bad_start_date_flagged(self):
        df = pd.DataFrame({
            "USUBJID": ["S1"],
            "PRTRT":   ["Biopsy"],
            "PRSTDTC": ["not-a-date"],
        })
        findings = validate_pr(df)
        self.assertIn("SDTM_PR_003", _rule_ids(findings))

    def test_bad_end_date_flagged(self):
        df = pd.DataFrame({
            "USUBJID": ["S1"],
            "PRTRT":   ["Biopsy"],
            "PRSTDTC": ["2024-01-01"],
            "PRENDTC": ["not-a-date"],
        })
        findings = validate_pr(df)
        self.assertIn("SDTM_PR_004", _rule_ids(findings))

    def test_start_after_end_flagged(self):
        df = pd.DataFrame({
            "USUBJID": ["S1"],
            "PRTRT":   ["Biopsy"],
            "PRSTDTC": ["2024-05-01"],
            "PRENDTC": ["2024-01-01"],
        })
        findings = validate_pr(df)
        self.assertIn("SDTM_PR_005", _rule_ids(findings))

    def test_invalid_prcat_flagged(self):
        df = pd.DataFrame({
            "USUBJID": ["S1"],
            "PRTRT":   ["Physical Therapy"],
            "PRCAT":   ["REHABILITATION"],
            "PRSTDTC": ["2024-01-01"],
        })
        findings = validate_pr(df)
        self.assertIn("SDTM_PR_006", _rule_ids(findings))


class TestValidateIrtConsistency(unittest.TestCase):

    def test_missing_required_columns_returns_crit(self):
        findings = validate_irt_consistency(
            pd.DataFrame({"USUBJID": ["S1"]}), pd.DataFrame({"USUBJID": ["S1"]}), pd.DataFrame()
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings.iloc[0]["severity"], "CRIT")
        self.assertEqual(findings.iloc[0]["rule_id"], "X_IRT_000")

    def test_clean_data_produces_no_findings(self):
        dm = pd.DataFrame({"USUBJID": ["S1"], "RFSTDTC": ["2024-01-01"], "RFENDTC": ["2024-06-01"]})
        ex = pd.DataFrame({"USUBJID": ["S1"], "EXSTDTC": ["2024-01-05"], "EXENDTC": ["2024-05-01"]})
        ds = pd.DataFrame({"USUBJID": ["S1"], "DSSTDTC": ["2024-06-01"]})
        findings = validate_irt_consistency(dm, ex, ds)
        self.assertEqual(len(findings), 0)

    def test_dosing_before_study_entry_flagged(self):
        dm = pd.DataFrame({"USUBJID": ["S1"], "RFSTDTC": ["2024-02-01"]})
        ex = pd.DataFrame({"USUBJID": ["S1"], "EXSTDTC": ["2024-01-20"]})
        findings = validate_irt_consistency(dm, ex, pd.DataFrame())
        self.assertIn("X_IRT_001", _rule_ids(findings))

    def test_dosing_after_rfendtc_flagged(self):
        dm = pd.DataFrame({"USUBJID": ["S1"], "RFSTDTC": ["2024-01-01"], "RFENDTC": ["2024-06-01"]})
        ex = pd.DataFrame({"USUBJID": ["S1"], "EXSTDTC": ["2024-01-05"], "EXENDTC": ["2024-07-01"]})
        findings = validate_irt_consistency(dm, ex, pd.DataFrame())
        self.assertIn("X_IRT_002", _rule_ids(findings))

    def test_dosing_after_ds_disposition_flagged(self):
        dm = pd.DataFrame({"USUBJID": ["S1"], "RFSTDTC": ["2024-01-01"]})
        ex = pd.DataFrame({"USUBJID": ["S1"], "EXSTDTC": ["2024-01-05"], "EXENDTC": ["2024-06-01"]})
        ds = pd.DataFrame({
            "USUBJID": ["S1"], "DSSTDTC": ["2024-03-01"], "DSCAT": ["DISPOSITION EVENT"],
        })
        findings = validate_irt_consistency(dm, ex, ds)
        self.assertIn("X_IRT_003", _rule_ids(findings))

    def test_uses_exstdtc_when_exendtc_missing(self):
        dm = pd.DataFrame({"USUBJID": ["S1"], "RFSTDTC": ["2024-01-01"]})
        ex = pd.DataFrame({"USUBJID": ["S1"], "EXSTDTC": ["2024-04-01"]})
        ds = pd.DataFrame({"USUBJID": ["S1"], "DSSTDTC": ["2024-03-01"]})
        findings = validate_irt_consistency(dm, ex, ds)
        self.assertIn("X_IRT_003", _rule_ids(findings))

    def test_protocol_milestone_ds_rows_excluded_when_disposition_event_present(self):
        dm = pd.DataFrame({"USUBJID": ["S1"], "RFSTDTC": ["2024-01-01"]})
        ex = pd.DataFrame({"USUBJID": ["S1"], "EXSTDTC": ["2024-01-05"], "EXENDTC": ["2024-05-01"]})
        ds = pd.DataFrame({
            "USUBJID":  ["S1", "S1"],
            "DSSTDTC":  ["2024-01-02", "2024-06-01"],
            "DSCAT":    ["PROTOCOL MILESTONE", "DISPOSITION EVENT"],
        })
        findings = validate_irt_consistency(dm, ex, ds)
        # the early PROTOCOL MILESTONE row (informed consent) must not be used as the exit date
        self.assertNotIn("X_IRT_003", _rule_ids(findings))

    def test_no_ds_data_skips_irt_003(self):
        dm = pd.DataFrame({"USUBJID": ["S1"], "RFSTDTC": ["2024-01-01"]})
        ex = pd.DataFrame({"USUBJID": ["S1"], "EXSTDTC": ["2024-01-05"]})
        findings = validate_irt_consistency(dm, ex, pd.DataFrame())
        self.assertEqual(len(findings), 0)


if __name__ == "__main__":
    unittest.main()
