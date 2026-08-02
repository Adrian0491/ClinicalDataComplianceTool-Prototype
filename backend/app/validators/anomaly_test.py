from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from app.validators.anomaly import (
    NUMERIC_COLS,
    RULES,
    apply_rules,
    build_frame_from_domains,
    detect_anomalies,
    load_generic,
    to_findings,
)

# mock_data/ lives at the repo root, not relative to CWD — this file moved
# from bk/validator/ (2 levels under repo root) to backend/app/validators/
# (3 levels under), and the intended test command runs from backend/, where
# a bare "mock_data/..." string never resolves.
_REPO_ROOT = Path(__file__).parents[3]
_MOCK_DATA = _REPO_ROOT / "mock_data"


class TestLoadGeneric(unittest.TestCase):

    def setUp(self):
        if not _MOCK_DATA.is_dir():
            self.skipTest(f"mock_data/ not found at {_MOCK_DATA}")

    def test_missing_required_columns_raises(self):
        with self.assertRaises(ValueError):
            load_generic(str(_MOCK_DATA / "dm.csv"))  # DM has no AGE/SYSBP/DOSE/VSDTC set

    def test_loads_generic_csv(self):
        df = load_generic(str(_MOCK_DATA / "mock_data.csv"))
        self.assertGreater(len(df), 0)
        for col in ["AGE", "SYSBP", "DOSE", "VSDTC"]:
            self.assertIn(col, df.columns)


class TestApplyRules(unittest.TestCase):

    def test_flags_out_of_range_age(self):
        df = pd.DataFrame({"AGE": [17, 18, 120, 121], "SYSBP": [None]*4, "DOSE": [None]*4})
        out = apply_rules(df)
        self.assertListEqual(list(out["AGE_valid"]), [0, 1, 1, 0])

    def test_flags_out_of_range_sysbp(self):
        df = pd.DataFrame({"AGE": [None]*4, "SYSBP": [89, 90, 180, 181], "DOSE": [None]*4})
        out = apply_rules(df)
        self.assertListEqual(list(out["SYSBP_valid"]), [0, 1, 1, 0])

    def test_flags_non_positive_dose(self):
        df = pd.DataFrame({"AGE": [None]*3, "SYSBP": [None]*3, "DOSE": [0, -1, 50]})
        out = apply_rules(df)
        self.assertListEqual(list(out["DOSE_valid"]), [0, 0, 1])

    def test_nan_is_invalid(self):
        df = pd.DataFrame({"AGE": [None], "SYSBP": [None], "DOSE": [None]})
        out = apply_rules(df)
        self.assertEqual(out["DOSE_valid"].iloc[0], 0)

    def test_missing_column_is_skipped_without_error(self):
        df = pd.DataFrame({"SYSBP": [120]})
        out = apply_rules(df)
        self.assertNotIn("AGE_valid", out.columns)
        self.assertIn("SYSBP_valid", out.columns)

    def test_date_valid_column(self):
        df = pd.DataFrame({"VSDTC": ["2024-01-01", None]})
        out = apply_rules(df)
        self.assertListEqual(list(out["date_valid"]), [1, 0])


class TestDetectAnomalies(unittest.TestCase):

    def _clean_frame(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "AGE":   [40 + (i % 5) for i in range(n)],
            "SYSBP": [120 + (i % 4) for i in range(n)],
            "DOSE":  [100.0 for _ in range(n)],
        })

    def test_below_row_floor_returns_all_zero(self):
        df = apply_rules(self._clean_frame(9))
        out = detect_anomalies(df)
        self.assertTrue((out["anomaly"] == 0).all())

    def test_at_row_floor_runs_the_model(self):
        df = apply_rules(self._clean_frame(10))
        out = detect_anomalies(df)
        self.assertIn("anomaly", out.columns)
        self.assertEqual(len(out), 10)

    def test_no_numeric_columns_returns_all_zero(self):
        df = pd.DataFrame({"CONDITION": ["A"] * 15})
        out = detect_anomalies(df)
        self.assertTrue((out["anomaly"] == 0).all())

    def test_flags_extreme_outlier(self):
        df = self._clean_frame(20)
        df.loc[0, "AGE"] = 400
        df.loc[0, "SYSBP"] = 900
        df.loc[0, "DOSE"] = -500
        df = apply_rules(df)
        out = detect_anomalies(df)
        self.assertEqual(out.loc[0, "anomaly"], 1)

    def test_deterministic_across_runs(self):
        df = self._clean_frame(30)
        df.loc[5, "AGE"] = 500
        df = apply_rules(df)
        out1 = detect_anomalies(df.copy())
        out2 = detect_anomalies(df.copy())
        self.assertListEqual(list(out1["anomaly"]), list(out2["anomaly"]))

    def test_missing_values_are_imputed_not_dropped(self):
        df = self._clean_frame(15)
        df.loc[3, "AGE"] = None
        df = apply_rules(df)
        out = detect_anomalies(df)
        self.assertEqual(len(out), 15)


class TestToFindings(unittest.TestCase):

    def test_no_flags_returns_empty(self):
        df = pd.DataFrame({"AGE_valid": [1, 1], "anomaly": [0, 0]})
        findings = to_findings(df)
        self.assertEqual(len(findings), 0)

    def test_rule_violation_carries_usubjid_and_evidence(self):
        df = pd.DataFrame({
            "USUBJID":   ["S1", "S2"],
            "AGE":       [10, 40],
            "AGE_valid": [0, 1],
        })
        findings = to_findings(df)
        self.assertEqual(len(findings), 1)
        row = findings.iloc[0]
        self.assertEqual(row["usubjid"], "S1")
        self.assertEqual(row["evidence"], "10")
        self.assertEqual(row["finding_type"], "SDTM_RULE")

    def test_anomaly_carries_usubjid_and_feature_evidence(self):
        df = pd.DataFrame({
            "USUBJID": ["S1"],
            "AGE_num": [400],
            "SYSBP_num": [900],
            "DOSE_num": [-500],
            "anomaly": [1],
        })
        findings = to_findings(df)
        self.assertEqual(len(findings), 1)
        row = findings.iloc[0]
        self.assertEqual(row["finding_type"], "ANOMALY")
        self.assertEqual(row["usubjid"], "S1")
        self.assertEqual(row["evidence"], "")  # NUMERIC_COLS names (AGE/SYSBP/DOSE) not present, only *_num

    def test_anomaly_evidence_uses_original_feature_columns(self):
        df = pd.DataFrame({
            "USUBJID": ["S1"],
            "AGE":     [400],
            "SYSBP":   [900],
            "DOSE":    [-500],
            "anomaly": [1],
        })
        findings = to_findings(df)
        row = findings.iloc[0]
        self.assertEqual(row["evidence"], "AGE=400, SYSBP=900, DOSE=-500")

    def test_missing_usubjid_column_defaults_to_empty(self):
        df = pd.DataFrame({"AGE": [10], "AGE_valid": [0]})
        findings = to_findings(df)
        self.assertEqual(findings.iloc[0]["usubjid"], "")


class TestBuildFrameFromDomains(unittest.TestCase):

    def test_empty_dm_returns_empty(self):
        out = build_frame_from_domains(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        self.assertEqual(len(out), 0)

    def test_age_from_dm_only(self):
        dm = pd.DataFrame({"USUBJID": ["S1", "S2"], "AGE": [30, 40]})
        out = build_frame_from_domains(dm, pd.DataFrame(), pd.DataFrame())
        self.assertListEqual(list(out["USUBJID"]), ["S1", "S2"])
        self.assertListEqual(list(out["AGE"]), [30, 40])
        self.assertTrue(out["SYSBP"].isna().all())
        self.assertTrue(out["DOSE"].isna().all())

    def test_sysbp_averaged_per_subject_from_vs(self):
        dm = pd.DataFrame({"USUBJID": ["S1"], "AGE": [30]})
        vs = pd.DataFrame({
            "USUBJID":  ["S1", "S1", "S1"],
            "VSTESTCD": ["SYSBP", "SYSBP", "DIABP"],
            "VSORRES":  [120, 130, 80],
            "VSDTC":    ["2024-01-01", "2024-02-01", "2024-01-01"],
        })
        out = build_frame_from_domains(dm, vs, pd.DataFrame())
        self.assertAlmostEqual(out.loc[0, "SYSBP"], 125.0)
        self.assertEqual(out.loc[0, "VSDTC"], "2024-01-01")

    def test_dose_averaged_per_subject_from_ex(self):
        dm = pd.DataFrame({"USUBJID": ["S1"], "AGE": [30]})
        ex = pd.DataFrame({"USUBJID": ["S1", "S1"], "EXDOSE": [100, 200]})
        out = build_frame_from_domains(dm, pd.DataFrame(), ex)
        self.assertAlmostEqual(out.loc[0, "DOSE"], 150.0)

    def test_subject_with_no_vs_or_ex_data_gets_nan(self):
        dm = pd.DataFrame({"USUBJID": ["S1", "S2"], "AGE": [30, 40]})
        vs = pd.DataFrame({
            "USUBJID": ["S1"], "VSTESTCD": ["SYSBP"], "VSORRES": [120], "VSDTC": ["2024-01-01"],
        })
        out = build_frame_from_domains(dm, vs, pd.DataFrame())
        self.assertTrue(pd.isna(out.loc[1, "SYSBP"]))

    def test_alt_averaged_per_subject_from_lb(self):
        dm = pd.DataFrame({"USUBJID": ["S1"], "AGE": [30]})
        lb = pd.DataFrame({
            "USUBJID":  ["S1", "S1", "S1"],
            "LBTESTCD": ["ALT", "ALT", "CREAT"],
            "LBSTRESN": [20, 30, 1.0],
        })
        out = build_frame_from_domains(dm, pd.DataFrame(), pd.DataFrame(), lb=lb)
        self.assertAlmostEqual(out.loc[0, "ALT"], 25.0)

    def test_alt_falls_back_to_lborres_when_lbstresn_absent(self):
        dm = pd.DataFrame({"USUBJID": ["S1"], "AGE": [30]})
        lb = pd.DataFrame({
            "USUBJID": ["S1"], "LBTESTCD": ["ALT"], "LBORRES": ["42"],
        })
        out = build_frame_from_domains(dm, pd.DataFrame(), pd.DataFrame(), lb=lb)
        self.assertAlmostEqual(out.loc[0, "ALT"], 42.0)

    def test_subject_with_no_alt_test_gets_nan_not_zero(self):
        dm = pd.DataFrame({"USUBJID": ["S1"], "AGE": [30]})
        lb = pd.DataFrame({"USUBJID": ["S1"], "LBTESTCD": ["CREAT"], "LBSTRESN": [1.0]})
        out = build_frame_from_domains(dm, pd.DataFrame(), pd.DataFrame(), lb=lb)
        self.assertTrue(pd.isna(out.loc[0, "ALT"]))

    def test_qtcf_prefers_qtcf_and_qt_testcds_from_eg(self):
        dm = pd.DataFrame({"USUBJID": ["S1"], "AGE": [30]})
        eg = pd.DataFrame({
            "USUBJID":  ["S1", "S1", "S1"],
            "EGTESTCD": ["QTCF", "QT", "HR"],
            "EGORRES":  ["410", "400", "72"],
        })
        out = build_frame_from_domains(dm, pd.DataFrame(), pd.DataFrame(), eg=eg)
        self.assertAlmostEqual(out.loc[0, "QTCF"], 405.0)

    def test_qs_score_averaged_per_subject(self):
        dm = pd.DataFrame({"USUBJID": ["S1"], "AGE": [30]})
        qs = pd.DataFrame({
            "USUBJID": ["S1", "S1"], "QSSTRESN": [1, 3],
        })
        out = build_frame_from_domains(dm, pd.DataFrame(), pd.DataFrame(), qs=qs)
        self.assertAlmostEqual(out.loc[0, "QS_SCORE"], 2.0)

    def test_pr_count_per_subject(self):
        dm = pd.DataFrame({"USUBJID": ["S1", "S2"], "AGE": [30, 40]})
        pr = pd.DataFrame({"USUBJID": ["S1", "S1", "S1"]})
        out = build_frame_from_domains(dm, pd.DataFrame(), pd.DataFrame(), pr=pr)
        self.assertEqual(out.loc[0, "PR_COUNT"], 3)

    def test_pr_count_is_zero_not_nan_when_subject_has_no_procedures(self):
        dm = pd.DataFrame({"USUBJID": ["S1", "S2"], "AGE": [30, 40]})
        pr = pd.DataFrame({"USUBJID": ["S1"]})
        out = build_frame_from_domains(dm, pd.DataFrame(), pd.DataFrame(), pr=pr)
        self.assertEqual(out.loc[1, "PR_COUNT"], 0)

    def test_pr_count_absent_entirely_when_no_pr_domain_supplied(self):
        dm = pd.DataFrame({"USUBJID": ["S1"], "AGE": [30]})
        out = build_frame_from_domains(dm, pd.DataFrame(), pd.DataFrame())
        self.assertNotIn("PR_COUNT", out.columns)


class TestCalibrationRegressions(unittest.TestCase):
    """
    Guards a real calibration bug found while extending anomaly detection to
    LB/EG: adding ALT/QTCF to RULES flagged nearly every subject as invalid,
    because a missing value there just means "not tested this visit", not a
    data error — unlike AGE/SYSBP/DOSE, which are always-expected attributes
    in the flat generic CSV.
    """

    def test_alt_and_qtcf_have_no_rules_threshold(self):
        self.assertNotIn("ALT", RULES)
        self.assertNotIn("QTCF", RULES)

    def test_untested_alt_does_not_produce_a_rule_violation(self):
        # Multiple subjects, only one of whom was ever tested for ALT —
        # the rest must not be flagged just for lacking that lab result.
        df = pd.DataFrame({
            "USUBJID": [f"S{i}" for i in range(12)],
            "ALT":     [25.0] + [None] * 11,
        })
        out = apply_rules(df)
        self.assertNotIn("ALT_valid", out.columns)
        findings = to_findings(out)
        self.assertEqual(len(findings), 0)

    def test_alt_and_qtcf_still_feed_anomaly_detection(self):
        self.assertIn("ALT", NUMERIC_COLS)
        self.assertIn("QTCF", NUMERIC_COLS)


if __name__ == "__main__":
    unittest.main()
