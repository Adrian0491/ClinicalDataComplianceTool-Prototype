import unittest
import polars as pl

from edc_validator.sdtm_rules import validate_dm, _mk_findings


class TestSdtmRules(unittest.TestCase):
    FINDINGS_COLS = [
        "finding_type",
        "rule_id",
        "severity",
        "field",
        "message",
        "row_index",
        "evidence",
    ]

    def assert_findings_schema(self, df: pl.DataFrame) -> None:
        """Validate that findings DF has the expected columns (order-agnostic)."""
        self.assertIsInstance(df, pl.DataFrame)
        for c in self.FINDINGS_COLS:
            self.assertIn(c, df.columns, f"Missing findings column: {c}")

    def test_mk_findings_returns_empty_schema_when_no_violations(self):
        df = pl.DataFrame({"USUBJID": ["01-001", "01-002"]})
        out = _mk_findings(
            df=df,
            mask=pl.col("USUBJID").is_null(),  # no nulls here
            rule_id="SDTM_DM_001",
            severity="HIGH",
            field="USUBJID",
            message="USUBJID is required and must be non-empty.",
        )
        self.assert_findings_schema(out)
        self.assertEqual(out.height, 0)

    def test_validate_dm_missing_usubjid_column_dataset_level_finding(self):
        df = pl.DataFrame({"STUDYID": ["ABC"]})
        out = validate_dm(df)

        self.assert_findings_schema(out)
        self.assertEqual(out.height, 1)

        row = out.row(0, named=True)
        self.assertEqual(row["rule_id"], "SDTM_DM_001")
        self.assertEqual(row["severity"], "HIGH")
        self.assertEqual(row["field"], "USUBJID")
        self.assertEqual(row["row_index"], -1)
        self.assertIn("Missing required column USUBJID", row["message"])

    def test_validate_dm_usubjid_required_flags_null_and_blank_and_whitespace(self):
        df = pl.DataFrame({"USUBJID": ["01-001", None, "", "   ", "01-002"]})
        out = validate_dm(df)

        self.assert_findings_schema(out)

        dm001 = out.filter(pl.col("rule_id") == "SDTM_DM_001")
        self.assertEqual(dm001.height, 3)  # None, "", "   "

        bad_rows = sorted(dm001.get_column("row_index").to_list())
        self.assertEqual(bad_rows, [1, 2, 3])

    def test_validate_dm_usubjid_unique_flags_duplicates_only(self):
        df = pl.DataFrame({"USUBJID": ["01-001", "01-001", "01-002", "01-002", None]})
        out = validate_dm(df)

        self.assert_findings_schema(out)

        dm002 = out.filter(pl.col("rule_id") == "SDTM_DM_002")
        self.assertEqual(dm002.height, 2)  # only the 2nd occurrences are marked duplicated

        self.assertEqual(sorted(dm002.get_column("row_index").to_list()), [1, 3])
        self.assertEqual(sorted(dm002.get_column("evidence").to_list()), ["01-001", "01-002"])

    def test_validate_dm_studyid_required_flags_blank_and_null(self):
        df = pl.DataFrame({"USUBJID": ["01-001", "01-002"], "STUDYID": ["", None]})
        out = validate_dm(df)

        self.assert_findings_schema(out)

        dm003 = out.filter(pl.col("rule_id") == "SDTM_DM_003")
        self.assertEqual(dm003.height, 2)
        self.assertEqual(sorted(dm003.get_column("row_index").to_list()), [0, 1])

    def test_validate_dm_sex_controlled_terms_flags_invalid_only(self):
        df = pl.DataFrame(
            {
                "USUBJID": ["01-001", "01-002", "01-003", "01-004", "01-005"],
                "SEX": ["M", "F", "U", None, "X"],
            }
        )
        out = validate_dm(df)

        self.assert_findings_schema(out)

        dm004 = out.filter(pl.col("rule_id") == "SDTM_DM_004")
        self.assertEqual(dm004.height, 1)

        row = dm004.row(0, named=True)
        self.assertEqual(row["row_index"], 4)
        self.assertEqual(row["evidence"], "X")

    def test_validate_dm_age_bounds_flags_negative_and_over_120_only(self):
        df = pl.DataFrame(
            {
                "USUBJID": ["01-001", "01-002", "01-003", "01-004"],
                "AGE": [-1, 0, 120, 121],
            }
        )
        out = validate_dm(df)

        self.assert_findings_schema(out)

        dm005 = out.filter(pl.col("rule_id") == "SDTM_DM_005")
        self.assertEqual(dm005.height, 2)
        self.assertEqual(sorted(dm005.get_column("row_index").to_list()), [0, 3])

    def test_validate_dm_age_non_numeric_does_not_flag_but_is_graceful(self):
        df = pl.DataFrame({"USUBJID": ["01-001", "01-002"], "AGE": ["abc", "42"]})
        out = validate_dm(df)

        self.assert_findings_schema(out)

        dm005 = out.filter(pl.col("rule_id") == "SDTM_DM_005")
        self.assertEqual(dm005.height, 0)

    def test_validate_dm_rfstdtc_iso_date_flags_bad_formats_only(self):
        df = pl.DataFrame(
            {
                "USUBJID": ["01-001", "01-002", "01-003", "01-004", "01-005"],
                "RFSTDTC": ["2026-01-22", "2026/01/22", "2026-1-2", None, ""],
            }
        )
        out = validate_dm(df)

        self.assert_findings_schema(out)

        dm006 = out.filter(pl.col("rule_id") == "SDTM_DM_006")
        self.assertEqual(dm006.height, 3)
        self.assertEqual(sorted(dm006.get_column("row_index").to_list()), [1, 2, 4])

    def test_validate_dm_multiple_rules_can_trigger_and_concat(self):
        df = pl.DataFrame(
            {
                "USUBJID": ["01-001", "01-001", ""],      # dup + blank
                "STUDYID": ["ABC", "", "XYZ"],           # one blank
                "SEX": ["M", "X", "F"],                  # one invalid
                "AGE": [30, 200, -5],                    # 200 and -5 invalid
                "RFSTDTC": ["2026-01-01", "bad", None],  # "bad" invalid; None ignored
            }
        )
        out = validate_dm(df)

        self.assert_findings_schema(out)
        self.assertEqual(out.height, 7)

        self.assertEqual(
            out.filter(pl.col("rule_id") == "SDTM_DM_002").get_column("row_index").to_list(),
            [1],
        )
        self.assertEqual(
            out.filter(pl.col("rule_id") == "SDTM_DM_001").get_column("row_index").to_list(),
            [2],
        )


if __name__ == "__main__":
    unittest.main()
