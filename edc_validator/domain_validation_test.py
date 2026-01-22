from __future__ import annotations
import polars as pl
import unittest

from unittest.mock import patch, MagicMock
from edc_validator.validator import _ensure_columns, load_data, apply_rules, detect_anomalies
from edc_validator.sdtm_rules import validate_dm

class TestDomainValidation(unittest.TestCase):
    """Unit tests for domain validation functions."""

    def setUp(self):
        # Sample data for testing
        self.sample_data = pl.DataFrame({
            "age": [25, 45, 17, 101, None],
            "systolic_bp": [120.0, 200.0, 85.0, 150.0, None],
            "treatment_dose": [50.0, -10.0, 0.0, 25.0, None],
            "visit_date": ["2023-01-01", None, "2023-03-15", "2023-04-20", "2023-05-30"]
        })

    def test_ensure_columns_all_present(self):
        """Test _ensure_columns with all required columns present."""
        try:
            _ensure_columns(self.sample_data, ["age", "systolic_bp", "treatment_dose", "visit_date"])
        except ValueError:
            self.fail("_ensure_columns raised ValueError unexpectedly!")

    def test_ensure_columns_missing(self):
        """Test _ensure_columns with missing columns."""
        with self.assertRaises(ValueError) as context:
            _ensure_columns(self.sample_data, ["age", "missing_col"])
        self.assertIn("Missing required columns", str(context.exception))

    @patch("edc_validator.validator.pl.read_csv")
    def test_load_data_file_not_found(self, mock_read_csv):
        """Test load_data raises FileNotFoundError for missing file."""
        mock_read_csv.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            load_data("non_existent_file.csv")

    def test_apply_rules(self):
        """Test apply_rules function."""
        result_df = apply_rules(self.sample_data)
        self.assertIn("age_valid", result_df.columns)
        self.assertIn("bp_valid", result_df.columns)
        self.assertIn("dose_valid", result_df.columns)
        self.assertIn("date_valid", result_df.columns)

        # Check specific rule results
        self.assertEqual(result_df.filter(pl.col("age") == 25).select("age_valid").item(), 1)
        self.assertEqual(result_df.filter(pl.col("age") == 17).select("age_valid").item(), 0)

    def test_detect_anomalies_few_rows(self):
        """Test detect_anomalies with fewer than 10 rows."""
        small_df = self
        
if __name__ == "__main__":
    unittest.main().sample_data.head(5).head(5)
        