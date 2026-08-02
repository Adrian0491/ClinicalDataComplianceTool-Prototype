from __future__ import annotations

import unittest

from app.validators.schemas import *


class TestSchemas(unittest.TestCase):
    
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
    
    def test_empty_findings(self):
        df = empty_findings()
        self.assertTrue(df.empty)
        self.assertListEqual(list(df.columns), FINDINGS_COLUMNS)
        self.assertDictEqual(df.dtypes.apply(lambda dt: dt.name).to_dict(), FINDINGS_DTYPES)

    def test_dataset_finding_defaults(self):
        df = dataset_finding(
            rule_id="R001",
            severity="HIGH",
            domain="DM",
            field="AGE",
            message="Age must be >= 18"
        )
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["finding_type"], "SDTM_RULE")
        self.assertEqual(row["rule_id"], "R001")
        self.assertEqual(row["severity"], "HIGH")
        self.assertEqual(row["domain"], "DM")
        self.assertEqual(row["field"], "AGE")
        self.assertEqual(row["message"], "Age must be >= 18")
        self.assertEqual(row["row_index"], -1)
        self.assertEqual(row["usubjid"], "")
        self.assertEqual(row["evidence"], "")

    def test_dataset_finding_custom(self):
        df = dataset_finding(
            rule_id="R002",
            severity="CRIT",
            domain="AE",
            field="AESTDTC",
            message="Start date is required",
            finding_type="ANOMALY",
            evidence="Missing in 5 records"
        )
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["finding_type"], "ANOMALY")
        self.assertEqual(row["rule_id"], "R002")
        self.assertEqual(row["severity"], "CRIT")
        self.assertEqual(row["domain"], "AE")
        self.assertEqual(row["field"], "AESTDTC")
        self.assertEqual(row["message"], "Start date is required")
        self.assertEqual(row["row_index"], -1)
        self.assertEqual(row["usubjid"], "")
        self.assertEqual(row["evidence"], "Missing in 5 records")
        
if __name__ == "__main__":
    unittest.main()