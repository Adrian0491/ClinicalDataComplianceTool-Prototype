from __future__ import annotations

import unittest

from app.ingest.datasets_json import DatasetJsonIO


class TestDatasetJsonIO(unittest.TestCase):
    def setUp(self) -> None:
        self.io = DatasetJsonIO()

    def test_validate_top_level_valid_clinical(self):
        doc = {
            "clinicalData": {
                "itemGroupData": {}
            }
        }
        findings = self.io.validate_top_level(doc)
        self.assertTrue(findings.empty)

    def test_validate_top_level_valid_reference(self):
        doc = {
            "referenceData": {
                "itemGroupData": {}
            }
        }
        findings = self.io.validate_top_level(doc)
        self.assertTrue(findings.empty)

    def test_validate_top_level_not_object(self):
        doc = []
        findings = self.io.validate_top_level(doc)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings.iloc[0]["rule_id"], "DJ_001")

    def test_validate_top_level_missing_root_key(self):
        doc = {"foo": {}}
        findings = self.io.validate_top_level(doc)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings.iloc[0]["rule_id"], "DJ_002")

    def test_validate_top_level_root_not_object(self):
        doc = {"clinicalData": []}
        findings = self.io.validate_top_level(doc)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings.iloc[0]["rule_id"], "DJ_002b")

    def test_validate_top_level_missing_itemGroupData(self):
        doc = {"clinicalData": {}}
        findings = self.io.validate_top_level(doc)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings.iloc[0]["rule_id"], "DJ_003")

    def test_validate_top_level_itemGroupData_not_object(self):
        doc = {"clinicalData": {"itemGroupData": []}}
        findings = self.io.validate_top_level(doc)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings.iloc[0]["rule_id"], "DJ_004")

if __name__ == "__main__":
    unittest.main()
