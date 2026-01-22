import polars as pl
import unittest

from edc_validator.domain_validation import (
    validate_dm_link, validate_dm_link, validate_vs, validate_ae, validate_cm,
    validate_vs_ae, validate_vs_cm
)
from edc_validator.sdtm_rules import validate_dm

class TestDMValidation(unittest.TestCase):
    def setUp(self):
        # Sample DM DataFrame
        self.dm = pl.DataFrame({
            "USUBJID": ["SUBJ001", "SUBJ002", "SUBJ003"]
        })

        # Sample VS DataFrame with one orphan USUBJID
        self.vs = pl.DataFrame({
            "USUBJID": ["SUBJ001", "SUBJ004"],  # SUBJ004 is orphan
            "VSTESTCD": ["HR", "BP"],
            "VISIT": ["BASELINE", "WEEK 1"]
        })

    def test_validate_dm_link(self):
        from edc_validator.run_validator import validate_dm_link

        findings = validate_dm_link(self.dm, self.vs, "VS")
        
        self.assertEqual(findings.height, 1)
        self.assertEqual(findings[0, "usubjid"], "SUBJ004")
        self.assertEqual(findings[0, "rule_id"], "X_DMLINK_VS_001")
        self.assertEqual(findings[0, "severity"], "HIGH")
        
    def run_all_validations(dm: pl.DataFrame, vs: pl.DataFrame, ae: pl.DataFrame, cm: pl.DataFrame) -> pl.DataFrame:
        all_findings = []

        all_findings.append(validate_dm(dm))
        all_findings.append(validate_vs(vs))
        all_findings.append(validate_ae(ae))
        all_findings.append(validate_cm(cm))
        all_findings.append(validate_vs_ae(vs, ae))
        all_findings.append(validate_vs_cm(vs, cm))
        all_findings.append(validate_dm_link(dm, vs, "VS"))
        all_findings.append(validate_dm_link(dm, ae, "AE"))
        all_findings.append(validate_dm_link(dm, cm, "CM"))

        return pl.concat(all_findings)
    
if __name__ == "__main__":
    unittest.main()