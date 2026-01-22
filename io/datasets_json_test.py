import polars as pl

import pytest

from edc_validator.io.datasets_json import DatasetJsonIO
from io.datasetjson_pipeline import DatasetJsonValidator
from edc_validator.sdtm_rules import validate_dm
from edc_validator.domain_validation import validate_vs, validate_ae, validate_cm
from edc_validator.run_validator import validate_dm_link



def test_validate_top_level_missing_required_fields():
    dj = DatasetJsonIO()
    doc = {"clinicalData": {"itemGroupData": {}}}

    f = dj.validate_top_level(doc)

    assert isinstance(f, pl.DataFrame)
    assert f.height >= 2
    assert "DJ_000" in f["rule_id"].to_list()


def test_validate_top_level_requires_root_object():
    dj = DatasetJsonIO()
    doc = {
        "creationDateTime": "2026-01-22T00:00:00",
        "datasetJSONVersion": "1.1",
        "clinicalData": [],  # invalid type
    }

    f = dj.validate_top_level(doc)
    assert "DJ_002" in f["rule_id"].to_list()


def test_itemgroup_to_polars_happy_path():
    dj = DatasetJsonIO()
    doc = {
        "creationDateTime": "2026-01-22T00:00:00",
        "datasetJSONVersion": "1.1",
        "clinicalData": {
            "itemGroupData": {
                "IG.DM": {
                    "name": "DM",
                    "items": [{"OID": "IT.USUBJID", "name": "USUBJID"}],
                    "itemData": [["01-001"], ["01-002"]],
                }
            }
        },
    }

    df, f = dj.itemgroup_to_polars(doc, "IG.DM")

    assert isinstance(df, pl.DataFrame)
    assert df.shape == (2, 1)
    assert df.columns == ["USUBJID"]
    assert f.height == 0


def test_itemgroup_to_polars_length_mismatch_flags_but_still_returns_df():
    dj = DatasetJsonIO()
    doc = {
        "creationDateTime": "2026-01-22T00:00:00",
        "datasetJSONVersion": "1.1",
        "clinicalData": {
            "itemGroupData": {
                "IG.VS": {
                    "name": "VS",
                    "items": [{"name": "USUBJID"}, {"name": "VSTESTCD"}],
                    "itemData": [["01-001"]],  # missing second value
                }
            }
        },
    }

    df, f = dj.itemgroup_to_polars(doc, "IG.VS")

    assert df.shape == (1, 2)
    assert "DJ_106" in f["rule_id"].to_list()
    # Should pad missing values with None
    assert df["VSTESTCD"].to_list() == [None]


def test_domain_to_polars_infers_by_name():
    dj = DatasetJsonIO()
    doc = {
        "clinicalData": {
            "itemGroupData": {
                "IG.SOMETHING": {
                    "name": "DM",
                    "items": [{"name": "USUBJID"}],
                    "itemData": [["01-001"]],
                }
            }
        }
    }

    df, f, oid = dj.domain_to_polars(doc, "DM")
    assert oid == "IG.SOMETHING"
    assert df.columns == ["USUBJID"]
    assert f.height == 0

def _mk_findings(rule_id: str, row_index: int = -1) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "finding_type": ["TEST"],
            "rule_id": [rule_id],
            "severity": ["LOW"],
            "field": ["X"],
            "message": ["ok"],
            "row_index": [row_index],
            "evidence": [""],
        }
    )


@pytest.fixture
def sample_doc():
    # Minimal Dataset-JSON with DM + VS + AE + CM
    return {
        "creationDateTime": "2026-01-22T00:00:00",
        "datasetJSONVersion": "1.1",
        "clinicalData": {
            "itemGroupData": {
                "IG.DM": {
                    "name": "DM",
                    "items": [{"name": "USUBJID"}, {"name": "STUDYID"}],
                    "itemData": [["01-001", "ABC"], ["01-002", "ABC"]],
                },
                "IG.VS": {
                    "name": "VS",
                    "items": [{"name": "USUBJID"}, {"name": "VSTESTCD"}, {"name": "VSORRES"}, {"name": "VSDTC"}],
                    "itemData": [["01-001", "SYSBP", "120", "2026-01-01"]],
                },
                "IG.AE": {
                    "name": "AE",
                    "items": [{"name": "USUBJID"}, {"name": "AETERM"}],
                    "itemData": [["01-001", "Headache"]],
                },
                "IG.CM": {
                    "name": "CM",
                    "items": [{"name": "USUBJID"}, {"name": "CMTRT"}],
                    "itemData": [["01-001", "Ibuprofen"]],
                },
            }
        },
    }


def test_pipeline_extracts_domains_and_runs_validators(monkeypatch, sample_doc):
    """
    Prove orchestration logic calls downstream validators and returns merged findings.
    """
    # ---- monkeypatch the imported validator functions inside the pipeline module ----
    # IMPORTANT: patch the names as imported in io.dataset_json_validator
    import io.dataset_json_validator as pipe

    monkeypatch.setattr(pipe, "validate_dm", lambda df: _mk_findings("DM_CALLED"))
    monkeypatch.setattr(pipe, "validate_vs", lambda df: _mk_findings("VS_CALLED"))
    monkeypatch.setattr(pipe, "validate_ae", lambda df: _mk_findings("AE_CALLED"))
    monkeypatch.setattr(pipe, "validate_cm", lambda df: _mk_findings("CM_CALLED"))
    monkeypatch.setattr(pipe, "validate_dm_link", lambda dm, ddf, dom: _mk_findings(f"DMLINK_{dom}"))

    v = DatasetJsonValidator(io=DatasetJsonIO())
    result = v.validate(sample_doc)

    assert "DM" in result.datasets
    assert "VS" in result.datasets
    assert result.datasets["DM"].height == 2

    # Ensure our stubbed validators were called
    rules = set(result.findings["rule_id"].to_list())
    assert "DM_CALLED" in rules
    assert "VS_CALLED" in rules
    assert "AE_CALLED" in rules
    assert "CM_CALLED" in rules

    # Cross-domain links expected for VS/AE/CM
    assert "DMLINK_VS" in rules
    assert "DMLINK_AE" in rules
    assert "DMLINK_CM" in rules


def test_pipeline_handles_missing_domain_gracefully(monkeypatch):
    """
    If a domain isn't present, it should still return findings (e.g., DJ_110) but not crash.
    """
    import io.dataset_json_validator as pipe

    # Stub validators to avoid dependency
    monkeypatch.setattr(pipe, "validate_dm", lambda df: _mk_findings("DM_CALLED"))
    monkeypatch.setattr(pipe, "validate_vs", lambda df: _mk_findings("VS_CALLED"))
    monkeypatch.setattr(pipe, "validate_ae", lambda df: _mk_findings("AE_CALLED"))
    monkeypatch.setattr(pipe, "validate_cm", lambda df: _mk_findings("CM_CALLED"))
    monkeypatch.setattr(pipe, "validate_dm_link", lambda dm, ddf, dom: _mk_findings(f"DMLINK_{dom}"))

    # Only DM exists
    doc = {
        "creationDateTime": "2026-01-22T00:00:00",
        "datasetJSONVersion": "1.1",
        "clinicalData": {
            "itemGroupData": {
                "IG.DM": {
                    "name": "DM",
                    "items": [{"name": "USUBJID"}],
                    "itemData": [["01-001"]],
                }
            }
        },
    }

    v = DatasetJsonValidator(io=DatasetJsonIO())
    result = v.validate(doc, domains=["DM", "VS"])  # ask for VS, but it's missing

    assert result.datasets["DM"].height == 1
    assert "VS" in result.datasets  # will likely be empty DF
    assert isinstance(result.findings, pl.DataFrame)

    # We should get a critical finding for missing VS mapping (DJ_110 from DatasetJsonIO.domain_to_polars)
    assert "DJ_110" in result.findings["rule_id"].to_list()