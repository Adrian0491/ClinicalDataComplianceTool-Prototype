from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.validators.schemas import (
    FINDINGS_COLUMNS,
    FINDINGS_DTYPES,
    concat_findings,
    empty_findings,
)


class DatasetJsonIO:
    """Loads a Dataset-JSON document and extracts domain datasets as DataFrames."""

    def __init__(self, finding_type: str = "DATASET_JSON") -> None:
        self.finding_type = finding_type

    def _finding(self, rule_id: str, severity: str, field: str, message: str,
                 row_index: int = -1, evidence: str = "") -> pd.DataFrame:
        row = {
            "finding_type": self.finding_type, "rule_id": rule_id,
            "severity": severity, "domain": "DATASET_JSON",
            "field": field, "message": message,
            "row_index": row_index, "usubjid": "", "evidence": evidence,
        }
        return pd.DataFrame([row], columns=FINDINGS_COLUMNS).astype(FINDINGS_DTYPES)

    def load(self, path: str | Path) -> dict[str, Any]:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def validate_top_level(self, doc: dict[str, Any]) -> pd.DataFrame:
        findings = []
        if not isinstance(doc, dict):
            return self._finding("DJ_001", "CRIT", "root",
                                 "Dataset-JSON document must be a JSON object.")
        root_key = next((k for k in ("clinicalData", "referenceData") if k in doc), None)
        if root_key is None:
            return self._finding("DJ_002", "CRIT", "clinicalData/referenceData",
                                 "Document must have a 'clinicalData' or 'referenceData' key.")
        root = doc[root_key]
        if not isinstance(root, dict):
            findings.append(self._finding("DJ_002b", "CRIT", root_key,
                                          f"{root_key} must be a JSON object."))
        elif "itemGroupData" not in root:
            findings.append(self._finding("DJ_003", "CRIT", "itemGroupData",
                                          "Missing 'itemGroupData' in document."))
        elif not isinstance(root["itemGroupData"], dict):
            findings.append(self._finding("DJ_004", "CRIT", "itemGroupData",
                                          "itemGroupData must be an object."))
        return concat_findings(findings)

    def _get_ig_map(self, doc: dict[str, Any]) -> dict[str, Any]:
        for k in ("clinicalData", "referenceData"):
            if k in doc:
                return doc[k].get("itemGroupData", {})
        return {}

    def list_itemgroups(self, doc: dict[str, Any]) -> list[str]:
        return list(self._get_ig_map(doc).keys())

    def infer_oid(self, doc: dict[str, Any], domain: str) -> str | None:
        domain_u = domain.strip().upper()
        ig = self._get_ig_map(doc)
        for k, ds in ig.items():
            if isinstance(ds, dict) and str(ds.get("name", "")).upper() == domain_u:
                return str(k)
        for k in ig:
            ku = str(k).upper()
            if ku.endswith(domain_u) or f".{domain_u}" in ku:
                return str(k)
        return None

    def itemgroup_to_df(self, doc: dict[str, Any], oid: str
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
        findings = []
        ig  = self._get_ig_map(doc)
        ds  = ig.get(oid)
        if ds is None or not isinstance(ds, dict):
            return pd.DataFrame(), self._finding("DJ_100", "CRIT", "itemGroupData",
                                                  f"OID '{oid}' not found.")
        items     = ds.get("items", [])
        item_data = ds.get("itemData", [])

        if not isinstance(items, list) or not items:
            return pd.DataFrame(), self._finding("DJ_101", "CRIT", f"{oid}.items",
                                                  "Non-empty 'items' array required.")
        if not isinstance(item_data, list):
            return pd.DataFrame(), self._finding("DJ_102", "CRIT", f"{oid}.itemData",
                                                  "'itemData' must be an array.")

        col_names = [str(it.get("name") or it.get("OID") or f"COL_{i}")
                     for i, it in enumerate(items) if isinstance(it, dict)]
        if not col_names:
            return pd.DataFrame(), self._finding("DJ_104", "CRIT", f"{oid}.items",
                                                  "Could not derive column names.")

        rows = []
        for r_idx, rec in enumerate(item_data):
            if not isinstance(rec, list):
                findings.append(self._finding("DJ_105", "HIGH", f"{oid}.itemData[{r_idx}]",
                                              "Each record must be an array.", row_index=r_idx))
                continue
            if len(rec) != len(col_names):
                findings.append(self._finding("DJ_106", "HIGH", f"{oid}.itemData[{r_idx}]",
                                              "Record length mismatch.", row_index=r_idx,
                                              evidence=f"got {len(rec)}, expected {len(col_names)}"))
            rows.append((rec + [None] * len(col_names))[: len(col_names)])

        df = pd.DataFrame(rows, columns=col_names) if rows else pd.DataFrame(columns=col_names)
        return df, concat_findings(findings)

    def domain_to_df(self, doc: dict[str, Any], domain: str
                     ) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
        oid = self.infer_oid(doc, domain)
        if oid is None:
            return pd.DataFrame(), self._finding(
                "DJ_110", "CRIT", "domain",
                f"Could not infer OID for domain '{domain}'.",
                evidence="; ".join(self.list_itemgroups(doc)),
            ), None
        df, f = self.itemgroup_to_df(doc, oid)
        return df, f, oid


@dataclass
class DatasetJsonValidationResult:
    datasets: dict[str, pd.DataFrame] = field(default_factory=dict)
    findings: pd.DataFrame            = field(default_factory=empty_findings)


class DatasetJsonValidator:
    """Orchestrates Dataset-JSON structural + SDTM domain validation."""

    def __init__(self, io: DatasetJsonIO | None = None) -> None:
        self.io = io or DatasetJsonIO()

    def validate(self, doc: dict[str, Any],
                 domains: list[str] | None = None) -> DatasetJsonValidationResult:
        from app.validators.domain import (
            validate_ae,
            validate_cm,
            validate_dm,
            validate_dm_link,
            validate_vs,
        )
        if domains is None:
            domains = ["DM", "VS", "AE", "CM"]

        datasets: dict[str, pd.DataFrame] = {}
        parts: list[pd.DataFrame] = [self.io.validate_top_level(doc)]

        for domain in domains:
            df, f, _ = self.io.domain_to_df(doc, domain)
            datasets[domain] = df
            parts.append(f)

        dm_df = datasets.get("DM", pd.DataFrame())
        if len(dm_df): parts.append(validate_dm(dm_df))

        vs_df = datasets.get("VS", pd.DataFrame())
        if len(vs_df): parts.append(validate_vs(vs_df))

        ae_df = datasets.get("AE", pd.DataFrame())
        if len(ae_df): parts.append(validate_ae(ae_df))

        cm_df = datasets.get("CM", pd.DataFrame())
        if len(cm_df): parts.append(validate_cm(cm_df))

        if len(dm_df):
            for dom in ["VS", "AE", "CM"]:
                ddf = datasets.get(dom, pd.DataFrame())
                if len(ddf):
                    parts.append(validate_dm_link(dm_df, ddf, dom))

        return DatasetJsonValidationResult(
            datasets=datasets,
            findings=concat_findings(parts),
        )
