from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import polars as pl
import json
from pathlib import Path

from edc_validator.sdtm_rules import validate_dm
from edc_validator.domain_validation import validate_vs, validate_ae, validate_cm  # adjust names to your file
from edc_validator.run_validator import validate_dm_link  # if you already have this helper


@dataclass
class DatasetJsonValidationResult:
    datasets: Dict[str, pl.DataFrame]
    findings: pl.DataFrame


class DatasetJsonIO:
    """
    Dataset-JSON loader + lightweight structural validator + extractor to Polars.

    Design goals:
    - Best-effort parsing to Polars DataFrame for a dataset (ItemGroup).
    - Generate findings as a Polars DataFrame with a consistent schema.
    - Avoid hard dependency on Define-XML for now (structural checks only).
    """

    FINDINGS_SCHEMA = {
        "finding_type": pl.Utf8,
        "rule_id": pl.Utf8,
        "severity": pl.Utf8,
        "field": pl.Utf8,
        "message": pl.Utf8,
        "row_index": pl.Int64,
        "evidence": pl.Utf8,
    }

    def __init__(self, *, finding_type: str = "DATASET_JSON"):
        self.finding_type = finding_type

    # ---------- Findings helpers ----------

    def empty_findings(self) -> pl.DataFrame:
        return pl.DataFrame(schema=self.FINDINGS_SCHEMA)

    def finding(
        self,
        rule_id: str,
        severity: str,
        field: str,
        message: str,
        *,
        row_index: int = -1,
        evidence: str = "",
    ) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "finding_type": [self.finding_type],
                "rule_id": [rule_id],
                "severity": [severity],
                "field": [field],
                "message": [message],
                "row_index": [row_index],
                "evidence": [evidence],
            },
            schema=self.FINDINGS_SCHEMA,
        )

    def concat_findings(self, parts: List[pl.DataFrame]) -> pl.DataFrame:
        parts = [p for p in parts if isinstance(p, pl.DataFrame) and p.width > 0]
        if not parts:
            return self.empty_findings()
        return pl.concat(parts, how="vertical_relaxed")

    # ---------- IO ----------

    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load Dataset-JSON from a path."""
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    # ---------- Structure inspection ----------

    def get_root(self, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return clinicalData or referenceData object."""
        root = doc.get("clinicalData") or doc.get("referenceData")
        return root if isinstance(root, dict) else None

    def get_itemgroup_map(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Return itemGroupData mapping or empty dict."""
        root = self.get_root(doc) or {}
        ig = root.get("itemGroupData") or {}
        return ig if isinstance(ig, dict) else {}

    def list_itemgroups(self, doc: Dict[str, Any]) -> List[str]:
        return list(self.get_itemgroup_map(doc).keys())

    def validate_top_level(self, doc: Dict[str, Any]) -> pl.DataFrame:
        """
        Lightweight structural validation (not full schema validation).
        Returns findings DataFrame.
        """
        findings: List[pl.DataFrame] = []

        # DJ_000: required fields
        for req in ("creationDateTime", "datasetJSONVersion"):
            if req not in doc:
                findings.append(
                    self.finding(
                        rule_id="DJ_000",
                        severity="CRIT",
                        field=req,
                        message=f"Missing required top-level attribute: {req}",
                    )
                )

        # DJ_001: must have clinicalData or referenceData
        has_clinical = "clinicalData" in doc
        has_reference = "referenceData" in doc
        if not (has_clinical or has_reference):
            findings.append(
                self.finding(
                    rule_id="DJ_001",
                    severity="CRIT",
                    field="clinicalData/referenceData",
                    message="Dataset-JSON must include either clinicalData or referenceData.",
                )
            )
            return self.concat_findings(findings)

        # DJ_002: root must be object
        root = doc.get("clinicalData") or doc.get("referenceData")
        if not isinstance(root, dict):
            findings.append(
                self.finding(
                    rule_id="DJ_002",
                    severity="CRIT",
                    field="clinicalData/referenceData",
                    message="clinicalData/referenceData must be a JSON object.",
                    evidence=str(type(root)),
                )
            )
            return self.concat_findings(findings)

        # DJ_003/004: itemGroupData existence and type
        if "itemGroupData" not in root:
            findings.append(
                self.finding(
                    rule_id="DJ_003",
                    severity="CRIT",
                    field="itemGroupData",
                    message="Missing itemGroupData object under clinicalData/referenceData.",
                )
            )
        elif not isinstance(root.get("itemGroupData"), dict):
            findings.append(
                self.finding(
                    rule_id="DJ_004",
                    severity="CRIT",
                    field="itemGroupData",
                    message="itemGroupData must be a JSON object mapping dataset OIDs to datasets.",
                    evidence=str(type(root.get("itemGroupData"))),
                )
            )

        return self.concat_findings(findings)

    # ---------- Domain helpers ----------

    def infer_itemgroup_oid_for_domain(self, doc: Dict[str, Any], domain: str) -> Optional[str]:
        """
        Best-effort: return itemGroup OID whose dataset 'name' equals domain.
        Falls back to key heuristics (endswith domain or contains '.DOMAIN').
        """
        domain_u = domain.strip().upper()
        ig = self.get_itemgroup_map(doc)

        # Prefer dataset objects with ds['name'] == domain
        for k, ds in ig.items():
            if isinstance(ds, dict) and str(ds.get("name", "")).upper() == domain_u:
                return str(k)

        # Fallback: key heuristics (e.g., "IG.DM")
        for k in ig.keys():
            ku = str(k).upper()
            if ku.endswith(domain_u) or f".{domain_u}" in ku:
                return str(k)

        return None

    # ---------- Extraction ----------

    def get_itemgroup(self, doc: Dict[str, Any], itemgroup_oid: str) -> Optional[Dict[str, Any]]:
        ds = self.get_itemgroup_map(doc).get(itemgroup_oid)
        return ds if isinstance(ds, dict) else None

    def itemgroup_to_polars(
        self,
        doc: Dict[str, Any],
        itemgroup_oid: str,
        *,
        strict: bool = False,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Convert one Dataset-JSON dataset (itemGroupData[OID]) into Polars DataFrame.

        Returns: (data_df, findings_df)
        """
        findings: List[pl.DataFrame] = []

        ds = self.get_itemgroup(doc, itemgroup_oid)
        if ds is None:
            return pl.DataFrame(), self.finding(
                rule_id="DJ_100",
                severity="CRIT",
                field="itemGroupData",
                message=f"ItemGroup '{itemgroup_oid}' not found in itemGroupData.",
                evidence=";".join(self.list_itemgroups(doc)),
            )

        items = ds.get("items")
        item_data = ds.get("itemData")

        if not isinstance(items, list) or len(items) == 0:
            findings.append(
                self.finding(
                    rule_id="DJ_101",
                    severity="CRIT",
                    field=f"{itemgroup_oid}.items",
                    message="Dataset must include a non-empty 'items' array.",
                )
            )
            if strict:
                return pl.DataFrame(), self.concat_findings(findings)

        if not isinstance(item_data, list):
            findings.append(
                self.finding(
                    rule_id="DJ_102",
                    severity="CRIT",
                    field=f"{itemgroup_oid}.itemData",
                    message="'itemData' must be an array of records.",
                    evidence=str(type(item_data)),
                )
            )
            if strict:
                return pl.DataFrame(), self.concat_findings(findings)

        # Build column names from items (prefer 'name', fallback to 'OID')
        col_names: List[str] = []
        for idx, it in enumerate(items or []):
            if not isinstance(it, dict):
                findings.append(
                    self.finding(
                        rule_id="DJ_103",
                        severity="HIGH",
                        field=f"{itemgroup_oid}.items[{idx}]",
                        message="Each element of 'items' must be an object.",
                        evidence=str(type(it)),
                    )
                )
                continue
            name = it.get("name") or it.get("OID") or f"COL_{idx}"
            col_names.append(str(name))

        if not col_names:
            findings.append(
                self.finding(
                    rule_id="DJ_104",
                    severity="CRIT",
                    field=f"{itemgroup_oid}.items",
                    message="Could not derive any column names from 'items'.",
                )
            )
            return pl.DataFrame(), self.concat_findings(findings)

        # Convert itemData: expected list of arrays aligned to items
        rows: List[List[Any]] = []
        if isinstance(item_data, list):
            for r_idx, rec in enumerate(item_data):
                if not isinstance(rec, list):
                    findings.append(
                        self.finding(
                            rule_id="DJ_105",
                            severity="HIGH",
                            field=f"{itemgroup_oid}.itemData[{r_idx}]",
                            message="Each itemData record should be an array of values.",
                            row_index=r_idx,
                            evidence=str(type(rec)),
                        )
                    )
                    continue

                if len(rec) != len(col_names):
                    findings.append(
                        self.finding(
                            rule_id="DJ_106",
                            severity="HIGH",
                            field=f"{itemgroup_oid}.itemData[{r_idx}]",
                            message="Record length does not match number of items.",
                            row_index=r_idx,
                            evidence=f"len(record)={len(rec)} len(items)={len(col_names)}",
                        )
                    )

                fixed = (rec + [None] * len(col_names))[: len(col_names)]
                rows.append(fixed)

        df = pl.DataFrame(rows, schema=col_names) if rows else pl.DataFrame(schema=col_names)
        return df, self.concat_findings(findings)

    # ---------- Convenience ----------

    def domain_to_polars(
        self,
        doc: Dict[str, Any],
        domain: str,
        *,
        strict: bool = False,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, Optional[str]]:
        """
        Convenience: infer itemGroup OID for a domain and convert to Polars.

        Returns: (df, findings, itemgroup_oid)
        """
        oid = self.infer_itemgroup_oid_for_domain(doc, domain)
        if oid is None:
            return pl.DataFrame(), self.finding(
                rule_id="DJ_110",
                severity="CRIT",
                field="domain",
                message=f"Could not infer itemGroup OID for domain '{domain}'.",
                evidence=";".join(self.list_itemgroups(doc)),
            ), None

        df, f = self.itemgroup_to_polars(doc, oid, strict=strict)
        return df, f, oid
