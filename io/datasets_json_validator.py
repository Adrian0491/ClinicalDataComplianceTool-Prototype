from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import polars as pl

from io.datasets_json import DatasetJsonIO
from edc_validator.sdtm_rules import validate_dm
from edc_validator.domain_validation import (
    validate_vs,
    validate_ae,
    validate_cm,
)
from edc_validator.run_validator import validate_dm_link


@dataclass
class DatasetJsonValidationResult:
    datasets: Dict[str, pl.DataFrame]
    findings: pl.DataFrame


class DatasetJsonValidator:
    """
    Orchestrates validation of Dataset-JSON inputs by:
    - running DatasetJsonIO structural checks
    - extracting domain datasets to Polars
    - applying SDTM and domain validators
    - applying cross-domain DM link checks
    """

    def __init__(self, io: Optional[DatasetJsonIO] = None):
        self.io = io or DatasetJsonIO()

    def validate(
        self,
        doc: Dict[str, Any],
        *,
        domains: Optional[List[str]] = None,
    ) -> DatasetJsonValidationResult:
        if domains is None:
            domains = ["DM", "VS", "AE", "CM"]

        datasets: Dict[str, pl.DataFrame] = {}
        findings_parts: List[pl.DataFrame] = []

        # 1) Dataset-JSON structural validation
        findings_parts.append(self.io.validate_top_level(doc))

        # 2) Extract datasets
        for domain in domains:
            df, f, oid = self.io.domain_to_polars(doc, domain)
            datasets[domain] = df
            findings_parts.append(f)

        # 3) Run domain validators
        dm_df = datasets.get("DM", pl.DataFrame())
        if dm_df.height > 0:
            findings_parts.append(validate_dm(dm_df))

        vs_df = datasets.get("VS", pl.DataFrame())
        if vs_df.height > 0:
            findings_parts.append(validate_vs(vs_df))

        ae_df = datasets.get("AE", pl.DataFrame())
        if ae_df.height > 0:
            findings_parts.append(validate_ae(ae_df))

        cm_df = datasets.get("CM", pl.DataFrame())
        if cm_df.height > 0:
            findings_parts.append(validate_cm(cm_df))

        # 4) Cross-domain DM link checks
        if dm_df.height > 0:
            for dom in ["VS", "AE", "CM"]:
                ddf = datasets.get(dom, pl.DataFrame())
                if ddf.height > 0:
                    findings_parts.append(validate_dm_link(dm_df, ddf, dom))

        # 5) Merge all findings
        findings = pl.concat(
            [f for f in findings_parts if isinstance(f, pl.DataFrame) and f.width > 0],
            how="vertical_relaxed",
        ) if findings_parts else pl.DataFrame()

        return DatasetJsonValidationResult(
            datasets=datasets,
            findings=findings,
        )
