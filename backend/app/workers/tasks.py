"""
Celery tasks for the CDTool validation pipeline.

run_validation_job:
  1. Load the ValidationJob and its Dataset records from PostgreSQL.
  2. Download each dataset file from storage.
  3. Dispatch to the appropriate validator (SDTM or Dataset-JSON).
  4. Persist findings to PostgreSQL.
  5. Index findings in Elasticsearch (if enabled).
  6. Mark the job as completed (or failed).
"""
from __future__ import annotations

import io
import json
import logging
import uuid
from datetime import datetime, timezone

import pandas as pd

from app.workers.celery_app import celery_app

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main validation task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="app.workers.tasks.run_validation_job", max_retries=2)
def run_validation_job(self, job_id: str) -> dict:
    """
    Execute the full validation pipeline for a ValidationJob.

    Returns a summary dict with counts by severity.
    """
    from app.config import get_settings
    from app.database import SessionLocal
    from app.models.dataset import Dataset
    from app.models.validation import ValidationJob
    from app.storage.backends import get_storage

    settings = get_settings()
    db = SessionLocal()
    storage = get_storage()

    try:
        # ── 1. Load job ──────────────────────────────────────────────────
        job: ValidationJob | None = db.get(ValidationJob, job_id)
        if job is None:
            log.error("ValidationJob %s not found.", job_id)
            return {"error": "job not found"}

        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        db.commit()

        # ── 2. Load datasets ─────────────────────────────────────────────
        dataset_ids = [uuid.UUID(did) for did in job.dataset_ids]
        datasets = db.query(Dataset).filter(Dataset.id.in_(dataset_ids)).all()

        if not datasets:
            job.status = "failed"
            job.error_message = "No datasets found for this job."
            db.commit()
            return {"error": "no datasets"}

        # ── 3. Parse into DataFrames keyed by domain ─────────────────────
        domain_frames: dict[str, pd.DataFrame] = {}
        all_findings_dfs: list[pd.DataFrame] = []

        for dataset in datasets:
            raw_bytes = storage.read(dataset.storage_uri)
            df, extra_findings = _parse_dataset(raw_bytes, dataset)
            if extra_findings is not None:
                all_findings_dfs.append(extra_findings)
            if df is not None and not df.empty:
                domain_frames[dataset.domain] = df

        # ── 4. Run SDTM validators ────────────────────────────────────────
        from app.validators.domain import (
            validate_ae,
            validate_cm,
            validate_dm,
            validate_dm_link,
            validate_ds,
            validate_eg,
            validate_ex,
            validate_irt_consistency,
            validate_lb,
            validate_pr,
            validate_qs,
            validate_rs,
            validate_vs,
            validate_vs_ae,
            validate_vs_cm,
        )
        from app.validators.schemas import concat_findings

        dm = domain_frames.get("DM", pd.DataFrame())
        vs = domain_frames.get("VS", pd.DataFrame())
        ae = domain_frames.get("AE", pd.DataFrame())
        cm = domain_frames.get("CM", pd.DataFrame())
        ex = domain_frames.get("EX", pd.DataFrame())
        lb = domain_frames.get("LB", pd.DataFrame())
        ds = domain_frames.get("DS", pd.DataFrame())
        eg = domain_frames.get("EG", pd.DataFrame())
        qs = domain_frames.get("QS", pd.DataFrame())
        rs = domain_frames.get("RS", pd.DataFrame())
        pr = domain_frames.get("PR", pd.DataFrame())

        if not dm.empty: all_findings_dfs.append(validate_dm(dm))
        if not vs.empty: all_findings_dfs.append(validate_vs(vs))
        if not ae.empty: all_findings_dfs.append(validate_ae(ae))
        if not cm.empty: all_findings_dfs.append(validate_cm(cm))
        if not ex.empty: all_findings_dfs.append(validate_ex(ex))
        if not lb.empty: all_findings_dfs.append(validate_lb(lb))
        if not ds.empty: all_findings_dfs.append(validate_ds(ds))
        if not eg.empty: all_findings_dfs.append(validate_eg(eg))
        if not qs.empty: all_findings_dfs.append(validate_qs(qs))
        if not rs.empty: all_findings_dfs.append(validate_rs(rs))
        if not pr.empty: all_findings_dfs.append(validate_pr(pr))

        # Cross-domain
        if not dm.empty:
            for label, df in [("VS", vs), ("AE", ae), ("CM", cm), ("EX", ex), ("LB", lb), ("DS", ds), ("EG", eg), ("QS", qs), ("RS", rs), ("PR", pr)]:
                if not df.empty:
                    all_findings_dfs.append(validate_dm_link(dm, df, label))
        if not dm.empty and not ex.empty:
            all_findings_dfs.append(validate_irt_consistency(dm, ex, ds))
        if not vs.empty and not ae.empty:
            all_findings_dfs.append(validate_vs_ae(vs, ae))
        if not vs.empty and not cm.empty:
            all_findings_dfs.append(validate_vs_cm(vs, cm))

        # ── 4b. Anomaly detection (Isolation Forest) ───────────────────────
        from app.validators.anomaly import (
            apply_rules,
            build_frame_from_domains,
            detect_anomalies,
            to_findings,
        )

        if not dm.empty:
            generic = build_frame_from_domains(dm, vs, ex, lb=lb, eg=eg, qs=qs, pr=pr)
            if not generic.empty:
                generic = apply_rules(generic)
                generic = detect_anomalies(generic)
                anomaly_findings = to_findings(generic)
                if not anomaly_findings.empty:
                    all_findings_dfs.append(anomaly_findings)

        # ── 5. Persist findings ──────────────────────────────────────────
        combined = concat_findings(all_findings_dfs) if all_findings_dfs else pd.DataFrame()
        finding_rows = _persist_findings(db, combined, job)
        db.commit()

        # ── 6. Index in Elasticsearch (best-effort) ───────────────────────
        if settings.ENABLE_ES_INDEXING and finding_rows:
            try:
                _index_findings_es(finding_rows, settings)
            except Exception as es_err:
                log.warning("Elasticsearch indexing failed (non-fatal): %s", es_err)

        # ── 7. Mark complete ─────────────────────────────────────────────
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        db.commit()

        summary = _build_summary(combined)
        log.info("Job %s completed: %s", job_id, summary)
        return summary

    except Exception as exc:
        log.exception("Validation job %s failed: %s", job_id, exc)
        try:
            job = db.get(ValidationJob, job_id)
            if job:
                job.status = "failed"
                job.error_message = str(exc)
                db.commit()
        except Exception:
            pass
        raise self.retry(exc=exc, countdown=30)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_dataset(
    raw_bytes: bytes, dataset
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Parse raw bytes into a DataFrame.
    Returns (domain_df, structural_findings_df).
    """
    fmt = dataset.file_format.lower()
    extra_findings = None

    if fmt == "csv":
        df = pd.read_csv(io.BytesIO(raw_bytes))
        return df, None

    if fmt in ("json", "dataset-json"):
        doc = json.loads(raw_bytes.decode("utf-8"))
        if fmt == "dataset-json":
            from app.ingest.datasets_json import DatasetJsonIO
            dio = DatasetJsonIO()
            top_findings = dio.validate_top_level(doc)
            if not top_findings.empty and top_findings["severity"].eq("CRIT").any():
                return None, top_findings
            df, extra_findings, _ = dio.domain_to_df(doc, dataset.domain)
            return df, extra_findings
        else:
            # Plain JSON — attempt records-orient DataFrame
            df = pd.DataFrame(doc if isinstance(doc, list) else doc.get("records", []))
            return df, None

    log.warning("Unknown file format %r for dataset %s", fmt, dataset.id)
    return None, None


def _persist_findings(
    db,
    combined: pd.DataFrame,
    job,
) -> list:
    """Insert Finding rows for every row in the findings DataFrame."""
    from app.models.finding import Finding

    if combined.empty:
        return []

    rows = []
    for _, row in combined.iterrows():
        f = Finding(
            tenant_id=job.tenant_id,
            job_id=job.id,
            study_id=job.study_id,
            finding_type=str(row.get("finding_type", "SDTM_RULE")),
            rule_id=str(row.get("rule_id", "")),
            severity=str(row.get("severity", "LOW")),
            domain=str(row.get("domain", "")),
            field=str(row.get("field", "")),
            message=str(row.get("message", "")),
            row_index=int(row.get("row_index", -1)),
            usubjid=str(row.get("usubjid", "")) or None,
            evidence=str(row.get("evidence", "")) or None,
        )
        db.add(f)
        rows.append(f)
    return rows


def _index_findings_es(findings: list, settings) -> None:
    """Bulk-index findings into Elasticsearch."""
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk

    es = Elasticsearch(settings.ELASTICSEARCH_URL)
    actions = [
        {
            "_index": settings.ELASTICSEARCH_INDEX_FINDINGS,
            "_id":    str(f.id),
            "_source": {
                "tenant_id":    str(f.tenant_id),
                "job_id":       str(f.job_id),
                "study_id":     str(f.study_id),
                "finding_type": f.finding_type,
                "rule_id":      f.rule_id,
                "severity":     f.severity,
                "domain":       f.domain,
                "field":        f.field,
                "message":      f.message,
                "row_index":    f.row_index,
                "usubjid":      f.usubjid,
                "evidence":     f.evidence,
                "status":       f.status,
                "created_at":   f.created_at.isoformat() if f.created_at else None,
            },
        }
        for f in findings
    ]
    bulk(es, actions)


def _build_summary(combined: pd.DataFrame) -> dict:
    if combined.empty:
        return {"total": 0, "CRIT": 0, "HIGH": 0, "MED": 0, "LOW": 0}
    return {
        "total": len(combined),
        **{sev: int((combined["severity"] == sev).sum()) for sev in ["CRIT", "HIGH", "MED", "LOW"]},
    }
