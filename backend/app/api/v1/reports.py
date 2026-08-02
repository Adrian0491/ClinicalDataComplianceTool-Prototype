"""
/api/v1/reports/*  — generate, list, sign, and download compliance reports.
"""
from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from typing import List
from xml.sax.saxutils import escape as _xml_escape

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import case
from sqlalchemy.orm import Session

from app.core.rbac import require_admin, require_validator, require_viewer
from app.database import get_db
from app.models.finding import Finding
from app.models.report import ComplianceReport
from app.models.user import User
from app.models.validation import ValidationJob
from app.schemas.report import ComplianceReportResponse, ReportGenerateRequest

router = APIRouter(prefix="/reports", tags=["reports"])


def _truncate_message(msg: str, limit: int = 90) -> str:
    """
    Truncate a finding message for the PDF table, but never at the cost of
    the trailing "(found: '...')" clause — that's the row-specific value
    the reader actually needs; the static rule text ahead of it is the
    same on every row and safe to cut instead.
    """
    if not msg or len(msg) <= limit:
        return msg
    marker = " (found: "
    idx = msg.rfind(marker)
    if idx == -1:
        return msg[:limit].rstrip() + "…"
    suffix = msg[idx:]
    avail = max(limit - len(suffix), 20)
    return msg[:avail].rstrip() + "…" + suffix


def _generate_pdf(job: ValidationJob, findings: list, report_type: str, tenant_name: str) -> bytes:
    """Generate a simple PDF report from validation findings."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        # Fallback: generate a simple text-based report as bytes
        return _generate_text_report(job, findings, report_type, tenant_name)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle('Title', parent=styles['Title'], textColor=colors.HexColor('#1A3C6E'), fontSize=18)
    story.append(Paragraph("Integris Clinical Platform", title_style))
    story.append(Paragraph("SDTM Validation Compliance Report", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))

    # Report metadata
    meta = [
        ["Report Type:", report_type.replace("_", " ").title()],
        ["Study ID:", str(job.study_id)],
        ["Job ID:", str(job.id)],
        ["Rule Profile:", job.rule_profile],
        ["Status:", job.status],
        ["Generated:", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")],
        ["Tenant:", tenant_name],
    ]
    if job.started_at and job.completed_at:
        duration_ms = (job.completed_at - job.started_at).total_seconds()
        meta.append(["Duration:", f"{duration_ms:.1f}s"])

    meta_table = Table(meta, colWidths=[1.5*inch, 5*inch])
    meta_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1A3C6E')),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#F5F7FA')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.3*inch))

    # Summary
    story.append(Paragraph("Findings Summary", styles['Heading2']))
    sev_counts = {"CRIT": 0, "HIGH": 0, "MED": 0, "LOW": 0}
    domain_counts: dict[str, int] = {}
    for f in findings:
        sev_counts[f.severity] = sev_counts.get(f.severity, 0) + 1
        domain_counts[f.domain] = domain_counts.get(f.domain, 0) + 1

    summary_data = [["Severity", "Count"], ["CRITICAL", str(sev_counts["CRIT"])],
                    ["HIGH", str(sev_counts["HIGH"])], ["MEDIUM", str(sev_counts["MED"])],
                    ["LOW", str(sev_counts["LOW"])], ["TOTAL", str(len(findings))]]
    summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A3C6E')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#E8ECF0')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor('#F5F7FA')]),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))

    # Findings detail
    if findings:
        story.append(Paragraph("Findings Detail", styles['Heading2']))

        # Cell content is rendered as Paragraph flowables (not raw strings) so
        # long Value/Message text wraps within its column instead of painting
        # over the next cell — plain strings in a reportlab Table don't wrap,
        # they just overflow, which is what caused the garbled overlap before.
        cell_style = ParagraphStyle('FindingsCell', parent=styles['Normal'], fontSize=7, leading=9)
        header_style = ParagraphStyle('FindingsHeader', parent=cell_style, textColor=colors.white, fontName='Helvetica-Bold')

        def _cell(text: str, style=cell_style) -> Paragraph:
            return Paragraph(_xml_escape(text or ""), style)

        findings_data = [[_cell(h, header_style) for h in ("Domain", "Rule ID", "Severity", "Field", "Value", "Message")]]
        for f in findings[:200]:  # limit to 200 rows
            findings_data.append([
                _cell(f.domain or ""),
                _cell(f.rule_id or ""),
                _cell(f.severity or ""),
                _cell(f.field or ""),
                _cell((f.evidence or "")[:120]),
                _cell(_truncate_message(f.message or "", limit=200)),
            ])
        findings_table = Table(findings_data, colWidths=[0.6*inch, 0.95*inch, 0.55*inch, 0.75*inch, 1*inch, 2.65*inch], repeatRows=1)
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A3C6E')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('PADDING', (0, 0), (-1, -1), 4),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F7FA')]),
        ]))
        story.append(findings_table)

    doc.build(story)
    return buffer.getvalue()


def _generate_text_report(job: ValidationJob, findings: list, report_type: str, tenant_name: str) -> bytes:
    """Fallback plain text report."""
    lines = [
        "INTEGRIS CLINICAL PLATFORM",
        "SDTM Validation Compliance Report",
        "=" * 60,
        f"Report Type: {report_type}",
        f"Study ID: {job.study_id}",
        f"Job ID: {job.id}",
        f"Rule Profile: {job.rule_profile}",
        f"Status: {job.status}",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Tenant: {tenant_name}",
        "",
        "FINDINGS SUMMARY",
        "-" * 40,
        f"Total Findings: {len(findings)}",
    ]
    sev_counts = {"CRIT": 0, "HIGH": 0, "MED": 0, "LOW": 0}
    for f in findings:
        sev_counts[f.severity] = sev_counts.get(f.severity, 0) + 1
    for sev, count in sev_counts.items():
        lines.append(f"{sev}: {count}")
    lines.extend(["", "FINDINGS DETAIL", "-" * 40])
    for f in findings:
        lines.append(f"[{f.severity}] {f.domain} | {f.rule_id} | {f.field}={f.evidence or ''} | {f.message}")
    return "\n".join(lines).encode("utf-8")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("", response_model=ComplianceReportResponse, status_code=status.HTTP_201_CREATED)
def generate_report(
    body: ReportGenerateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_validator),
):
    """Generate a compliance report for a completed validation job."""
    job = db.query(ValidationJob).filter(
        ValidationJob.id == body.job_id,
        ValidationJob.tenant_id == current_user.tenant_id,
    ).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Validation job not found.")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Validation job is not completed.")

    # Group by domain, then true severity priority (not alphabetical —
    # 'LOW' would otherwise sort before 'MED') — an unordered query previously
    # left findings in arbitrary DB row order, interleaving domains/severities.
    severity_rank = case(
        (Finding.severity == "CRIT", 0),
        (Finding.severity == "HIGH", 1),
        (Finding.severity == "MED", 2),
        (Finding.severity == "LOW", 3),
        else_=4,
    )
    findings = (
        db.query(Finding)
        .filter(Finding.job_id == job.id)
        .order_by(Finding.domain, severity_rank, Finding.row_index)
        .all()
    )

    # Get tenant name
    tenant_name = str(current_user.tenant_id)
    try:
        from app.models.tenant import Tenant
        tenant = db.get(Tenant, current_user.tenant_id)
        if tenant:
            tenant_name = tenant.name
    except Exception:
        pass

    # Generate PDF
    pdf_bytes = _generate_pdf(job, findings, body.report_type, tenant_name)

    # Save to storage
    from app.storage.backends import get_storage
    storage = get_storage()
    filename = f"report_{job.id}_{body.report_type}.pdf"
    storage_uri = storage.save(pdf_bytes, str(current_user.tenant_id), filename)

    # Create report record
    report = ComplianceReport(
        tenant_id=current_user.tenant_id,
        job_id=job.id,
        study_id=job.study_id,
        report_type=body.report_type,
        storage_uri=storage_uri,
        generated_by=current_user.id,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


@router.get("", response_model=List[ComplianceReportResponse])
def list_reports(
    study_id: uuid.UUID | None = None,
    job_id: uuid.UUID | None = None,
    offset: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    q = db.query(ComplianceReport).filter(
        ComplianceReport.tenant_id == current_user.tenant_id
    )
    if study_id:
        q = q.filter(ComplianceReport.study_id == study_id)
    if job_id:
        q = q.filter(ComplianceReport.job_id == job_id)
    return q.order_by(ComplianceReport.created_at.desc()).offset(offset).limit(limit).all()


@router.get("/{report_id}", response_model=ComplianceReportResponse)
def get_report(
    report_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    report = db.query(ComplianceReport).filter(
        ComplianceReport.id == report_id,
        ComplianceReport.tenant_id == current_user.tenant_id,
    ).first()
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report


@router.delete("/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_report(
    report_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Delete a compliance report (admin only).

    Signed reports are 21 CFR Part 11 records and are intentionally
    immutable — they can't be deleted, only regenerated as a new report.
    """
    report = db.query(ComplianceReport).filter(
        ComplianceReport.id == report_id,
        ComplianceReport.tenant_id == current_user.tenant_id,
    ).first()
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    if report.signed_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Signed reports cannot be deleted (21 CFR Part 11 record).",
        )

    if report.storage_uri:
        from app.storage.backends import get_storage
        try:
            get_storage().delete(report.storage_uri)
        except Exception:
            pass  # file already gone / unreachable — don't block deleting the record

    db.delete(report)
    db.commit()


@router.post("/{report_id}/sign", response_model=ComplianceReportResponse)
def sign_report(
    report_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_validator),
):
    """Apply electronic signature (21 CFR Part 11)."""
    report = db.query(ComplianceReport).filter(
        ComplianceReport.id == report_id,
        ComplianceReport.tenant_id == current_user.tenant_id,
    ).first()
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    if report.signed_at:
        raise HTTPException(status_code=400, detail="Report is already signed.")

    report.signed_at = datetime.now(timezone.utc)
    report.signed_by = current_user.id
    db.commit()
    db.refresh(report)
    return report


@router.get("/{report_id}/download")
def download_report(
    report_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    """Download the report PDF via signed URL."""
    report = db.query(ComplianceReport).filter(
        ComplianceReport.id == report_id,
        ComplianceReport.tenant_id == current_user.tenant_id,
    ).first()
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    if not report.storage_uri:
        raise HTTPException(status_code=404, detail="Report file not yet generated.")

    from app.storage.backends import get_storage
    from fastapi.responses import RedirectResponse
    import datetime

    storage = get_storage()

    # Try GCS signed URL first
    try:
        parts = report.storage_uri.replace("gs://", "").split("/", 1)
        blob = storage._bucket.blob(parts[1])
        signed_url = blob.generate_signed_url(
            expiration=datetime.timedelta(minutes=15),
            method="GET",
            version="v4",
        )
        return RedirectResponse(url=signed_url)
    except Exception:
        # Fallback: stream the file
        pdf_bytes = storage.read(report.storage_uri)
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=report_{report_id}.pdf"},
        )