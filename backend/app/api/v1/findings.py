"""
/api/v1/validation/findings/*  — list and resolve findings.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import case
from sqlalchemy.orm import Session

from app.core.rbac import get_current_user, require_validator, require_viewer
from app.database import get_db
from app.models.finding import Finding
from app.models.user import User
from app.schemas.finding import FindingResponse, FindingResolveRequest

router = APIRouter(prefix="/validation/findings", tags=["findings"])

_VALID_STATUSES   = {"open", "resolved", "waived"}
_VALID_SEVERITIES = {"CRIT", "HIGH", "MED", "LOW"}


@router.get("", response_model=List[FindingResponse])
def list_findings(
    study_id:   Optional[uuid.UUID] = Query(None),
    job_id:     Optional[uuid.UUID] = Query(None),
    domain:     Optional[str]       = Query(None),
    severity:   Optional[str]       = Query(None),
    find_status: Optional[str]      = Query(None, alias="status"),
    usubjid:    Optional[str]       = Query(None),
    offset:     int                 = Query(0, ge=0),
    limit:      int                 = Query(100, ge=1, le=1000),
    db:           Session = Depends(get_db),
    current_user: User    = Depends(require_viewer),
):
    q = db.query(Finding).filter(Finding.tenant_id == current_user.tenant_id)

    if study_id:    q = q.filter(Finding.study_id == study_id)
    if job_id:      q = q.filter(Finding.job_id   == job_id)
    if domain:      q = q.filter(Finding.domain   == domain.upper())
    if severity:    q = q.filter(Finding.severity == severity.upper())
    if find_status: q = q.filter(Finding.status   == find_status.lower())
    if usubjid:     q = q.filter(Finding.usubjid  == usubjid)

    # True severity priority, not alphabetical — plain `.order_by(Finding.severity)`
    # would sort 'LOW' ahead of 'MED' since L < M.
    severity_rank = case(
        (Finding.severity == "CRIT", 0),
        (Finding.severity == "HIGH", 1),
        (Finding.severity == "MED", 2),
        (Finding.severity == "LOW", 3),
        else_=4,
    )
    return (
        q.order_by(severity_rank, Finding.created_at)
        .offset(offset)
        .limit(limit)
        .all()
    )


@router.get("/{finding_id}", response_model=FindingResponse)
def get_finding(
    finding_id:   uuid.UUID,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(require_viewer),
):
    f = db.query(Finding).filter(
        Finding.id        == finding_id,
        Finding.tenant_id == current_user.tenant_id,
    ).first()
    if f is None:
        raise HTTPException(status_code=404, detail="Finding not found.")
    return f


@router.patch("/{finding_id}", response_model=FindingResponse)
def resolve_finding(
    finding_id:   uuid.UUID,
    body:         FindingResolveRequest,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(require_validator),
):
    """Resolve or waive a finding (requires validator role or above)."""
    if body.status not in {"resolved", "waived"}:
        raise HTTPException(status_code=400, detail="status must be 'resolved' or 'waived'.")

    f = db.query(Finding).filter(
        Finding.id        == finding_id,
        Finding.tenant_id == current_user.tenant_id,
    ).first()
    if f is None:
        raise HTTPException(status_code=404, detail="Finding not found.")
    if f.status != "open":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Finding is already '{f.status}'.",
        )

    f.status          = body.status
    f.resolved_by     = current_user.id
    f.resolved_at     = datetime.now(timezone.utc)
    f.resolution_note = body.resolution_note
    db.commit()
    return f
