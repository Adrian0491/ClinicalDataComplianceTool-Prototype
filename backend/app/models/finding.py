from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TenantScopedMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.validation import ValidationJob


class Finding(Base, TimestampMixin, TenantScopedMixin):
    """
    One validation finding (a single rule violation or anomaly).

    PostgreSQL is the canonical store; Elasticsearch holds a mirror
    for real-time dashboard aggregation queries.
    """
    __tablename__ = "findings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("validation_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    study_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("studies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Mirror of the app.validators.schemas FINDINGS_COLUMNS structure
    finding_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # SDTM_RULE | CROSS_DOMAIN | DATASET_JSON | ANOMALY
    rule_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(
        String(10), nullable=False, index=True
    )  # CRIT | HIGH | MED | LOW
    domain: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    field: Mapped[str] = mapped_column(String(255), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    row_index: Mapped[int] = mapped_column(Integer, nullable=False, default=-1)
    usubjid: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Resolution workflow
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="open", index=True
    )  # open | resolved | waived
    resolved_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    resolution_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    job:      Mapped[ValidationJob] = relationship("ValidationJob", back_populates="findings")
    resolver: Mapped[User | None]   = relationship("User", foreign_keys=[resolved_by])

    def __repr__(self) -> str:
        return (
            f"<Finding id={self.id} rule_id={self.rule_id!r} "
            f"severity={self.severity!r} status={self.status!r}>"
        )
