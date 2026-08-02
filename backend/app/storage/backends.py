"""
Storage abstraction with two backends:
  - LocalBackend  : writes to the local filesystem (development / testing)
  - GCSBackend    : writes to Google Cloud Storage (production)

The active backend is selected by Settings.STORAGE_BACKEND.
"""
from __future__ import annotations

import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    @abstractmethod
    def save(self, data: bytes, tenant_id: str, filename: str) -> str:
        """
        Persist `data` and return the storage URI string
        (e.g. file:///... or gs://...).
        """
        ...

    @abstractmethod
    def read(self, uri: str) -> bytes:
        """Read the file at `uri` and return its raw bytes."""
        ...

    @abstractmethod
    def delete(self, uri: str) -> None:
        """Delete the file at `uri`. Silent if already absent."""
        ...


# ---------------------------------------------------------------------------
# Local filesystem backend (default for local dev / tests)
# ---------------------------------------------------------------------------

class LocalBackend(StorageBackend):
    def __init__(self, base_path: str = "./storage") -> None:
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)

    def _path_for(self, tenant_id: str, filename: str) -> Path:
        tenant_dir = self.base / tenant_id
        tenant_dir.mkdir(parents=True, exist_ok=True)
        return tenant_dir / filename

    def save(self, data: bytes, tenant_id: str, filename: str) -> str:
        # Prefix with a UUID to avoid collisions
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        dest = self._path_for(tenant_id, unique_name)
        dest.write_bytes(data)
        return f"file://{dest.resolve()}"

    def read(self, uri: str) -> bytes:
        path = Path(uri.replace("file://", ""))
        return path.read_bytes()

    def delete(self, uri: str) -> None:
        path = Path(uri.replace("file://", ""))
        if path.exists():
            path.unlink()


# ---------------------------------------------------------------------------
# Google Cloud Storage backend (production)
# ---------------------------------------------------------------------------

class GCSBackend(StorageBackend):
    def __init__(self, bucket_name: str, project_id: str | None = None) -> None:
        from google.cloud import storage as gcs
        self._client = gcs.Client(project=project_id)
        self._bucket = self._client.bucket(bucket_name)

    def save(self, data: bytes, tenant_id: str, filename: str) -> str:
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        blob_name = f"{tenant_id}/{unique_name}"
        blob = self._bucket.blob(blob_name)
        blob.upload_from_string(data)
        return f"gs://{self._bucket.name}/{blob_name}"

    def read(self, uri: str) -> bytes:
        # uri = gs://bucket/path/to/file
        if not uri.startswith("gs://"):
            raise ValueError(
                f"GCSBackend cannot read non-gs:// URI: {uri!r} "
                "(dataset was likely saved under a different STORAGE_BACKEND)."
            )
        parts = uri.replace("gs://", "").split("/", 1)
        blob = self._bucket.blob(parts[1])
        return blob.download_as_bytes()

    def delete(self, uri: str) -> None:
        if not uri.startswith("gs://"):
            raise ValueError(
                f"GCSBackend cannot delete non-gs:// URI: {uri!r} "
                "(dataset was likely saved under a different STORAGE_BACKEND)."
            )
        parts = uri.replace("gs://", "").split("/", 1)
        blob = self._bucket.blob(parts[1])
        blob.delete()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_storage() -> StorageBackend:
    """Return the configured storage backend (singleton per process)."""
    from app.config import get_settings
    s = get_settings()
    if s.STORAGE_BACKEND == "gcs":
        return GCSBackend(bucket_name=s.GCS_BUCKET_NAME, project_id=s.GCS_PROJECT_ID)
    return LocalBackend(base_path=s.STORAGE_LOCAL_PATH)
