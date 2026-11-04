"""Safe local filesystem storage for uploaded plant images."""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from pathlib import Path

from settings import settings

_ALLOWED_MIME_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8\xff"
_WEBP_RIFF = b"RIFF"
_WEBP_MARKER = b"WEBP"


class StorageError(Exception):
    """Raised when an image cannot be validated or accessed safely."""


def _normalise_session_id(session_id: uuid.UUID | str) -> str:
    try:
        return str(uuid.UUID(str(session_id)))
    except (TypeError, ValueError) as exc:
        raise StorageError("Invalid session id.") from exc


def _extension_for_mime_type(mime_type: str) -> str:
    extension = _ALLOWED_MIME_TYPES.get(mime_type)
    if extension is None:
        raise StorageError("Unsupported image type.")
    return extension


def _has_valid_signature(data: bytes, mime_type: str) -> bool:
    if mime_type == "image/png":
        return data.startswith(_PNG_SIGNATURE)
    if mime_type == "image/jpeg":
        return data.startswith(_JPEG_SIGNATURE)
    if mime_type == "image/webp":
        return (
            len(data) >= 12
            and data[:4] == _WEBP_RIFF
            and data[8:12] == _WEBP_MARKER
        )
    return False


def _validate_image(data: bytes, mime_type: str) -> str:
    extension = _extension_for_mime_type(mime_type)
    if not data:
        raise StorageError("Image file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise StorageError("Image exceeds the maximum upload size.")
    if not _has_valid_signature(data, mime_type):
        raise StorageError("Image content does not match its declared type.")
    return extension


def _resolve_storage_path(storage_key: str) -> Path:
    if not isinstance(storage_key, str) or not storage_key.strip():
        raise StorageError("Invalid storage key.")

    try:
        relative = Path(storage_key)
    except (TypeError, ValueError) as exc:
        raise StorageError("Invalid storage key.") from exc

    if relative.is_absolute() or relative.anchor:
        raise StorageError("Invalid storage key.")
    if ".." in relative.parts:
        raise StorageError("Invalid storage key.")

    try:
        root = settings.uploads_dir.resolve()
        target = (root / relative).resolve()
    except OSError as exc:
        raise StorageError("Invalid storage key.") from exc

    if not target.is_relative_to(root):
        raise StorageError("Invalid storage key.")
    return target


def save_image(
    session_id: uuid.UUID | str,
    data: bytes,
    mime_type: str,
    original_filename: str | None = None,
) -> dict[str, str | int]:
    """Validate, store and describe an uploaded image."""
    extension = _validate_image(data, mime_type)
    session_key = _normalise_session_id(session_id)

    storage_key = f"sessions/{session_key}/{uuid.uuid4().hex}{extension}"
    final_path = _resolve_storage_path(storage_key)
    session_dir = final_path.parent
    session_dir.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        dir=session_dir,
        prefix=".upload-",
        suffix=".part",
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temp_path, final_path)
    except Exception as exc:
        raise StorageError("Could not save the image.") from exc
    finally:
        temp_path.unlink(missing_ok=True)

    return {
        "storage_key": storage_key,
        "mime_type": mime_type,
        "byte_size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def read_image(storage_key: str) -> bytes:
    """Read an image, refusing any path outside the uploads directory."""
    path = _resolve_storage_path(storage_key)
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise StorageError("Image not found.") from exc
    except OSError as exc:
        raise StorageError("Could not read the image.") from exc


def delete_image(storage_key: str) -> None:
    """Delete an image if present, refusing any path outside uploads."""
    path = _resolve_storage_path(storage_key)
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise StorageError("Could not delete the image.") from exc
