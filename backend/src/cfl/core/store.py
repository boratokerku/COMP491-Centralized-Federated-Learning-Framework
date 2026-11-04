"""Storage for model and update files (safetensors bytes), addressed by relative keys such as
"<run_id>/global_model_v3.safetensors".

PostgreSQL stores only keys, checksums and sizes; the bytes live here. The simulation uses
InMemoryModelStore, the distributed mode FileModelStore, behind the same interface.
"""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath


class ModelStore(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes) -> None:
        """Store `data` atomically: readers see either the old file or the complete new one."""

    @abstractmethod
    def get(self, key: str) -> bytes:
        """Return the stored bytes; raise KeyError if the key does not exist."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...


def _check_key(key: str) -> PurePosixPath:
    path = PurePosixPath(key)
    if not key or path.is_absolute() or ".." in path.parts or "\\" in key:
        raise ValueError(f"invalid store key {key!r}")
    return path


class InMemoryModelStore(ModelStore):
    def __init__(self):
        self._files: dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> None:
        _check_key(key)
        self._files[key] = bytes(data)

    def get(self, key: str) -> bytes:
        _check_key(key)
        return self._files[key]

    def exists(self, key: str) -> bool:
        _check_key(key)
        return key in self._files


class FileModelStore(ModelStore):
    """Files under `root`. Writes go to a temporary file in the same directory, are fsynced and
    then renamed over the target, so a crash never leaves a half-written model behind."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root.joinpath(*_check_key(key).parts)

    def put(self, key: str, data: bytes) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def get(self, key: str) -> bytes:
        path = self._path(key)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            raise KeyError(key) from None

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()
