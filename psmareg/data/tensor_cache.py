"""Atomic on-disk tensor cache shared by the datasets."""

from __future__ import annotations

import os
from pathlib import Path
import uuid
from typing import Any

import torch


class TensorCache:
    """Store tensors or tensor dictionaries under caller-provided keys."""

    def __init__(self, directory: Path) -> None:
        """Create the cache directory if needed.

        Args:
            directory: Root of one dataset cache.
        """
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        """Return the entry path of one dataset-defined key."""
        return self.directory / f"{key}.pt"

    @staticmethod
    def load(path: Path) -> Any:
        """Load a tensor or tensor dictionary from disk on the CPU."""
        return torch.load(path, map_location="cpu", weights_only=True)

    @staticmethod
    def save(path: Path, value: Any) -> None:
        """Publish a cache entry atomically, preserving earlier valid data.

        Args:
            path: Destination cache entry.
            value: Tensor or dictionary to serialize with Torch.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(
            f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            torch.save(value, temporary_path)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
