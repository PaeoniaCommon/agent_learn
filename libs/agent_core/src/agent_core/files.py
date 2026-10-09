"""Safe shared-file writes: atomic replace + cross-process file locks."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from filelock import FileLock


def atomic_write(path: Path, content: str) -> None:
    """Write via a temp file and os.replace, so readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def lock_for(path: Path, root: Path, lock_dir: Path, timeout: float = 30) -> FileLock:
    """A lock for `path`, kept in `lock_dir` (not next to the data files)."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    rel = path.relative_to(root).as_posix().replace("/", "__")
    return FileLock(str(lock_dir / f"{rel}.lock"), timeout=timeout)
