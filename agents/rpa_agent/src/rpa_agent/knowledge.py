"""Knowledge files: knowledge/views.md (one line per view) and knowledge/params/{name}.md.

Files are shared by all users and processes: reads reload on mtime change, writes are
atomic and hold a file lock.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from filelock import FileLock

from agent_core.files import atomic_write, lock_for

VIEWS_HEADER = "# Views\n<!-- managed by rpa_agent; one line per view; keep descriptions short -->\n"
PARAM_FIELDS = ["description", "kind", "date_format", "format", "pattern", "valid_examples", "invalid_examples"]
LIST_FIELDS = {"valid_examples", "invalid_examples"}
_SAFE = re.compile(r"[^A-Za-z0-9_.\-]+")


@dataclass
class ParamKnowledge:
    name: str
    description: str = ""
    kind: str = ""            # "date" | "text" | ""
    date_format: str = ""     # strftime
    format: str = ""
    pattern: str = ""
    valid_examples: list[str] = field(default_factory=list)
    invalid_examples: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any(getattr(self, f) for f in PARAM_FIELDS)

    def render(self) -> str:
        lines = [f"# {self.name}"]
        for f in PARAM_FIELDS:
            v = getattr(self, f)
            if f in LIST_FIELDS:
                v = "; ".join(v)
            if v:
                lines.append(f"- {f}: {v}")
        return "\n".join(lines) + "\n"

    def prompt_line(self) -> str:
        """Compact single-line form for prompts."""
        parts = [self.name]
        for f in ("description", "format", "date_format"):
            v = getattr(self, f)
            if v:
                parts.append(f"{f}={v}" if f != "description" else v)
        if self.valid_examples:
            parts.append("e.g. " + ", ".join(self.valid_examples))
        return " | ".join(parts)


def _parse_param(name: str, text: str) -> ParamKnowledge:
    pk = ParamKnowledge(name=name)
    for line in text.splitlines():
        m = re.match(r"^-\s*([a-z_]+):\s?(.*)$", line.strip())
        if not m or m.group(1) not in PARAM_FIELDS:
            continue
        key, val = m.group(1), m.group(2).strip()
        if key in LIST_FIELDS:
            setattr(pk, key, [v.strip() for v in val.split(";") if v.strip()])
        else:
            setattr(pk, key, val)
    return pk


class KnowledgeBase:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.params_dir = self.root / "params"
        self.log_dir = self.root / "_log"
        self.lock_dir = self.root / "_locks"
        self.views_path = self.root / "views.md"
        self.params_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[Path, tuple[float, object]] = {}
        self._mem_lock = threading.Lock()

    # -- locking / caching ----------------------------------------------------
    def lock(self, path: Path) -> FileLock:
        return lock_for(path, self.root, self.lock_dir)

    def _read_cached(self, path: Path, parse):
        try:
            mtime = path.stat().st_mtime_ns
        except FileNotFoundError:
            return parse("")
        with self._mem_lock:
            hit = self._cache.get(path)
            if hit and hit[0] == mtime:
                return hit[1]
        value = parse(path.read_text(encoding="utf-8"))
        with self._mem_lock:
            self._cache[path] = (mtime, value)
        return value

    # -- views ------------------------------------------------------------------
    @staticmethod
    def _parse_views(text: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in text.splitlines():
            m = re.match(r"^-\s*([^:]+?):\s?(.*)$", line.strip())
            if m:
                out[m.group(1).strip()] = m.group(2).strip()
        return out

    def views(self) -> dict[str, str]:
        return dict(self._read_cached(self.views_path, self._parse_views))

    def _write_views(self, views: dict[str, str]) -> None:
        body = "".join(f"- {k}: {v}".rstrip() + "\n" for k, v in views.items())
        atomic_write(self.views_path, VIEWS_HEADER + body)

    def sync_views(self, view_names: list[str]) -> None:
        """Add new views (empty description), drop views no longer in RPA_VIEWS."""
        with self.lock(self.views_path):
            current = self._parse_views(self.views_path.read_text(encoding="utf-8")) if self.views_path.exists() else {}
            synced = {v: current.get(v, "") for v in view_names}
            if synced != current or not self.views_path.exists():
                self._write_views(synced)

    def set_view_description(self, view: str, description: str) -> None:
        with self.lock(self.views_path):
            current = self._parse_views(self.views_path.read_text(encoding="utf-8")) if self.views_path.exists() else {}
            current[view] = description.replace("\n", " ").strip()
            self._write_views(current)

    def views_prompt(self, names: list[str]) -> str:
        v = self.views()
        return "\n".join(f"- {n}: {v.get(n, '')}".rstrip() for n in names)

    # -- params -------------------------------------------------------------------
    def param_path(self, name: str) -> Path:
        return self.params_dir / f"{_SAFE.sub('_', name)}.md"

    def param(self, name: str) -> ParamKnowledge:
        path = self.param_path(name)
        pk = self._read_cached(path, lambda text: _parse_param(name, text))
        return ParamKnowledge(**{**pk.__dict__, "valid_examples": list(pk.valid_examples),
                                 "invalid_examples": list(pk.invalid_examples)})

    def ensure_param(self, name: str) -> None:
        path = self.param_path(name)
        if not path.exists():
            with self.lock(path):
                if not path.exists():
                    atomic_write(path, ParamKnowledge(name=name).render())

    def save_param(self, pk: ParamKnowledge) -> None:
        path = self.param_path(pk.name)
        with self.lock(path):
            atomic_write(path, pk.render())

    def update_param(self, name: str, mutate) -> ParamKnowledge | None:
        """Read-modify-write under the file lock. `mutate(pk) -> bool changed`."""
        path = self.param_path(name)
        with self.lock(path):
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            pk = _parse_param(name, text)
            if mutate(pk):
                atomic_write(path, pk.render())
                return pk
        return None

    def params_prompt(self, names: list[str]) -> str:
        lines = []
        for n in names:
            pk = self.param(n)
            if not pk.is_empty():
                lines.append("- " + pk.prompt_line())
        return "\n".join(lines)
