"""In-memory store for retrieved DataFrames (LRU). DataFrames never enter graph state."""

from __future__ import annotations

import secrets
import threading
from collections import OrderedDict
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class DatasetRecord(BaseModel):
    id: str
    view: str
    params: dict[str, Any]
    retrieved_at: datetime  # UTC
    n_rows: int
    n_cols: int
    columns: list[str]


class DataStore:
    def __init__(self, max_entries: int = 50):
        self.max_entries = max_entries
        self._data: OrderedDict[str, tuple[Any, DatasetRecord]] = OrderedDict()
        self._lock = threading.Lock()

    def _new_id(self) -> str:
        while True:
            i = "ds_" + secrets.token_hex(3)
            if i not in self._data:
                return i

    def put(self, df, view: str, params: dict, retrieved_at: datetime) -> str:
        with self._lock:
            dataset_id = self._new_id()
            rec = DatasetRecord(
                id=dataset_id, view=view, params=dict(params), retrieved_at=retrieved_at,
                n_rows=int(df.shape[0]), n_cols=int(df.shape[1]), columns=[str(c) for c in df.columns],
            )
            self._data[dataset_id] = (df, rec)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
            return dataset_id

    def get(self, dataset_id: str):
        with self._lock:
            df, _ = self._data[dataset_id]
            self._data.move_to_end(dataset_id)
            return df

    def meta(self, dataset_id: str) -> DatasetRecord:
        with self._lock:
            return self._data[dataset_id][1]

    def list(self) -> list[DatasetRecord]:
        with self._lock:
            return [rec for _, rec in self._data.values()]

    def __contains__(self, dataset_id: str) -> bool:
        return dataset_id in self._data

    def __len__(self) -> int:
        return len(self._data)


_DEFAULT: DataStore | None = None


def get_data_store(max_entries: int = 50) -> DataStore:
    """Process-wide singleton shared with the host app."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = DataStore(max_entries)
    return _DEFAULT
