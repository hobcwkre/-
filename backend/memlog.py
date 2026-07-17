"""Lightweight RSS memory logging around the hot paths.

Prints `[mem] <tag>: <RSS> MB` to stdout (shows up in Render logs) so OOM
hunts can point at the exact step. Falls back to a no-op when psutil is
missing, so the app never depends on it to run.
"""
from __future__ import annotations

import os

try:
    import psutil

    _proc = psutil.Process(os.getpid())

    def log_mem(tag: str) -> None:
        rss_mb = _proc.memory_info().rss / 1048576
        print(f"[mem] {tag}: {rss_mb:.1f} MB", flush=True)

except ImportError:  # pragma: no cover

    def log_mem(tag: str) -> None:  # noqa: ARG001
        pass
