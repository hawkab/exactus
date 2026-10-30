"""Require useful provenance; declared metadata is never called an attestation."""

import json
from pathlib import Path
import subprocess
import time
import urllib.request


class QualificationBusy(RuntimeError):
    """No model call is allowed when the advisory gate cannot prove idle."""


class QualificationGate:
    """Advisory idle gate, not an exclusive reservation of the shared backend."""

    def __init__(self, capacity_url, metrics_url):
        self.capacity_url, self.metrics_url = capacity_url, metrics_url
        self.elapsed_s = 0.0

    def __call__(self):
        try:
            from .bench_external_computers import idle
        except ImportError:
            from bench_external_computers import idle
        started = time.monotonic()
        try:
            with urllib.request.urlopen(self.capacity_url, timeout=5) as response:
                capacity = json.load(response)
            with urllib.request.urlopen(self.metrics_url, timeout=5) as response:
                raw = response.read(1048577)
            if (
                type(capacity) is not dict
                or len(raw) > 1048576
                or not idle(capacity, raw.decode("utf-8"))
            ):
                raise QualificationBusy("qualification refused: backend is busy")
        except (OSError, ValueError) as exc:
            raise QualificationBusy("qualification endpoints unavailable") from exc
        finally:
            self.elapsed_s += time.monotonic() - started


def source_provenance(root):
    """Never attribute an extracted archive to its parent's unrelated git repo."""
    root = Path(root).resolve()
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    if top.returncode or Path(top.stdout.strip()).resolve() != root:
        return {
            "source_sha": None,
            "dirty": None,
            "git_scope": "archive_or_nested_directory",
        }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True
    )
    return {
        "source_sha": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout) if status.returncode == 0 else None,
        "git_scope": "source_root",
    }


def load_runtime(path, arm_names=None):
    if path is None:
        return None
    raw = json.loads(Path(path).read_text())
    profiles = raw.get("arms") if isinstance(raw, dict) else None
    if isinstance(raw, dict) and "arms" in raw:
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError("non-empty runtime arms required")
        if arm_names is not None and set(profiles) != set(arm_names):
            raise ValueError("runtime arm names must exactly match benchmark arms")
    profiles = profiles.values() if isinstance(profiles, dict) else [raw]
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ValueError("runtime manifest must be an object")
        for key in (
            "model_revision",
            "tokenizer_revision",
            "vllm_revision",
            "container_digest",
        ):
            value = profile.get(key)
            if (
                not isinstance(value, str)
                or not value.strip()
                or value.lower() in ("unknown", "todo")
            ):
                raise ValueError("runtime manifest requires concrete " + key)
        if not isinstance(profile.get("profile"), dict) or not profile["profile"]:
            raise ValueError("runtime manifest requires non-empty profile flags")
    return raw
