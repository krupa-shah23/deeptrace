"""
P-Fusion generic utility helpers.

Stdlib-only. No third-party dependencies.
No detector logic, no attribution rules, no p_audio/p_video imports.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def load_json(path: Path | str) -> dict[str, Any]:
    """Load and return a JSON file as a dictionary.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file exists but is not valid JSON, or the top-level
            value is not a JSON object (dict).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object at the top level of {path}, "
            f"got {type(data).__name__}"
        )
    return data


# ---------------------------------------------------------------------------
# Dictionary / object validation
# ---------------------------------------------------------------------------

def check_required_keys(data: dict[str, Any], required: Sequence[str]) -> list[str]:
    """Return a list of keys from *required* that are absent from *data*.

    An empty return value means all required keys are present.
    """
    return [k for k in required if k not in data]


def validate_required_keys(
    data: dict[str, Any],
    required: Sequence[str],
    source_label: str = "input",
) -> None:
    """Raise a ValueError listing every missing required key.

    Args:
        data: The dictionary to validate.
        required: Keys that must be present.
        source_label: Human-readable label used in the error message.

    Raises:
        ValueError: If one or more required keys are absent.
    """
    missing = check_required_keys(data, required)
    if missing:
        raise ValueError(
            f"{source_label} is missing required fields: {missing}"
        )


# ---------------------------------------------------------------------------
# Safe nested field access
# ---------------------------------------------------------------------------

_SENTINEL = object()


def get_nested(
    data: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """Safely traverse a nested dict using a sequence of keys.

    Returns *default* if any key is absent or if an intermediate value is
    not a dict.  Never raises KeyError.

    Example:
        get_nested(obj, "metadata", "encoder", default="")
    """
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, _SENTINEL)
        if current is _SENTINEL:
            return default
    return current


def has_nested(data: dict[str, Any], *keys: str) -> bool:
    """Return True if the nested key path resolves to a value that is not None."""
    sentinel = object()
    value = get_nested(data, *keys, default=sentinel)
    return value is not sentinel and value is not None


# ---------------------------------------------------------------------------
# SHA-256 file hashing
# ---------------------------------------------------------------------------

def sha256_file(path: Path | str, chunk_size: int = 65536) -> str:
    """Return the hex-encoded SHA-256 digest of a file's contents.

    Reads in chunks to avoid loading large files into memory at once.

    Args:
        path: Path to the file to hash.
        chunk_size: Read buffer size in bytes (default 64 KiB).

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found for hashing: {path}")

    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Deterministic formatting helpers
# ---------------------------------------------------------------------------

def format_confidence(value: Optional[float]) -> str:
    """Format a 0.0–1.0 confidence float as a percentage string.

    Returns "N/A" for None (missing / UNKNOWN confidence).

    Example:
        format_confidence(0.873)  → "87.3%"
        format_confidence(None)   → "N/A"
    """
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def format_mismatch_rate(value: Optional[float]) -> str:
    """Format a 0.0–1.0 mismatch rate as a percentage string.

    Returns "N/A" for None (missing data).

    Example:
        format_mismatch_rate(0.05)  → "5.0%"
        format_mismatch_rate(None)  → "N/A"
    """
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def pluralise(count: int, singular: str, plural: Optional[str] = None) -> str:
    """Return a count + noun phrase, pluralising automatically.

    Args:
        count: The integer count.
        singular: Singular form of the noun.
        plural: Optional explicit plural. Defaults to *singular* + "s".

    Example:
        pluralise(1, "frame")   → "1 frame"
        pluralise(3, "frame")   → "3 frames"
        pluralise(1, "match")   → "1 match"  (with plural="matches")
    """
    noun = singular if count == 1 else (plural if plural else singular + "s")
    return f"{count} {noun}"
