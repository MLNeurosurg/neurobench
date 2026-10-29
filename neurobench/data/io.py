"""Provide shared CSV/JSON I/O, response normalization, and output directory safeguards."""

import ast
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile


def read_csv(path, required=()):
    """Load evaluation CSV rows and validate headers, required columns, and row widths."""
    csv.field_size_limit(sys.maxsize)
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if not header or any(not name.strip() for name in header) or len(set(header)) != len(header):
            raise ValueError('CSV must have a nonblank, unique header')
        if missing := set(required) - set(header):
            raise ValueError(f'CSV missing required columns: {sorted(missing)}')
        rows = []
        for index, values in enumerate(reader, 1):
            if len(values) != len(header):
                raise ValueError(f'CSV record {index}: expected {len(header)} fields, found {len(values)}')
            rows.append(dict(zip(header, values)))
    if not rows:
        raise ValueError('Input CSV has no rows')
    return header, rows


def save_csv(path, records, columns):
    """Write an ordered CSV atomically, including its header when no records are supplied."""
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    writer.writerows(records)
    atomic_text(Path(path), buffer.getvalue())


def normalize_answer(value: str) -> str:
    """Decode serialized strings and flat string sequences for answer parsing and judging."""
    text = value.strip()
    if not text:
        return value
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        if text[0] not in "[('\"":
            return value
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return value
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, (list, tuple)) and all(
        isinstance(item, str) for item in parsed
    ):
        return "\n".join(parsed)
    return value


def parse_judgment(raw, label_field, allowed, explanation_field='rationale'):
    """Parse plain or fenced judge JSON using the caller's exact explanation and label schema."""
    text = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise ValueError(f'Expected a JSON object with {explanation_field} and {label_field}') from exc
    if not isinstance(value, dict) or set(value) != {explanation_field, label_field}:
        raise ValueError(f'Expected exactly {explanation_field} and {label_field}')
    if not isinstance(value[explanation_field], str) or not value[explanation_field].strip():
        raise ValueError(f'{explanation_field} must be nonblank text')
    if not isinstance(value[label_field], str) or value[label_field] not in allowed:
        raise ValueError(f'Unrecognized {label_field}')
    value[explanation_field] = value[explanation_field].strip()
    return value


def digest(value) -> str:
    """Compute a deterministic SHA-256 hash for evaluation result and request identifiers."""
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    """Replace a UTF-8 file atomically so saved evaluation and download records remain complete."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path: Path, value) -> None:
    """Write a formatted JSON configuration or manifest atomically."""
    atomic_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def write_jsonl(path: Path, rows) -> None:
    """Replace a JSONL audit file atomically, preserving the supplied record order."""
    atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
        ),
    )


def append_jsonl(path: Path, row) -> None:
    """Append and flush one JSON record to a growing judge audit log."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def require_empty_directory(path: str | Path) -> None:
    """Require a new or empty evaluation directory before any run artifacts are written."""
    path = Path(path)
    if path.exists():
        if not path.is_dir():
            raise ValueError(f'Output path already exists and is not a directory: {path}')
        if any(path.iterdir()):
            raise ValueError(f'Output directory already exists and is nonempty: {path}; choose a new output directory')


@contextmanager
def directory_lock(path: str | Path):
    """Create a directory and hold a nonblocking POSIX lock while its files are updated."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is using {path}") from exc
        yield path
    finally:
        os.close(fd)
