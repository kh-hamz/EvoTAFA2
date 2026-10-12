"""Strict streaming logical records; structural errors retain their record identities."""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Record:
    number: int
    line: int
    values: tuple[str, ...]
    error: str = ""


def header(path: Path) -> tuple[str, ...]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            names = tuple(next(csv.reader(stream, strict=True)))
    except (StopIteration, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Unreadable CSV header: {path.name}") from exc
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError(f"Empty or duplicate CSV header: {path.name}")
    return names


def records(path: Path):
    """Do not guess record boundaries following a lexical/encoding failure."""
    names = header(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        next(reader)
        number = 0
        try:
            for number, values in enumerate(reader, 1):
                error = "" if len(values) == len(names) else "wrong_width"
                yield Record(number, reader.line_num, tuple(values), error)
        except (UnicodeError, csv.Error) as exc:
            raise ValueError(f"CSV parsing failed after logical record {number}: {path.name}") from exc
