"""Isolated streamed Wireshark adapter; tool output alone does not prove correspondence."""

import csv
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from edgefl.data.schema import projected_time
from edgefl.data.storage import sha256

EXTRA_FIELDS = ("frame.number", "frame.time_epoch", "tcp.stream", "udp.stream",
                "ip.proto", "tcp.flags", "udp.srcport", "udp.dstport")


def run_tool(arguments: list[str], output: Path, timeout: int = 1800):
    """Spool stdout/stderr to disk, bound timeout and diagnostic memory, check status."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stdout, tempfile.TemporaryFile() as stderr:
        try:
            result = subprocess.run(arguments, stdout=stdout, stderr=stderr,
                                    timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError(f"Wireshark invocation failed: {type(exc).__name__}") from exc
        if result.returncode:
            stderr.seek(0)
            detail = stderr.read(2048).decode("utf-8", errors="replace")
            raise ValueError(f"Wireshark exited {result.returncode}: {detail}")


def tool_identity(name: str, work: Path) -> dict:
    executable = shutil.which(name)
    if not executable:
        fallback = Path("C:/Program Files/Wireshark") / (name + ".exe")
        executable = str(fallback) if fallback.is_file() else None
    if not executable:
        raise ValueError(f"Missing Wireshark executable: {name}")
    output = work / (name + "-version.txt")
    run_tool([executable, "-v"], output, 30)
    return {"path": executable, "sha256": sha256(Path(executable)),
            "version": output.read_text(encoding="utf-8", errors="replace").splitlines()[0]}


def extract(executable: str, capture: Path, names, work: Path, timeout=1800) -> Path:
    fields = list(dict.fromkeys([*(name for name in names if name not in ("Attack_label", "Attack_type")),
                                 *EXTRA_FIELDS]))
    command = [executable, "-n", "-r", str(capture), "-T", "fields", "-E", "header=y",
               "-E", "separator=/t", "-E", "quote=d", "-E", "occurrence=a", "-E", "aggregator=,"]
    for name in fields:
        command.extend(("-e", name))
    output = work / "packets.tsv"
    run_tool(command, output, timeout)
    return output


def packet_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", strict=True)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Missing/duplicate Wireshark field header")
        if not {"frame.number","frame.time_epoch"} <= set(reader.fieldnames):
            raise ValueError("Missing required Wireshark fields")
        try:
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Malformed Wireshark field output")
                try:
                    frame = int(row["frame.number"])
                    epoch = Decimal(row["frame.time_epoch"])
                    if frame <= 0 or not epoch.is_finite() or epoch <= 0:
                        raise ValueError
                except (ValueError, ArithmeticError) as exc:
                    raise ValueError("Invalid packet identifier/timestamp") from exc
                yield row
        except (UnicodeError,csv.Error) as exc:
            raise ValueError("Invalid Wireshark field encoding/quoting") from exc


def clock_offset(source_time: str, epoch: str) -> float | None:
    projected = projected_time(source_time)
    if projected is None:
        return None
    year, clock = projected.split()
    instant = datetime.fromtimestamp(float(epoch), timezone.utc)
    if abs(int(year) - instant.year) > 1:
        return None
    hours, minutes, seconds = clock.split(":")
    source_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    packet_seconds = instant.hour * 3600 + instant.minute * 60 + instant.second + instant.microsecond / 1e6
    return round((source_seconds - packet_seconds + 43200) % 86400 - 43200, 4)
