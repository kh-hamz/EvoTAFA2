"""Deterministic registration without rewriting any source."""

from pathlib import Path

from edgefl.config import workspace_path
from edgefl.data.storage import sha256


def register(root: Path, values: dict, progress=lambda **kw: None) -> dict:
    archive = workspace_path(root, values["archive"])
    source_root = workspace_path(root, values["source_root"])
    if not archive.is_dir():
        raise ValueError(f"Missing archive: {archive}")
    files = sorted((p for p in archive.rglob("*") if p.is_file()),
                   key=lambda p: p.relative_to(root).as_posix())
    entries, by_name = [], {}
    for path in files:
        relative = path.relative_to(root).as_posix()
        safe = workspace_path(root, relative)
        if not safe.is_relative_to(archive):
            raise ValueError("Archive symlink escapes immutable archive")
        before = safe.stat()
        digest = sha256(safe)
        after = safe.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source changed while registering")
        entries.append({"path": relative, "size": before.st_size, "sha256": digest,
                        "kind": path.suffix.lower().lstrip(".") or "other"})
        if safe.is_relative_to(source_root):
            by_name.setdefault(path.name, []).append(relative)
        progress(event="registered", path=relative, bytes=before.st_size)
    pairs = []
    for csv_name, pcap_name in sorted(values["pairings"].items()):
        csvs, pcaps = by_name.get(csv_name, []), by_name.get(pcap_name, [])
        if len(csvs) != 1 or len(pcaps) != 1:
            raise ValueError(f"Pairing must resolve uniquely: {csv_name} -> {pcap_name}")
        pairs.append({"csv": csvs[0], "pcap": pcaps[0],
                      "capture_id": csv_name.removesuffix(".csv")})
    selected = set(values["selected"].values())
    registered_csv = {e["path"] for e in entries if e["kind"] == "csv"}
    paired = {p["csv"] for p in pairs}
    if not selected <= registered_csv or not paired <= registered_csv:
        raise ValueError("Selected/source CSV missing from registry")
    if len({p["pcap"] for p in pairs}) != len(pairs):
        raise ValueError("A capture cannot be paired with multiple source CSVs")
    return {"schema_version": "phase-b.v2", "files": entries, "pairs": pairs,
            "selected": values["selected"], "unpaired_csv": sorted(registered_csv - paired - selected)}


def verify_sources(root: Path, registry: dict):
    for entry in registry["files"]:
        path = workspace_path(root, entry["path"])
        if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Registered source changed: {entry['path']}")
