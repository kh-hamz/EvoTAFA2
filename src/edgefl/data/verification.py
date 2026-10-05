"""Conservative correspondence requires predictors, packet uniqueness, order, and clock evidence."""

from collections import Counter
from pathlib import Path

from edgefl.contracts.phase_b import MANIFEST_FIELDS
from edgefl.data import pcap
from edgefl.data.csv_reader import header, records
from edgefl.data.schema import evidence_key
from edgefl.data.storage import database, rows, writer
from edgefl.data.capture_diagnostics import CaptureDiagnostics


def verify(root: Path, registry: dict, selected_role: str, provenance: Path, candidates: Path,
           directory: Path, inactivity: int, progress=lambda **kw: None, *, diagnostic=False, captures=None) -> dict:
    work = directory / "_work"
    work.mkdir(parents=True, exist_ok=True)
    tshark, capinfos = pcap.tool_identity("tshark", work), pcap.tool_identity("capinfos", work)
    selected_path = root / registry["selected"][selected_role]
    names = header(selected_path)
    db = database(work / "capture.sqlite")
    anomaly_stream = reversal_stream = None
    db.executescript("""
    CREATE TABLE wanted(oid TEXT PRIMARY KEY, key TEXT, ekey TEXT, label TEXT,
                        source_record INTEGER,capture TEXT,stamp TEXT);
    CREATE INDEX wanted_key ON wanted(key);
    CREATE INDEX wanted_capture ON wanted(capture,source_record);
    CREATE INDEX wanted_evidence ON wanted(capture,ekey);
    CREATE TABLE candidate(key TEXT, capture TEXT, n INTEGER);
    CREATE INDEX candidate_key ON candidate(key);
    CREATE TABLE matched(capture TEXT, ekey TEXT, frame INTEGER, epoch TEXT, session TEXT, n INTEGER,
                         PRIMARY KEY(capture,ekey)) WITHOUT ROWID;
    CREATE TABLE session(id TEXT PRIMARY KEY, start REAL, end REAL) WITHOUT ROWID;
    """)
    try:
        for row in rows(candidates):
            db.execute("INSERT INTO candidate VALUES (?,?,?)",
                       (row["record_key"], row["capture_id"], int(row["source_record"])))
        db.commit()
        for row, record in zip(rows(provenance), records(selected_path), strict=True):
            if row["record"] != str(record.number):
                raise ValueError("Selected record/provenance order mismatch")
            if row["status"] == "verified" and row["evidence_key"]:
                origins = db.execute("SELECT capture,n FROM candidate WHERE key=?",
                                     (row["record_key"],)).fetchall()
                if len(origins) != 1:
                    raise ValueError("Provenance candidate count mismatch")
                capture, n = origins[0]
                stamp = record.values[names.index("frame.time")]
                db.execute("INSERT INTO wanted VALUES (?,?,?,?,?,?,?)",
                           (row["observation_id"], row["record_key"], row["evidence_key"],
                            row["label"], n, capture, stamp))
        db.commit()
        capture_reports, accepted = {}, set()
        anomaly_stream, anomaly_output = writer(directory / "timing_anomalies.csv", (
            "capture_id", "previous_frame", "frame", "previous_epoch", "epoch", "backward_seconds"))
        reversal_stream, reversal_output = writer(directory / "anchor_reversals.csv", (
            "capture_id", "previous_source_record", "source_record", "previous_frame", "frame", "source_time", "epoch"))
        for pair in registry["pairs"]:
            capture = pair["capture_id"]
            if captures and capture not in captures:
                continue
            progress(event="capture_start", capture=capture)
            report = {"pcap": pair["pcap"], "csv": pair["csv"]}
            diagnostics = CaptureDiagnostics()
            try:
                summary = work / "capinfos.txt"
                pcap.run_tool([capinfos["path"], "-c", "-a", "-e", str(root / pair["pcap"])], summary, 120)
                report["capinfos"] = summary.read_text(encoding="utf-8", errors="replace")
                packet_file = pcap.extract(tshark["path"], root / pair["pcap"], names, work)
                counts, udp_sessions, protocols = Counter(), {}, Counter()
                previous, first, last = None, None, None
                for packet in pcap.packet_rows(packet_file):
                    anomaly = diagnostics.packet(int(packet["frame.number"]), packet["frame.time_epoch"])
                    if anomaly:
                        anomaly_output.writerow({"capture_id": capture, **anomaly})
                    epoch = float(packet["frame.time_epoch"])
                    counts["packets"] += 1
                    number = packet.get("ip.proto","")
                    protocol = {"6":"TCP","17":"UDP","1":"ICMP"}.get(number,number) if number else ("ARP" if packet.get("arp.opcode","") else "other_non_ip")
                    protocols[protocol] += 1
                    counts["timestamp_regressions"] += int(previous is not None and epoch < previous)
                    previous = epoch
                    first = epoch if first is None else min(first, epoch)
                    last = epoch if last is None else max(last, epoch)
                    if packet.get("tcp.stream", "").isdigit():
                        session = capture + ":tcp:" + packet["tcp.stream"]
                        counts["tcp"] += 1
                    elif packet.get("udp.stream", "").isdigit():
                        stream_id = packet["udp.stream"]
                        old_epoch, segment = udp_sessions.get(stream_id, (epoch, 0))
                        if epoch - old_epoch > inactivity or epoch < old_epoch:
                            segment += 1
                        udp_sessions[stream_id] = (epoch, segment)
                        session = f"{capture}:udp:{stream_id}:{segment}"
                        counts["udp"] += 1
                    else:
                        session = ""
                        counts["other"] += 1
                    if session:
                        db.execute("""INSERT INTO session VALUES (?,?,?) ON CONFLICT(id) DO UPDATE
                            SET start=min(start,excluded.start),end=max(end,excluded.end)""",
                                   (session, epoch, epoch))
                    values = tuple(packet.get(name, "") for name in names)
                    key = evidence_key(names, values, packet=True)
                    if key and db.execute("SELECT 1 FROM wanted WHERE capture=? AND ekey=? LIMIT 1",
                                          (capture, key)).fetchone():
                        db.execute("""INSERT INTO matched VALUES (?,?,?,?,?,1)
                            ON CONFLICT(capture,ekey) DO UPDATE SET n=n+1""",
                                   (capture, key, int(packet["frame.number"]), packet["frame.time_epoch"], session))
                    if counts["packets"] % 10000 == 0:
                        db.commit()
                    if counts["packets"] % 100000 == 0:
                        progress(event="capture_progress", capture=capture, packets=counts["packets"])
                db.commit()
                report.update({"counts": dict(counts), "protocol_distribution":dict(protocols), "first_epoch": first, "last_epoch": last})
                anchors = db.execute("""SELECT w.source_record,w.stamp,m.frame,m.epoch
                    FROM wanted w JOIN matched m ON m.capture=w.capture AND m.ekey=w.ekey
                    WHERE w.capture=? AND m.n=1 ORDER BY w.source_record,w.oid""", (capture,))
                previous_pair, offsets, anchor_count, reversals = None, Counter(), 0, 0
                for source_record, stamp, frame, epoch in anchors:
                    current = (source_record, frame)
                    if previous_pair == current:
                        continue
                    if previous_pair and (source_record <= previous_pair[0] or frame <= previous_pair[1]):
                        reversals += 1
                        reversal_output.writerow({"capture_id": capture, "previous_source_record": previous_pair[0],
                            "source_record": source_record, "previous_frame": previous_pair[1], "frame": frame,
                            "source_time": stamp, "epoch": epoch})
                    diagnostics.anchor(stamp, epoch)
                    offsets[str(pcap.clock_offset(stamp, epoch))] += 1
                    anchor_count += 1
                    previous_pair = current
                report.update({"sequence_anchors": anchor_count, "sequence_reversals": reversals,
                               "observed_clock_offsets_seconds": dict(offsets), "diagnostics": diagnostics.summary()})
                report["correspondence"] = {
                    "wanted_observations": db.execute("SELECT count(*) FROM wanted WHERE capture=?", (capture,)).fetchone()[0],
                    "unique_packet_keys": db.execute("SELECT count(*) FROM matched WHERE capture=? AND n=1", (capture,)).fetchone()[0],
                    "ambiguous_packet_keys": db.execute("SELECT count(*) FROM matched WHERE capture=? AND n>1", (capture,)).fetchone()[0],
                    "unmatched_observations": db.execute("SELECT count(*) FROM wanted w LEFT JOIN matched m ON w.capture=m.capture AND w.ekey=m.ekey WHERE w.capture=? AND m.frame IS NULL", (capture,)).fetchone()[0]}
                if (anchor_count >= 2 and not reversals and counts["timestamp_regressions"] == 0
                        and len(offsets) == 1 and "None" not in offsets):
                    if not diagnostic:
                        accepted.add(capture)
                    report["status"] = "diagnostic_only" if diagnostic else "verified"
                else:
                    report.update(status="quarantined",
                                  reason="insufficient_or_inconsistent_packet_sequence_and_clock_evidence")
            except ValueError as exc:
                report.update(status="quarantined", reason=str(exc))
            capture_reports[capture] = report
            progress(event="capture_complete", capture=capture, status=report["status"])
        anomaly_stream.close()
        reversal_stream.close()
        totals = Counter()
        stream, out = writer(directory / "evidence.csv", MANIFEST_FIELDS["evidence"])
        with stream:
            for row in rows(provenance):
                match = db.execute("""SELECT w.capture,m.frame,m.epoch,m.session,s.start,s.end,m.n
                    FROM wanted w JOIN matched m ON m.capture=w.capture AND m.ekey=w.ekey
                    LEFT JOIN session s ON s.id=m.session WHERE w.oid=?""",
                                   (row["observation_id"],)).fetchone()
                valid = match is not None and match[0] in accepted and match[6] == 1
                state = "verified" if valid else "quarantined"
                reason = "full_predictors_unique_packet_ordered_anchors_stable_clock" if valid else (
                    row["reason"] if row["status"] != "verified" else "capture_or_packet_semantics_unverified")
                capture, frame, epoch, session, start, end = match[:6] if valid else ("", "", "", "", "", "")
                if valid and not session:
                    start = end = epoch
                out.writerow(dict(zip(MANIFEST_FIELDS["evidence"],
                    (row["observation_id"],row["label"],state,reason,capture,frame,epoch,
                     session,start,end,row["evidence_key"]))))
                totals[state] += 1
        return {"statuses": dict(totals), "tools": {"tshark": tshark, "capinfos": capinfos},
                "captures": capture_reports,
                "diagnostic_only": diagnostic, "recovery_authorized": False,
                "timestamp_policy": "Packet epoch supplies the date; CSV clock offset is observed, not assumed."}
    finally:
        for stream in (anomaly_stream, reversal_stream):
            if stream is not None:
                stream.close()
        db.close()
