"""Declared field semantics and equivalences; no generic numeric coercion of text."""

import hashlib
import ipaddress
import json
import re
from decimal import Decimal, InvalidOperation
from functools import lru_cache

VERSION = "edgeiiot.fields.v2"
BOOLEAN = frozenset("dns.qry.qu dns.retransmission http.response mqtt.conflag.cleansess tcp.flags.ack".split())
TEXT = frozenset("""
frame.time ip.src_host ip.dst_host arp.dst.proto_ipv4 arp.src.proto_ipv4
http.file_data http.request.uri.query http.request.method http.referer
http.request.full_uri http.request.version tcp.options tcp.payload dns.qry.name
icmp.unused mqtt.msg_decoded_as mqtt.msg mqtt.protoname mqtt.topic Attack_type
""".split())
NUMERIC = frozenset("""
arp.opcode arp.hw.size icmp.checksum icmp.seq_le icmp.transmit_timestamp mqtt.conack.flags
http.content_length http.response http.tls_port tcp.ack tcp.ack_raw tcp.checksum
tcp.connection.fin tcp.connection.rst tcp.connection.syn tcp.connection.synack
tcp.dstport tcp.flags tcp.flags.ack tcp.len tcp.seq tcp.srcport udp.port udp.stream
udp.time_delta dns.qry.name.len dns.qry.qu dns.qry.type dns.retransmission
dns.retransmit_request dns.retransmit_request_in mqtt.conflag.cleansess mqtt.conflags
mqtt.hdrflags mqtt.len mqtt.msgtype mqtt.proto_len mqtt.topic_len mqtt.ver
mbtcp.len mbtcp.trans_id mbtcp.unit_id Attack_label
""".split())
HEX = frozenset("icmp.checksum tcp.checksum tcp.flags mqtt.conflags mqtt.hdrflags mqtt.conack.flags".split())
ADDRESSES = frozenset("ip.src_host ip.dst_host arp.dst.proto_ipv4 arp.src.proto_ipv4".split())
LABELS = frozenset(("Normal", "Backdoor", "DDoS_HTTP", "DDoS_ICMP", "DDoS_TCP", "DDoS_UDP",
                    "Fingerprinting", "MITM", "Password", "Port_Scanning", "Ransomware",
                    "SQL_injection", "Uploading", "Vulnerability_scanner", "XSS"))
EXCLUDE_FEATURES = frozenset(("frame.time", "ip.src_host", "ip.dst_host", "Attack_label", "Attack_type"))
PARTIAL_TIME = re.compile(r"^(\d{4})\s+(\d{2}:\d{2}:\d{2})(?:\.(\d+))?$")
PACKET_TIME = re.compile(r"(\d{4})\s+(\d{2}:\d{2}:\d{2})(?:\.(\d+))?")
NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


@lru_cache(maxsize=65536)
def canonical(field: str, value: str) -> str:
    # The archive explicitly fills missing protocol values with numeric zero.
    # This equivalence is restricted to the declared Edge-IIoT field schema.
    raw = value.strip()
    if field in BOOLEAN and raw.lower() in ("true", "false"):
        return "1" if raw.lower() == "true" else "0"
    if field == "icmp.unused" and raw and set(raw) == {"0"}:
        return "0"
    if field in (NUMERIC | TEXT) - {"frame.time", "Attack_type"} and raw in ("", "0", "0.0"):
        return "0"
    if field in NUMERIC:
        try:
            if field in HEX and raw.lower().startswith("0x"):
                return str(int(raw, 16))
            if NUMBER.fullmatch(raw):
                number = Decimal(raw)
                if number.is_finite():
                    return format(number.normalize(), "f")
        except (ValueError, InvalidOperation):
            pass
    return raw


def label(value: str, aliases: dict) -> str:
    raw = value.strip()
    return aliases.get(raw, raw)


@lru_cache(maxsize=65536)
def value_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def canonical_values(names, values, aliases):
    return tuple(label(value, aliases) if name == "Attack_type" else canonical(name, value)
                 for name, value in zip(names, values))


def record_key(names, values, aliases) -> str:
    return digest((names, canonical_values(names, values, aliases)))


def time_category(value: str) -> str:
    raw = value.strip()
    if PARTIAL_TIME.fullmatch(raw):
        return "year_and_time_missing_month_day"
    if raw in ("", "0", "0.0", "6.0"):
        return "invalid_or_sentinel"
    return "other_unverified_format"


def projected_time(value: str) -> str | None:
    match = PACKET_TIME.search(value.strip())
    if not match:
        return None
    h, m, sec = map(int, match[2].split(":"))
    if h > 23 or m > 59 or sec > 60:
        return None
    return f"{match[1]} {match[2]}.{(match[3] or '').ljust(9, '0')[:9]}"


def evidence_key(names, values, *, packet=False) -> str | None:
    """Full declared predictor comparison; dates come only from the matched packet."""
    if set(names) - NUMERIC - TEXT:
        return None
    fields = dict(zip(names, values))
    stamp = projected_time(fields.get("frame.time", ""))
    if stamp is None and not packet:
        return None
    # Require addresses plus packet-level identifiers; zero vectors cannot match.
    active = [name for name in ("tcp.seq", "tcp.ack_raw", "tcp.checksum", "tcp.srcport",
                               "tcp.dstport", "icmp.checksum", "icmp.seq_le", "arp.opcode")
              if canonical(name, fields.get(name, "")) not in ("", "0")]
    if len(active) < 3:
        return None
    normalized = [(name, stamp if name == "frame.time" else canonical(name, value))
                  for name, value in zip(names, values) if name not in ("frame.time", "Attack_label", "Attack_type")]
    return digest(normalized)


def semantic_issues(names, values, aliases) -> list[str]:
    result = []
    fields = dict(zip(names, values))
    for name, value in fields.items():
        raw = value.strip()
        if name in NUMERIC and canonical(name, raw) != "0":
            try:
                number = Decimal(canonical(name, raw))
                if not number.is_finite():
                    result.append(name + ":nonfinite")
                elif name != "udp.time_delta" and (number < 0 or number != number.to_integral_value()):
                    result.append(name + ":invalid_integer")
                elif name in ("tcp.srcport", "tcp.dstport", "udp.port") and number > 65535:
                    result.append(name + ":invalid_port")
            except InvalidOperation:
                result.append(name + ":malformed_numeric")
        if name in ADDRESSES and canonical(name, raw) != "0":
            try:
                ipaddress.ip_address(raw)
            except ValueError:
                result.append(name + ":invalid_address")
    if "Attack_label" in fields and "Attack_type" in fields:
        expected = "0" if label(fields["Attack_type"], aliases) == "Normal" else "1"
        if canonical("Attack_label", fields["Attack_label"]) != expected:
            result.append("label_inconsistent")
    return result
