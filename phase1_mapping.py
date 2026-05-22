#!/usr/bin/env python3
"""
PACMO Phase I steps 2-4 for waveform-injection profiling traces.

Inputs:
  - exp1.pcap or exp1.pcapng captured by injection_script.py

Outputs:
  - filtered_freq.pcap
  - packets_analysis.csv
  - phase1_flow_summary.csv
  - phase1_windows.csv
  - phase1_candidates.csv
  - waveform_mapping.csv

The implementation follows Section 5.1 of the paper:
  Step 2: select the high-rate UDP motion flow and most frequent packet length.
  Step 3: discover responsive candidate offsets using variance of byte diffs.
  Step 4: select offset/encoding hypotheses using frequency-domain matching.
"""

from __future__ import annotations

import argparse
import csv
import math
import socket
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


BEACON_PREFIX = "SYNC_BEACON|"
DEFAULT_REPLAY_IDS = [
    "16_fix",
    "16_x",
    "16_y",
    "16_z",
    "16_i",
    "16_j",
    "16_k",
    "16_w",
    *(f"16_left_{axis}" for axis in ["x", "y", "z", "i", "j", "k", "w"]),
    *(f"16_right_{axis}" for axis in ["x", "y", "z", "i", "j", "k", "w"]),
]


@dataclass(frozen=True)
class Packet:
    pkt_no: int
    ts: float
    raw: bytes
    eth_dst: str
    eth_src: str
    eth_type: int
    ip_src: str
    ip_dst: str
    ip_proto: int
    udp_sport: int
    udp_dport: int
    ip_header: bytes
    udp_header: bytes
    payload: bytes

    @property
    def flow(self) -> tuple[str, str, int, int]:
        return (self.ip_src, self.ip_dst, self.udp_sport, self.udp_dport)

    @property
    def frame_len(self) -> int:
        return len(self.raw)

    @property
    def beacon_type(self) -> str:
        try:
            text = self.payload.decode("ascii", errors="replace")
        except Exception:
            return ""
        return text if text.startswith(BEACON_PREFIX) else ""


def mac_addr(raw: bytes) -> str:
    return ":".join(f"{b:02x}" for b in raw)


def ip_addr(raw: bytes) -> str:
    return socket.inet_ntoa(raw)


def padded_len(n: int) -> int:
    return (n + 3) & ~3


def iter_capture_records(path: Path) -> Iterable[tuple[float, bytes]]:
    data = path.read_bytes()
    magic = data[:4]

    if magic in (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d"):
        yield from iter_pcap_records(data)
        return

    if magic == b"\x0a\x0d\x0d\x0a":
        yield from iter_pcapng_records(data)
        return

    raise ValueError(f"Unsupported capture format: {path}")


def iter_pcap_records(data: bytes) -> Iterable[tuple[float, bytes]]:
    magic = data[:4]
    endian = "<" if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1") else ">"
    nano = magic in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d")
    offset = 24
    scale = 1_000_000_000 if nano else 1_000_000
    while offset + 16 <= len(data):
        sec, frac, caplen, _origlen = struct.unpack_from(endian + "IIII", data, offset)
        offset += 16
        raw = data[offset : offset + caplen]
        offset += caplen
        if len(raw) == caplen:
            yield sec + frac / scale, raw


def iter_pcapng_records(data: bytes) -> Iterable[tuple[float, bytes]]:
    offset = 0
    endian = "<"
    ts_scale_by_if: dict[int, float] = defaultdict(lambda: 1_000_000.0)

    while offset + 12 <= len(data):
        block_type_le, block_len_le = struct.unpack_from("<II", data, offset)
        if block_len_le < 12 or offset + block_len_le > len(data):
            break

        if block_type_le == 0x0A0D0D0A:
            bom = data[offset + 8 : offset + 12]
            endian = "<" if bom == b"\x4d\x3c\x2b\x1a" else ">"

        elif block_type_le == 0x00000001:
            # Interface Description Block. Parse only if_tsresol when present.
            body = data[offset + 8 : offset + block_len_le - 4]
            if len(body) >= 8:
                options = body[8:]
                opt_off = 0
                while opt_off + 4 <= len(options):
                    code, length = struct.unpack_from(endian + "HH", options, opt_off)
                    opt_off += 4
                    value = options[opt_off : opt_off + length]
                    opt_off += padded_len(length)
                    if code == 0:
                        break
                    if code == 9 and value:
                        raw = value[0]
                        base = 2 if (raw & 0x80) else 10
                        exponent = raw & 0x7F
                        ts_scale_by_if[len(ts_scale_by_if)] = float(base**exponent)

        elif block_type_le == 0x00000006:
            body = data[offset + 8 : offset + block_len_le - 4]
            if len(body) >= 20:
                if_id, ts_high, ts_low, caplen, _origlen = struct.unpack_from(endian + "IIIII", body, 0)
                pkt_start = 20
                raw = body[pkt_start : pkt_start + caplen]
                ticks = (ts_high << 32) | ts_low
                yield ticks / ts_scale_by_if[if_id], raw

        offset += block_len_le


def parse_udp_packet(pkt_no: int, ts: float, raw: bytes) -> Packet | None:
    if len(raw) < 14:
        return None

    eth_dst = mac_addr(raw[0:6])
    eth_src = mac_addr(raw[6:12])
    eth_type = struct.unpack("!H", raw[12:14])[0]
    cursor = 14

    if eth_type == 0x8100 and len(raw) >= 18:
        eth_type = struct.unpack("!H", raw[16:18])[0]
        cursor = 18

    if eth_type != 0x0800 or len(raw) < cursor + 20:
        return None

    ip_start = cursor
    version_ihl = raw[ip_start]
    ihl = (version_ihl & 0x0F) * 4
    if version_ihl >> 4 != 4 or len(raw) < ip_start + ihl:
        return None

    ip_proto = raw[ip_start + 9]
    if ip_proto != 17:
        return None

    total_len = struct.unpack("!H", raw[ip_start + 2 : ip_start + 4])[0]
    udp_start = ip_start + ihl
    if len(raw) < udp_start + 8:
        return None

    udp_sport, udp_dport, udp_len, _udp_sum = struct.unpack("!HHHH", raw[udp_start : udp_start + 8])
    payload_start = udp_start + 8
    payload_end = min(payload_start + max(udp_len - 8, 0), ip_start + total_len, len(raw))

    return Packet(
        pkt_no=pkt_no,
        ts=ts,
        raw=raw,
        eth_dst=eth_dst,
        eth_src=eth_src,
        eth_type=eth_type,
        ip_src=ip_addr(raw[ip_start + 12 : ip_start + 16]),
        ip_dst=ip_addr(raw[ip_start + 16 : ip_start + 20]),
        ip_proto=ip_proto,
        udp_sport=udp_sport,
        udp_dport=udp_dport,
        ip_header=raw[ip_start:udp_start],
        udp_header=raw[udp_start:payload_start],
        payload=raw[payload_start:payload_end],
    )


def load_udp_packets(path: Path) -> list[Packet]:
    packets: list[Packet] = []
    for pkt_no, (ts, raw) in enumerate(iter_capture_records(path), 1):
        pkt = parse_udp_packet(pkt_no, ts, raw)
        if pkt is not None:
            packets.append(pkt)
    return packets


def write_pcap(path: Path, packets: Iterable[Packet]) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for pkt in packets:
            sec = int(pkt.ts)
            usec = int(round((pkt.ts - sec) * 1_000_000))
            if usec >= 1_000_000:
                sec += 1
                usec -= 1_000_000
            f.write(struct.pack("<IIII", sec, usec, len(pkt.raw), len(pkt.raw)))
            f.write(pkt.raw)


def write_packet_csv(path: Path, packets: Iterable[Packet]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pkt_no",
                "timestamp",
                "eth_dst",
                "eth_src",
                "eth_type",
                "ip_src",
                "ip_dst",
                "ip_proto",
                "udp_sport",
                "udp_dport",
                "beacon_type",
                "eth_hex",
                "ip_hex",
                "udp_hex",
                "payload_hex",
            ]
        )
        for out_no, pkt in enumerate(packets, 1):
            writer.writerow(
                [
                    out_no,
                    f"{pkt.ts:.6f}",
                    pkt.eth_dst,
                    pkt.eth_src,
                    hex(pkt.eth_type),
                    pkt.ip_src,
                    pkt.ip_dst,
                    pkt.ip_proto,
                    pkt.udp_sport,
                    pkt.udp_dport,
                    pkt.beacon_type,
                    pkt.raw[:14].hex(),
                    pkt.ip_header.hex(),
                    pkt.udp_header.hex(),
                    pkt.payload.hex(),
                ]
            )


def is_beacon(pkt: Packet, beacon_port: int) -> bool:
    return pkt.udp_dport == beacon_port and pkt.beacon_type.startswith(BEACON_PREFIX)


def select_motion_flow(
    packets: list[Packet],
    beacon_port: int,
    min_payload_len: int,
) -> tuple[tuple[str, str, int, int], int, list[dict[str, object]]]:
    beacon_sources = Counter(pkt.ip_src for pkt in packets if is_beacon(pkt, beacon_port))
    preferred_src = beacon_sources.most_common(1)[0][0] if beacon_sources else None

    groups: dict[tuple[str, str, int, int], list[Packet]] = defaultdict(list)
    for pkt in packets:
        if is_beacon(pkt, beacon_port):
            continue
        groups[pkt.flow].append(pkt)

    summaries: list[dict[str, object]] = []
    for flow, rows in groups.items():
        if len(rows) < 5:
            continue
        duration = max(rows[-1].ts - rows[0].ts, 1e-6)
        rate = len(rows) / duration
        motion_like_rows = [pkt for pkt in rows if len(pkt.payload) >= min_payload_len]
        length_source = motion_like_rows if motion_like_rows else rows
        lengths = Counter(pkt.frame_len for pkt in length_source)
        mode_len, mode_count = lengths.most_common(1)[0]
        payload_lengths = Counter(len(pkt.payload) for pkt in length_source)
        mode_payload_len, _ = payload_lengths.most_common(1)[0]
        sustained = min(duration / 30.0, 1.0)
        preferred = 4.0 if preferred_src is not None and flow[0] == preferred_src else 1.0
        score = preferred * sustained * rate * math.log1p(mode_count) * (1.0 / max(mode_payload_len, 1)) ** 0.15
        summaries.append(
            {
                "ip_src": flow[0],
                "ip_dst": flow[1],
                "udp_sport": flow[2],
                "udp_dport": flow[3],
                "packet_count": len(rows),
                "duration_sec": duration,
                "packet_rate": rate,
                "most_common_frame_len": mode_len,
                "most_common_frame_len_count": mode_count,
                "most_common_payload_len": mode_payload_len,
                "motion_like_packet_count": len(motion_like_rows),
                "preferred_beacon_src": preferred_src or "",
                "sustained_factor": sustained,
                "score": score,
            }
        )

    if not summaries:
        raise RuntimeError("No candidate UDP motion flow found.")

    summaries.sort(key=lambda row: (row["score"], row["packet_count"]), reverse=True)
    best = summaries[0]
    flow = (str(best["ip_src"]), str(best["ip_dst"]), int(best["udp_sport"]), int(best["udp_dport"]))
    return flow, int(best["most_common_frame_len"]), summaries


def write_dict_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_replay_windows(packets: list[Packet], beacon_port: int) -> list[dict[str, object]]:
    starts: dict[str, list[float]] = defaultdict(list)
    windows: list[dict[str, object]] = []
    occurrence_by_id: Counter[str] = Counter()

    for pkt in packets:
        if not is_beacon(pkt, beacon_port):
            continue
        parts = pkt.beacon_type.split("|")
        if len(parts) < 3:
            continue
        phase = parts[1]
        replay_id = parts[2] if len(parts) >= 3 else ""
        if not replay_id:
            continue
        if phase == "replay_start":
            starts[replay_id].append(pkt.ts)
        elif phase == "replay_end" and starts[replay_id]:
            start = starts[replay_id].pop(0)
            occurrence_by_id[replay_id] += 1
            windows.append(
                {
                    "replay_id": replay_id,
                    "occurrence": occurrence_by_id[replay_id],
                    "start_ts": start,
                    "end_ts": pkt.ts,
                    "duration_sec": pkt.ts - start,
                }
            )

    windows.sort(key=lambda row: float(row["start_ts"]))
    return windows


def payload_matrix_for_window(
    packets: list[Packet],
    flow: tuple[str, str, int, int],
    frame_len: int,
    start_ts: float,
    end_ts: float,
    trim_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    start = start_ts + trim_sec
    end = end_ts - trim_sec
    selected = [
        pkt
        for pkt in packets
        if pkt.flow == flow and pkt.frame_len == frame_len and start <= pkt.ts <= end and pkt.payload
    ]
    if not selected:
        return np.array([]), np.empty((0, 0), dtype=np.uint8)

    min_len = min(len(pkt.payload) for pkt in selected)
    times = np.array([pkt.ts for pkt in selected], dtype=float)
    payloads = np.array([list(pkt.payload[:min_len]) for pkt in selected], dtype=np.uint8)
    return times, payloads


def responsive_scores(matrices: list[np.ndarray]) -> np.ndarray:
    scores = []
    for mat in matrices:
        if mat.shape[0] < 3:
            continue
        diffs = np.diff(mat.astype(np.float64), axis=0)
        scores.append(np.var(diffs, axis=0))
    if not scores:
        return np.array([])
    min_len = min(len(s) for s in scores)
    return np.mean([s[:min_len] for s in scores], axis=0)


def candidate_groups(scores: np.ndarray, percentile: float) -> tuple[list[dict[str, object]], float]:
    if scores.size == 0:
        return [], 0.0

    positive = scores[scores > 0]
    if positive.size == 0:
        threshold = 0.0
    else:
        med = float(np.median(positive))
        mad = float(np.median(np.abs(positive - med)))
        threshold = max(float(np.percentile(positive, percentile)), med + 3.0 * mad)

    active = scores >= threshold
    groups = []
    start = None
    for idx, value in enumerate(active):
        if value and start is None:
            start = idx
        if start is not None and (not value or idx == len(active) - 1):
            end = idx if value and idx == len(active) - 1 else idx - 1
            group_scores = scores[start : end + 1]
            groups.append(
                {
                    "candidate_start": start,
                    "candidate_end": end,
                    "candidate_width": end - start + 1,
                    "max_responsiveness": float(np.max(group_scores)),
                    "mean_responsiveness": float(np.mean(group_scores)),
                    "threshold": threshold,
                }
            )
            start = None
    return groups, threshold


DECODERS = {
    "float32_le": ("<f", 4),
    "float32_be": (">f", 4),
    "int16_le": ("<h", 2),
    "int16_be": (">h", 2),
    "uint16_le": ("<H", 2),
    "uint16_be": (">H", 2),
    "int32_le": ("<i", 4),
    "int32_be": (">i", 4),
    "uint32_le": ("<I", 4),
    "uint32_be": (">I", 4),
}


def decode_series(mat: np.ndarray, offset: int, fmt: str, width: int) -> np.ndarray | None:
    if mat.shape[1] < offset + width:
        return None
    out = []
    for row in mat:
        try:
            out.append(struct.unpack(fmt, bytes(row[offset : offset + width]))[0])
        except Exception:
            return None
    arr = np.asarray(out, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.nanmean(x)
    std = np.nanstd(x)
    if not np.isfinite(std) or std <= 1e-12:
        return np.zeros_like(x)
    return x / std


def fft_cosine(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 8:
        return 0.0
    fa = np.abs(np.fft.rfft(zscore(a[:n])))[1:]
    fb = np.abs(np.fft.rfft(zscore(b[:n])))[1:]
    denom = np.linalg.norm(fa) * np.linalg.norm(fb)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(fa, fb) / denom)


def sine_reference(n: int, cycles: float) -> np.ndarray:
    # endpoint=False avoids duplicating the first sample at the end of the replay.
    return np.sin(np.linspace(0.0, 2.0 * np.pi * cycles, n, endpoint=False))


def fit_affine(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    n = min(len(source), len(target))
    if n < 2 or np.var(source[:n]) <= 1e-12:
        return 1.0, 0.0
    a, c = np.polyfit(source[:n], target[:n], 1)
    return float(a), float(c)


def evaluate_mapping(
    matrices: list[np.ndarray],
    scores: np.ndarray,
    top_offsets: int,
    sine_cycles: list[float],
    include_delta: bool,
) -> list[dict[str, object]]:
    if scores.size == 0 or not matrices:
        return []

    offset_order = np.argsort(scores)[::-1]
    candidate_offsets = sorted(int(i) for i in offset_order[: min(top_offsets, len(offset_order))] if scores[i] > 0)
    rows: list[dict[str, object]] = []

    for encoding, (fmt, width) in DECODERS.items():
        for offset in candidate_offsets:
            if offset + width > scores.size:
                continue

            decoded_parts = []
            for mat in matrices:
                series = decode_series(mat, offset, fmt, width)
                if series is None or len(series) < 8:
                    decoded_parts = []
                    break
                decoded_parts.append(series)
            if not decoded_parts:
                continue

            for delta in ([False, True] if include_delta else [False]):
                y_parts = [np.diff(s, prepend=s[0]) if delta else s for s in decoded_parts]
                if sum(np.var(part) > 1e-12 for part in y_parts) == 0:
                    continue

                best_score = -1.0
                best_cycle = sine_cycles[0]
                best_ref_parts = []
                for cycles in sine_cycles:
                    ref_parts = [sine_reference(len(part), cycles) for part in y_parts]
                    if delta:
                        ref_parts = [np.diff(ref, prepend=ref[0]) for ref in ref_parts]
                    part_scores = [fft_cosine(ref, y) for ref, y in zip(ref_parts, y_parts)]
                    score = float(np.mean(part_scores)) if part_scores else 0.0
                    if score > best_score:
                        best_score = score
                        best_cycle = cycles
                        best_ref_parts = ref_parts

                y_concat = np.concatenate(y_parts)
                x_concat = np.concatenate(best_ref_parts) if best_ref_parts else np.array([])
                affine_a, affine_c = fit_affine(y_concat, x_concat)
                rows.append(
                    {
                        "offset": offset,
                        "width": width,
                        "encoding": encoding,
                        "delta_view": delta,
                        "score": best_score,
                        "sine_cycles": best_cycle,
                        "responsiveness": float(np.mean(scores[offset : offset + width])),
                        "affine_scale": affine_a,
                        "affine_offset": affine_c,
                    }
                )

    rows.sort(key=lambda row: (row["score"], row["responsiveness"]), reverse=True)
    return rows


def channel_name(replay_id: str) -> str:
    name = replay_id
    if name.startswith("16_"):
        name = name[3:]
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PACMO Phase I steps 2-4 on a waveform profiling pcap.")
    parser.add_argument("--input", default="5.App5_waveform/exp1.pcap", type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    parser.add_argument("--beacon-port", default=55555, type=int)
    parser.add_argument("--trim-sec", default=0.5, type=float)
    parser.add_argument("--candidate-percentile", default=95.0, type=float)
    parser.add_argument("--min-payload-len", default=50, type=int)
    parser.add_argument("--top-offsets", default=48, type=int)
    parser.add_argument("--top-candidates-per-channel", default=25, type=int)
    parser.add_argument("--sine-cycles", default="1,2,3", help="Comma-separated fallback sine cycle counts.")
    parser.add_argument("--min-window-packets", default=20, type=int)
    parser.add_argument("--include-fix", action="store_true", help="Also score 16_fix as a channel.")
    args = parser.parse_args()

    input_path = args.input.resolve()
    out_dir = (args.out_dir or input_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sine_cycles = [float(x.strip()) for x in args.sine_cycles.split(",") if x.strip()]

    packets = load_udp_packets(input_path)
    if not packets:
        raise RuntimeError(f"No UDP packets found in {input_path}")

    flow, frame_len, flow_summary = select_motion_flow(packets, args.beacon_port, args.min_payload_len)
    write_dict_csv(out_dir / "phase1_flow_summary.csv", flow_summary)

    filtered_packets = [pkt for pkt in packets if (pkt.flow == flow and pkt.frame_len == frame_len) or is_beacon(pkt, args.beacon_port)]
    write_pcap(out_dir / "filtered_freq.pcap", filtered_packets)
    write_packet_csv(out_dir / "packets_analysis.csv", filtered_packets)

    windows = parse_replay_windows(packets, args.beacon_port)
    write_dict_csv(out_dir / "phase1_windows.csv", windows)

    matrices_by_replay: dict[str, list[np.ndarray]] = defaultdict(list)
    window_counts: dict[str, int] = defaultdict(int)
    for win in windows:
        replay_id = str(win["replay_id"])
        times, mat = payload_matrix_for_window(
            packets,
            flow,
            frame_len,
            float(win["start_ts"]),
            float(win["end_ts"]),
            args.trim_sec,
        )
        if mat.shape[0] >= args.min_window_packets:
            matrices_by_replay[replay_id].append(mat)
            window_counts[replay_id] += 1

    candidate_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    fix_scores = responsive_scores(matrices_by_replay.get("16_fix", []))

    for replay_id in DEFAULT_REPLAY_IDS:
        if replay_id == "16_fix" and not args.include_fix:
            continue
        matrices = matrices_by_replay.get(replay_id, [])
        if not matrices:
            continue

        scores = responsive_scores(matrices)
        if fix_scores.size:
            n = min(len(scores), len(fix_scores))
            adjusted_scores = scores.copy()
            adjusted_scores[:n] = np.maximum(adjusted_scores[:n] - fix_scores[:n], 0.0)
        else:
            adjusted_scores = scores

        groups, threshold = candidate_groups(adjusted_scores, args.candidate_percentile)
        for group in groups:
            candidate_rows.append(
                {
                    "replay_id": replay_id,
                    "channel": channel_name(replay_id),
                    "windows_used": window_counts[replay_id],
                    **group,
                }
            )

        eval_scores = adjusted_scores.copy()
        eval_scores[eval_scores < threshold] = 0.0

        evaluated = evaluate_mapping(
            matrices=matrices,
            scores=eval_scores,
            top_offsets=args.top_offsets,
            sine_cycles=sine_cycles,
            include_delta=True,
        )
        for rank, row in enumerate(evaluated[: args.top_candidates_per_channel], 1):
            mapping_rows.append(
                {
                    "replay_id": replay_id,
                    "channel": channel_name(replay_id),
                    "rank": rank,
                    "windows_used": window_counts[replay_id],
                    "responsive_threshold": threshold,
                    "flow_ip_src": flow[0],
                    "flow_ip_dst": flow[1],
                    "flow_udp_sport": flow[2],
                    "flow_udp_dport": flow[3],
                    "frame_len": frame_len,
                    **row,
                }
            )

    write_dict_csv(
        out_dir / "phase1_candidates.csv",
        candidate_rows,
        [
            "replay_id",
            "channel",
            "windows_used",
            "candidate_start",
            "candidate_end",
            "candidate_width",
            "max_responsiveness",
            "mean_responsiveness",
            "threshold",
        ],
    )
    write_dict_csv(out_dir / "waveform_mapping.csv", mapping_rows)

    print(f"[OK] input: {input_path}")
    print(f"[OK] selected flow: {flow[0]}:{flow[2]} -> {flow[1]}:{flow[3]}")
    print(f"[OK] selected frame length: {frame_len}")
    print(f"[OK] filtered packets: {len(filtered_packets)}")
    print(f"[OK] replay windows: {len(windows)}")
    print(f"[OK] candidate rows: {len(candidate_rows)}")
    print(f"[OK] mapping rows: {len(mapping_rows)}")
    print(f"[OK] outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
