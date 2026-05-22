#!/usr/bin/env python3
"""
PACMO Phase II reconstruction from a victim/held-out capture.

This script consumes waveform_mapping.csv produced by phase1_mapping.py,
keeps only rank-1 mappings whose responsive_threshold is high enough, and
decodes the same payload offsets from a new pcap/pcapng trace.
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from phase1_mapping import Packet, load_udp_packets, write_dict_csv, write_pcap


OUTPUT_COLUMNS = [
    "timestamp",
    "time",
    "fps",
    "hmd_x",
    "hmd_y",
    "hmd_z",
    "hmd_qx",
    "hmd_qy",
    "hmd_qz",
    "hmd_qw",
    "left_x",
    "left_y",
    "left_z",
    "left_qx",
    "left_qy",
    "left_qz",
    "left_qw",
    "right_x",
    "right_y",
    "right_z",
    "right_qx",
    "right_qy",
    "right_qz",
    "right_qw",
]


CHANNEL_TO_COLUMN = {
    "x": "hmd_x",
    "y": "hmd_y",
    "z": "hmd_z",
    "i": "hmd_qx",
    "j": "hmd_qy",
    "k": "hmd_qz",
    "w": "hmd_qw",
    "left_x": "left_x",
    "left_y": "left_y",
    "left_z": "left_z",
    "left_i": "left_qx",
    "left_j": "left_qy",
    "left_k": "left_qz",
    "left_w": "left_qw",
    "right_x": "right_x",
    "right_y": "right_y",
    "right_z": "right_z",
    "right_i": "right_qx",
    "right_j": "right_qy",
    "right_k": "right_qz",
    "right_w": "right_qw",
}


DECODER_FORMATS = {
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


def load_rank1_mappings(mapping_path: Path, threshold: float) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen: set[str] = set()

    with mapping_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["rank"]) != 1:
                continue
            channel = row["channel"]
            if channel in seen:
                continue
            responsive_threshold = float(row["responsive_threshold"])
            if responsive_threshold < threshold:
                continue
            if channel not in CHANNEL_TO_COLUMN:
                continue
            encoding = row["encoding"]
            if encoding not in DECODER_FORMATS:
                continue

            fmt, expected_width = DECODER_FORMATS[encoding]
            width = int(row["width"])
            if width < expected_width:
                continue

            selected.append(
                {
                    "channel": channel,
                    "column": CHANNEL_TO_COLUMN[channel],
                    "offset": int(row["offset"]),
                    "width": expected_width,
                    "encoding": encoding,
                    "fmt": fmt,
                    "delta_view": str(row["delta_view"]).lower() == "true",
                    "affine_scale": float(row["affine_scale"]),
                    "affine_offset": float(row["affine_offset"]),
                    "responsive_threshold": responsive_threshold,
                    "frame_len": int(row["frame_len"]),
                }
            )
            seen.add(channel)

    return selected


def add_default_mapping_overrides(mappings: list[dict[str, object]]) -> None:
    existing_columns = {str(mapping["column"]) for mapping in mappings}
    if "hmd_qy" not in existing_columns:
        mappings.append(
            {
                "channel": "j",
                "column": "hmd_qy",
                "offset": 54,
                "width": 2,
                "encoding": "int16_le",
                "fmt": "<h",
                "delta_view": False,
                # Profiling did not produce a reliable hmd_qy rank-1 entry, so
                # use raw normalized i16 scale. q sign remains application-specific.
                "affine_scale": 1.0 / 32767.0,
                "affine_offset": 0.0,
                "responsive_threshold": "manual_offset_54",
                "frame_len": choose_frame_len(mappings),
            }
        )


def choose_frame_len(mappings: list[dict[str, object]]) -> int:
    if not mappings:
        raise RuntimeError("No mappings survived rank/threshold filtering.")
    return Counter(int(m["frame_len"]) for m in mappings).most_common(1)[0][0]


def select_reconstruction_flow(packets: list[Packet], frame_len: int) -> tuple[tuple[str, str, int, int], list[dict[str, object]]]:
    by_flow: dict[tuple[str, str, int, int], list[Packet]] = defaultdict(list)
    for pkt in packets:
        if pkt.frame_len == frame_len:
            by_flow[pkt.flow].append(pkt)

    rows: list[dict[str, object]] = []
    for flow, flow_packets in by_flow.items():
        if len(flow_packets) < 5:
            continue
        duration = max(flow_packets[-1].ts - flow_packets[0].ts, 1e-9)
        payload_len = Counter(len(pkt.payload) for pkt in flow_packets).most_common(1)[0][0]
        rate = len(flow_packets) / duration
        rows.append(
            {
                "ip_src": flow[0],
                "ip_dst": flow[1],
                "udp_sport": flow[2],
                "udp_dport": flow[3],
                "frame_len": frame_len,
                "packet_count": len(flow_packets),
                "duration_sec": duration,
                "packet_rate": rate,
                "most_common_payload_len": payload_len,
                "score": len(flow_packets) * math.log1p(rate),
            }
        )

    if not rows:
        raise RuntimeError(f"No UDP flow has frame_len={frame_len}.")

    rows.sort(key=lambda r: (r["score"], r["packet_count"]), reverse=True)
    best = rows[0]
    flow = (str(best["ip_src"]), str(best["ip_dst"]), int(best["udp_sport"]), int(best["udp_dport"]))
    return flow, rows


def decode_value(payload: bytes, mapping: dict[str, object]) -> float:
    offset = int(mapping["offset"])
    width = int(mapping["width"])
    if offset + width > len(payload):
        return math.nan
    raw = struct.unpack(str(mapping["fmt"]), payload[offset : offset + width])[0]
    return float(mapping["affine_scale"]) * float(raw) + float(mapping["affine_offset"])


def instantaneous_fps(times: np.ndarray) -> np.ndarray:
    if len(times) == 0:
        return np.array([])
    if len(times) == 1:
        return np.array([0.0])
    diffs = np.diff(times)
    valid = diffs[diffs > 1e-9]
    fallback = 1.0 / float(np.median(valid)) if len(valid) else 0.0
    fps = np.empty(len(times), dtype=float)
    fps[0] = fallback
    fps[1:] = np.where(diffs > 1e-9, 1.0 / diffs, fallback)
    return fps


def reconstruct_raw_rows(packets: list[Packet], mappings: list[dict[str, object]], flow: tuple[str, str, int, int], frame_len: int) -> list[dict[str, object]]:
    selected_packets = [pkt for pkt in packets if pkt.flow == flow and pkt.frame_len == frame_len]
    if not selected_packets:
        return []

    times_abs = np.array([pkt.ts for pkt in selected_packets], dtype=float)
    times_rel = times_abs - times_abs[0]
    fps = instantaneous_fps(times_abs)

    rows: list[dict[str, object]] = []
    previous_decoded: dict[str, float] = {}

    for idx, pkt in enumerate(selected_packets):
        row = {col: "" for col in OUTPUT_COLUMNS}
        row["timestamp"] = f"{times_abs[idx]:.6f}"
        row["time"] = f"{times_rel[idx]:.6f}"
        row["fps"] = f"{fps[idx]:.6f}"

        for mapping in mappings:
            column = str(mapping["column"])
            value = decode_value(pkt.payload, mapping)
            if bool(mapping["delta_view"]):
                value = previous_decoded.get(column, 0.0) + value
                previous_decoded[column] = value
            if math.isfinite(value):
                row[column] = f"{value:.9f}"

        rows.append(row)

    return rows


QUATERNION_GROUPS = [
    ("hmd_qx", "hmd_qy", "hmd_qz", "hmd_qw"),
    ("left_qx", "left_qy", "left_qz", "left_qw"),
    ("right_qx", "right_qy", "right_qz", "right_qw"),
]


def infer_missing_quaternion_components(rows: list[dict[str, object]]) -> int:
    inferred = 0

    for row in rows:
        for group in QUATERNION_GROUPS:
            missing = [col for col in group if row[col] == ""]
            if len(missing) != 1:
                continue

            known_values = []
            for col in group:
                if col == missing[0]:
                    continue
                try:
                    known_values.append(float(row[col]))
                except ValueError:
                    known_values = []
                    break

            if len(known_values) != 3:
                continue

            remainder = 1.0 - sum(value * value for value in known_values)
            if remainder < 0.0 and remainder > -1e-6:
                remainder = 0.0
            if remainder < 0.0:
                continue

            # q and -q encode the same orientation. With only the unit-norm
            # constraint available, choose the positive root deterministically.
            row[missing[0]] = f"{math.sqrt(remainder):.9f}"
            inferred += 1

    return inferred


def resample_rows(rows: list[dict[str, object]], fps: float) -> list[dict[str, object]]:
    if not rows or fps <= 0:
        return rows

    src_time = np.array([float(r["time"]) for r in rows], dtype=float)
    if len(src_time) < 2:
        return rows

    dst_time = np.arange(src_time[0], src_time[-1] + 0.5 / fps, 1.0 / fps)
    start_ts = float(rows[0]["timestamp"])
    out_rows: list[dict[str, object]] = []

    for t in dst_time:
        row = {col: "" for col in OUTPUT_COLUMNS}
        row["timestamp"] = f"{start_ts + t:.6f}"
        row["time"] = f"{t:.6f}"
        row["fps"] = f"{fps:.6f}"

        for col in OUTPUT_COLUMNS[3:]:
            values = np.array([float(r[col]) if r[col] != "" else np.nan for r in rows], dtype=float)
            good = np.isfinite(values)
            if np.count_nonzero(good) >= 2:
                row[col] = f"{np.interp(t, src_time[good], values[good]):.9f}"
            elif np.count_nonzero(good) == 1:
                row[col] = f"{values[good][0]:.9f}"
        out_rows.append(row)

    return out_rows


def write_reconstruction(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct motion CSV from pcapng using waveform_mapping.csv.")
    parser.add_argument("--input", default="5.App5_waveform/p1_example.pcapng", type=Path)
    parser.add_argument("--mapping", default="5.App5_waveform/waveform_mapping.csv", type=Path)
    parser.add_argument("--output", default="5.App5_waveform/p1_reconstructed_motion.csv", type=Path)
    parser.add_argument("--threshold", default=10000.0, type=float)
    parser.add_argument("--resample-rate", default=0.0, type=float, help="Uniform Hz output. 0 keeps packet timestamps.")
    parser.add_argument("--no-default-overrides", action="store_true", help="Disable built-in manual offsets such as hmd_qy=54.")
    parser.add_argument("--write-filtered-pcap", action="store_true")
    args = parser.parse_args()

    input_path = args.input.resolve()
    mapping_path = args.mapping.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mappings = load_rank1_mappings(mapping_path, args.threshold)
    if not args.no_default_overrides:
        add_default_mapping_overrides(mappings)
    frame_len = choose_frame_len(mappings)

    packets = load_udp_packets(input_path)
    flow, flow_rows = select_reconstruction_flow(packets, frame_len)
    selected_packets = [pkt for pkt in packets if pkt.flow == flow and pkt.frame_len == frame_len]

    rows = reconstruct_raw_rows(packets, mappings, flow, frame_len)
    inferred_quaternion_values = infer_missing_quaternion_components(rows)
    if args.resample_rate > 0:
        rows = resample_rows(rows, args.resample_rate)
    write_reconstruction(output_path, rows)

    write_dict_csv(output_path.with_name(output_path.stem + "_flow_summary.csv"), flow_rows)
    write_dict_csv(
        output_path.with_name(output_path.stem + "_mapping_used.csv"),
        [{k: v for k, v in m.items() if k != "fmt"} for m in mappings],
    )

    if args.write_filtered_pcap:
        write_pcap(output_path.with_suffix(".filtered.pcap"), selected_packets)

    mapped_columns = [str(m["column"]) for m in mappings]
    blank_columns = [
        col
        for col in OUTPUT_COLUMNS[3:]
        if rows and all(row.get(col, "") == "" for row in rows)
    ]

    print(f"[OK] input: {input_path}")
    print(f"[OK] mapping: {mapping_path}")
    print(f"[OK] threshold: {args.threshold}")
    print(f"[OK] selected frame length: {frame_len}")
    print(f"[OK] selected flow: {flow[0]}:{flow[2]} -> {flow[1]}:{flow[3]}")
    print(f"[OK] selected packets: {len(selected_packets)}")
    print(f"[OK] output rows: {len(rows)}")
    print(f"[OK] inferred quaternion values: {inferred_quaternion_values}")
    print(f"[OK] mapped columns: {', '.join(mapped_columns)}")
    print(f"[OK] blank columns: {', '.join(blank_columns)}")
    print(f"[OK] wrote: {output_path}")


if __name__ == "__main__":
    main()
