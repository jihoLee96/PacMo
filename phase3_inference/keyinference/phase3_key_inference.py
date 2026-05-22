#!/usr/bin/env python3
"""
Phase III: virtual key inference from reconstructed PACMO motion.

This implements the paper's Phase III typing pipeline at the artifact level:
  1. Use reconstructed head/hand trajectories from Phase II.
  2. Use click timing from the typing task / click channel.
  3. For each click, extract a fixed-length motion segment around the click.
  4. Convert the segment into motion-dynamics features.
  5. Train/evaluate a LightGBM key classifier for PIN/passcode/sentence inference.

Required event CSV columns:
  - key: the typed key/character label
  - time or timestamp: click time

Recommended event CSV columns:
  - user or source_file: which reconstructed motion CSV to use
  - sequence_id: PIN/passcode/sentence id for exact sequence recovery
  - position: key position within the sequence
  - split: optional train/test/validate split
  - task: optional pin/passcode/sentence tag

Example:
  user,sequence_id,position,time,key,task,split
  p1,trial001,0,12.312,4,pin,train
  p1,trial001,1,13.004,8,pin,train
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


MOTION_COLUMNS = [
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


def user_id_from_path(path: Path) -> str:
    match = re.match(r"(.+?)_reconstructed_motion\.csv$", path.name)
    if match:
        return match.group(1)
    return path.stem


def load_motion_files(input_dir: Path, pattern: str) -> dict[str, pd.DataFrame]:
    motions: dict[str, pd.DataFrame] = {}
    for path in sorted(input_dir.glob(pattern)):
        user = user_id_from_path(path)
        df = pd.read_csv(path)
        required = ["timestamp", "time", *MOTION_COLUMNS]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

        df = df.copy()
        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["timestamp", "time"]).sort_values("time").reset_index(drop=True)

        for col in MOTION_COLUMNS:
            if df[col].notna().sum() >= 2:
                df[col] = df[col].interpolate(limit_direction="both")
        df = df.dropna(subset=MOTION_COLUMNS).reset_index(drop=True)
        if not df.empty:
            motions[user] = df

    if not motions:
        raise RuntimeError(f"No reconstructed motion CSVs matched {input_dir / pattern}")
    return motions


def load_events(events_path: Path) -> pd.DataFrame:
    events = pd.read_csv(events_path)
    if "key" not in events.columns:
        raise ValueError(f"{events_path} needs a 'key' column.")
    if "time" not in events.columns and "timestamp" not in events.columns:
        raise ValueError(f"{events_path} needs either 'time' or 'timestamp' column.")

    events = events.copy()
    events["key"] = events["key"].astype(str)
    if "time" in events.columns:
        events["time"] = pd.to_numeric(events["time"], errors="coerce")
    if "timestamp" in events.columns:
        events["timestamp"] = pd.to_numeric(events["timestamp"], errors="coerce")

    if "user" not in events.columns and "source_file" not in events.columns:
        events["user"] = ""
    if "sequence_id" not in events.columns:
        events["sequence_id"] = ""
    if "position" not in events.columns:
        events["position"] = np.arange(len(events))
    if "split" not in events.columns:
        events["split"] = ""
    if "task" not in events.columns:
        events["task"] = ""

    events = events.dropna(subset=["key"]).reset_index(drop=True)
    return events


def event_user(row: pd.Series, motions: dict[str, pd.DataFrame]) -> str:
    if "user" in row and str(row["user"]) not in ("", "nan"):
        return str(row["user"])
    if "source_file" in row and str(row["source_file"]) not in ("", "nan"):
        return user_id_from_path(Path(str(row["source_file"])))
    if len(motions) == 1:
        return next(iter(motions.keys()))
    raise ValueError("Event row has no user/source_file, but multiple motion files are loaded.")


def event_time(row: pd.Series, motion: pd.DataFrame) -> float:
    if "time" in row and pd.notna(row["time"]):
        return float(row["time"])
    ts = float(row["timestamp"])
    first_ts = float(motion["timestamp"].iloc[0])
    # If the event timestamp looks absolute, convert it to motion-relative time.
    if ts > float(motion["time"].iloc[-1]) + 10.0:
        return ts - first_ts
    return ts


def resample_segment(segment: pd.DataFrame, start: float, stop: float, samples: int) -> np.ndarray | None:
    if len(segment) < 2:
        return None

    src_t = segment["time"].to_numpy(dtype=float)
    dst_t = np.linspace(start, stop, samples)
    values = []
    for col in MOTION_COLUMNS:
        y = segment[col].to_numpy(dtype=float)
        values.append(np.interp(dst_t, src_t, y))
    return np.vstack(values)


def feature_names(samples: int) -> list[str]:
    stats = ["min", "max", "mean", "median", "std", "range", "rms", "start", "end", "delta"]
    names: list[str] = []
    for prefix in ["pre", "post", "all"]:
        for col in MOTION_COLUMNS:
            names.extend(f"{prefix}_{col}_{stat}" for stat in stats)
            names.extend([f"{prefix}_{col}_vel_mean", f"{prefix}_{col}_vel_std", f"{prefix}_{col}_vel_abs_mean"])
    # Keep a light trajectory signature around the click, matching the paper's
    # click-centered motion-segment classifier without requiring deep models.
    for col in MOTION_COLUMNS:
        for idx in range(samples):
            names.append(f"trace_{col}_{idx}")
    names.extend(["segment_duration", "pre_samples", "post_samples"])
    return names


def stats_for_values(values: np.ndarray, dt: float) -> list[float]:
    if values.size == 0:
        return [0.0] * 13

    base = [
        float(np.min(values)),
        float(np.max(values)),
        float(np.mean(values)),
        float(np.median(values)),
        float(np.std(values)),
        float(np.max(values) - np.min(values)),
        float(math.sqrt(np.mean(values * values))),
        float(values[0]),
        float(values[-1]),
        float(values[-1] - values[0]),
    ]
    if len(values) >= 2 and dt > 0:
        velocity = np.diff(values) / dt
    else:
        velocity = np.array([0.0])
    base.extend([float(np.mean(velocity)), float(np.std(velocity)), float(np.mean(np.abs(velocity)))])
    return base


def extract_click_features(
    motion: pd.DataFrame,
    click_time: float,
    pre_sec: float,
    post_sec: float,
    samples: int,
    min_samples: int,
) -> np.ndarray | None:
    start = click_time - pre_sec
    stop = click_time + post_sec
    segment = motion[(motion["time"] >= start) & (motion["time"] <= stop)]
    if len(segment) < min_samples:
        return None

    resampled = resample_segment(segment, start, stop, samples)
    if resampled is None:
        return None

    mid = max(1, int(round(samples * (pre_sec / (pre_sec + post_sec)))))
    dt = (pre_sec + post_sec) / max(samples - 1, 1)
    feats: list[float] = []

    for view in [resampled[:, :mid], resampled[:, mid:], resampled]:
        for channel_values in view:
            feats.extend(stats_for_values(channel_values, dt))

    feats.extend(resampled.reshape(-1).tolist())
    feats.extend([pre_sec + post_sec, float(mid), float(samples - mid)])
    return np.asarray(feats, dtype=float)


def build_dataset(
    motions: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    pre_sec: float,
    post_sec: float,
    samples: int,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    meta_rows: list[dict[str, object]] = []

    for idx, event in events.iterrows():
        user = event_user(event, motions)
        if user not in motions:
            continue
        motion = motions[user]
        click_time = event_time(event, motion)
        feats = extract_click_features(motion, click_time, pre_sec, post_sec, samples, min_samples)
        if feats is None:
            continue

        x_rows.append(feats)
        y_rows.append(str(event["key"]))
        meta_rows.append(
            {
                "event_index": int(idx),
                "user": user,
                "key": str(event["key"]),
                "click_time": click_time,
                "sequence_id": str(event.get("sequence_id", "")),
                "position": event.get("position", ""),
                "task": str(event.get("task", "")),
                "split": str(event.get("split", "")),
            }
        )

    if not x_rows:
        raise RuntimeError("No click-centered feature rows were produced. Check event times and --pre-sec/--post-sec.")
    return np.vstack(x_rows), np.asarray(y_rows), meta_rows


def split_indices(y: np.ndarray, meta: list[dict[str, object]], train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    split_values = [str(row.get("split", "")).lower() for row in meta]
    if any(value in ("train", "test", "validate", "val") for value in split_values):
        train = [i for i, value in enumerate(split_values) if value == "train"]
        test = [i for i, value in enumerate(split_values) if value in ("test", "validate", "val")]
        if train and test:
            return np.asarray(train, dtype=int), np.asarray(test, dtype=int)

    # Temporal per-key split prevents a frequent key from disappearing in train.
    train_idx: list[int] = []
    test_idx: list[int] = []
    for key in np.unique(y):
        idx = np.where(y == key)[0]
        if len(idx) < 2:
            train_idx.extend(idx.tolist())
            continue
        split = int(math.floor(len(idx) * train_fraction))
        split = min(max(split, 1), len(idx) - 1)
        train_idx.extend(idx[:split].tolist())
        test_idx.extend(idx[split:].tolist())
    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-12] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def lightgbm_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    num_trees: int,
    learning_rate: float,
    num_leaves: int,
    device_type: str,
    random_state: int,
) -> tuple[np.ndarray, dict[str, int]]:
    try:
        from lightgbm import LGBMClassifier
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "."
        ) from exc

    labels = sorted(np.unique(train_y).tolist())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    y_encoded = np.asarray([label_to_id[label] for label in train_y], dtype=np.int64)

    clf = LGBMClassifier(
        boosting_type="goss",
        objective="multiclass",
        n_estimators=num_trees,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=-1,
        min_child_weight=7,
        min_data_in_leaf=20,
        colsample_bytree=0.6933333333333332,
        subsample=1.0,
        reg_alpha=0.7894736842105263,
        reg_lambda=0.894736842105263,
        min_split_gain=0.9473684210526315,
        max_bin=63,
        n_jobs=-1,
        device_type=device_type,
        random_state=random_state,
    )
    clf.fit(train_x, y_encoded)
    pred_ids = clf.predict(test_x)
    pred = np.asarray([id_to_label[int(idx)] for idx in pred_ids])
    return pred, label_to_id


def accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(pred == true)) if len(true) else float("nan")


def sequence_recovery(meta: list[dict[str, object]], pred: np.ndarray, true_idx: np.ndarray) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[tuple[int, str, str]]] = defaultdict(list)
    for idx, predicted in zip(true_idx, pred):
        row = meta[int(idx)]
        sequence_id = str(row.get("sequence_id", ""))
        if not sequence_id:
            continue
        try:
            position = int(float(row.get("position", 0)))
        except Exception:
            position = len(grouped[(str(row["user"]), sequence_id)])
        grouped[(str(row["user"]), sequence_id)].append((position, str(row["key"]), str(predicted)))

    rows: list[dict[str, object]] = []
    for (user, sequence_id), values in sorted(grouped.items()):
        values.sort(key=lambda item: item[0])
        true_text = "".join(item[1] for item in values)
        pred_text = "".join(item[2] for item in values)
        rows.append(
            {
                "user": user,
                "sequence_id": sequence_id,
                "true_text": true_text,
                "pred_text": pred_text,
                "exact_match": true_text == pred_text,
                "char_accuracy": sum(a == b for a, b in zip(true_text, pred_text)) / max(len(true_text), 1),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase III virtual key inference from reconstructed motion.")
    parser.add_argument("--input-dir", default="5.App5_waveform", type=Path)
    parser.add_argument("--motion-pattern", default="p*_reconstructed_motion.csv")
    parser.add_argument("--events", required=True, type=Path, help="CSV with key labels and click times.")
    parser.add_argument("--out-dir", default="phase3_inference/keyinference/results", type=Path)
    parser.add_argument("--pre-sec", default=0.35, type=float)
    parser.add_argument("--post-sec", default=0.45, type=float)
    parser.add_argument("--samples", default=32, type=int)
    parser.add_argument("--min-samples", default=8, type=int)
    parser.add_argument("--train-fraction", default=0.7, type=float)
    parser.add_argument("--num-trees", default=200, type=int)
    parser.add_argument("--learning-rate", default=0.1, type=float)
    parser.add_argument("--num-leaves", default=33, type=int)
    parser.add_argument("--device-type", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--random-state", default=42, type=int)
    args = parser.parse_args()

    motions = load_motion_files(args.input_dir.resolve(), args.motion_pattern)
    events = load_events(args.events.resolve())
    x, y, meta = build_dataset(motions, events, args.pre_sec, args.post_sec, args.samples, args.min_samples)
    train_idx, test_idx = split_indices(y, meta, args.train_fraction)

    if len(np.unique(y[train_idx])) < 2:
        raise RuntimeError("Key inference needs at least two key classes in the training split.")
    if len(test_idx) == 0:
        raise RuntimeError("No test click events were produced. Add split=test rows or more events per key.")

    train_x, test_x = standardize(x[train_idx], x[test_idx])
    train_y, test_y = y[train_idx], y[test_idx]

    lightgbm_pred, label_to_id = lightgbm_predict(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        num_trees=args.num_trees,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        device_type=args.device_type,
        random_state=args.random_state,
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_rows = []
    names = feature_names(args.samples)
    for row, feats in zip(meta, x):
        feature_rows.append({**row, **{name: value for name, value in zip(names, feats)}})
    write_csv(out_dir / "click_features.csv", feature_rows, ["event_index", "user", "key", "click_time", "sequence_id", "position", "task", "split", *names])

    prediction_rows = []
    for idx, pred in zip(test_idx, lightgbm_pred):
        row = meta[int(idx)]
        prediction_rows.append(
            {
                "event_index": row["event_index"],
                "user": row["user"],
                "sequence_id": row["sequence_id"],
                "position": row["position"],
                "task": row["task"],
                "click_time": row["click_time"],
                "true_key": row["key"],
                "lightgbm_pred": pred,
                "lightgbm_correct": pred == row["key"],
            }
        )
    write_csv(out_dir / "key_predictions.csv", prediction_rows)

    seq_lightgbm = sequence_recovery(meta, lightgbm_pred, test_idx)
    write_csv(out_dir / "sequence_recovery_lightgbm.csv", seq_lightgbm)

    summary = {
        "motion_files": sorted(motions.keys()),
        "events": str(args.events.resolve()),
        "total_clicks_used": int(len(y)),
        "train_clicks": int(len(train_idx)),
        "test_clicks": int(len(test_idx)),
        "key_classes": sorted(np.unique(y).tolist()),
        "pre_sec": args.pre_sec,
        "post_sec": args.post_sec,
        "samples": args.samples,
        "model": "LightGBM",
        "num_trees": args.num_trees,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "device_type": args.device_type,
        "label_to_id": label_to_id,
        "lightgbm_key_accuracy": accuracy(lightgbm_pred, test_y),
        "lightgbm_sequence_exact": float(np.mean([r["exact_match"] for r in seq_lightgbm])) if seq_lightgbm else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] motion users/files: {', '.join(summary['motion_files'])}")
    print(f"[OK] total/train/test clicks: {len(y)}/{len(train_idx)}/{len(test_idx)}")
    print(f"[OK] key classes: {', '.join(summary['key_classes'])}")
    print(f"[OK] LightGBM key accuracy: {summary['lightgbm_key_accuracy']:.4f}")
    if seq_lightgbm:
        print(f"[OK] LightGBM sequence exact-match: {summary['lightgbm_sequence_exact']:.4f}")
    print(f"[OK] wrote results to: {out_dir}")


if __name__ == "__main__":
    main()
