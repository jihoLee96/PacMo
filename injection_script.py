import time
import subprocess
import socket
import os
from pathlib import Path


# ============================
# --- CONFIGURATION ---
# ============================

# Interface name or index from tshark -D
PCAP_INTERFACE = "Wi-Fi"   # Using index (e.g. "2") can sometimes be more stable
PCAP_FILE = "exp1.pcap"

# VCR paths
VCR_DIR = r"C:\Users\Anonymous\vr-capture-replay\_out\VCR_1.0.3_dev"
INSTALL_CMD = str(Path(VCR_DIR, "2_install_replay.cmd"))
REPLAY_CMD = str(Path(VCR_DIR, "3_replay.cmd"))
UNINSTALL_CMD = str(Path(VCR_DIR, "4_uninstall_replay.cmd"))
BUSY_FLAG = str(Path(VCR_DIR, "tape", "busy.flag"))

axes = ["x", "y", "z", "i", "j", "k", "w"]

# Replay IDs passed into 3_replay.cmd
REPLAY_IDS = [
    "16_trimmed", "16_fix",
    "16_x", "16_x", "16_x",
    "16_y", "16_y", "16_y",
    "16_z", "16_z", "16_z",
    "16_i", "16_i", "16_i",
    "16_j", "16_j", "16_j",
    "16_k", "16_k", "16_k",
    "16_w", "16_w", "16_w",
    *(f"16_left_{axis}" for axis in axes for _ in range(3)),
    *(f"16_right_{axis}" for axis in axes for _ in range(3)),
]

# Approximate replay duration in seconds
# Mid beacon is sent roughly halfway through
EXPECTED_REPLAY_DURATION = 5.0

# Sync beacon UDP settings
SYNC_IP = "192.168.1.3"
SYNC_PORT = 55555


# ============================
# --- PCAP CAPTURE HELPERS ---
# ============================

def start_capture(pcap_file):
    return subprocess.Popen(
        ["tshark", "-i", PCAP_INTERFACE, "-w", pcap_file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def stop_capture(proc):
    proc.terminate()
    time.sleep(0.5)
    proc.wait()


# ============================
# --- SYNC BEACON HELPERS ---
# ============================

def send_sync_beacon(phase, replay_id=None):
    """
    phase: "global_start", "replay_start", "replay_mid", "replay_end", etc.
    replay_id: replay identifier (optional)
    """
    payload = f"SYNC_BEACON|{phase}"
    if replay_id is not None:
        payload += f"|{replay_id}"

    data = payload.encode("utf-8")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(data, (SYNC_IP, SYNC_PORT))
    sock.close()


# ============================
# --- CMD EXECUTION HELPER ---
# ============================

def run_cmd(cmd_path, args="", log_file=None):
    if args:
        cmd = ["cmd.exe", "/c", "call", cmd_path, args]
    else:
        cmd = ["cmd.exe", "/c", "call", cmd_path]

    # Optional log file
    log_f = open(log_file, "a", encoding="utf-8", errors="ignore") if log_file else None

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Stream output in real time and write to log if enabled
    for line in proc.stdout:
        print(line, end="")
        if log_f:
            log_f.write(line)

    proc.wait()

    if log_f:
        log_f.close()


# ============================
# --- TIMESTAMP LOGGING ---
# ============================

timestamps = []

def log(msg, replay_id=None, phase=None, extra=None):
    """
    msg: event name
    replay_id: associated replay id (optional)
    phase: start/mid/end tag (optional)
    extra: optional extra info (stringified)
    """
    t = time.monotonic_ns()
    t_wall = time.time_ns()

    extra_str = ""
    if extra is not None:
        extra_str = str(extra)

    timestamps.append((msg, t, t_wall, replay_id or "", phase or "", extra_str))


# ============================
# --- BUSY.FLAG MONITORING ---
# ============================

def wait_for_replay_done(
    flag_file,
    replay_id,
    expected_duration=None,
    timeout=60.0,
    poll_interval=0.1,
    log_func=None
):
    """
    Track replay lifecycle using busy.flag:

      1. Flag appears  -> actual replay start
         -> send replay_start beacon + log

      2. Flag disappears -> replay end
         -> send replay_end beacon + log
    """

    start_wait = time.monotonic()
    flag_path = Path(flag_file)

    # Wait for flag to appear
    while not flag_path.exists():
        if time.monotonic() - start_wait > timeout:
            raise TimeoutError("busy.flag did not appear in time")
        time.sleep(poll_interval)

    # Flag appeared: replay start
    send_sync_beacon("replay_start", replay_id)
    if log_func:
        log_func("replay_flag_appeared", replay_id=replay_id, phase="start")

    # Wait until flag disappears
    while flag_path.exists():
        time.sleep(poll_interval)

    # Flag cleared: replay end
    send_sync_beacon("replay_end", replay_id)
    if log_func:
        log_func("replay_flag_cleared", replay_id=replay_id, phase="end")


# ============================
# --- MAIN EXPERIMENT ---
# ============================

def main():
    # Experiment start
    log("experiment_start")

    # Start PCAP capture
    cap = start_capture(PCAP_FILE)
    log("pcap_started")

    # Global sync beacon for the entire session
    time.sleep(1.0)
    send_sync_beacon("global_start", replay_id=None)
    log("global_sync_beacon_sent")
    time.sleep(0.5)

    # ----------------------------
    # NVIDIA VR Capture Replay
    # ----------------------------

    # Install VCR once
    vcr_log = str(Path(VCR_DIR, "vcr_runner.log"))
    run_cmd(INSTALL_CMD, log_file=vcr_log)
    log("vcr_installed")

    # Execute all replays sequentially
    for replay_id in REPLAY_IDS:
        log("replay_prepare", replay_id=replay_id)

        # Pre-replay beacon
        send_sync_beacon("replay_pre", replay_id)
        log("replay_pre_beacon", replay_id=replay_id)

        # Launch replay
        log("replay_cmd_start", replay_id=replay_id)
        run_cmd(REPLAY_CMD, args=replay_id, log_file=vcr_log)
        log("replay_cmd_returned", replay_id=replay_id)

        # Observe busy.flag for actual replay window
        wait_for_replay_done(
            flag_file=BUSY_FLAG,
            replay_id=replay_id,
            expected_duration=EXPECTED_REPLAY_DURATION,
            timeout=60.0,
            poll_interval=0.1,
            log_func=log
        )

        log("replay_done", replay_id=replay_id)

        # Gap between replays
        time.sleep(1.0)

    # Uninstall VCR
    run_cmd(UNINSTALL_CMD, log_file=vcr_log)
    log("vcr_uninstalled")

    # Stop capture
    time.sleep(1.0)
    stop_capture(cap)
    log("pcap_stopped")

    # Save timestamps to CSV
    with open("timestamps.csv", "w", encoding="utf-8") as f:
        f.write("event,t_ns,t_wall,replay_id,phase,extra\n")
        for msg, t, t_wall, rid, phase, extra in timestamps:
            f.write(f"{msg},{t},{t_wall},{rid},{phase},{extra}\n")


if __name__ == "__main__":
    main()
