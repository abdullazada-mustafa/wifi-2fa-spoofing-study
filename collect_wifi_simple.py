"""
Simple Wi-Fi collector (Windows) - no questions, just scans and saves.

Run it, let it go for a few minutes, then press Ctrl+C to stop (or it stops
on its own after the set number of scans). It writes wifi_dataset.csv in the
same columns as the original dataset:
    RPi, SSID, Frequency (Hz), RSSI (dBm), Location, Label

Set LABEL to "1" for scans at the user's position and "0" for scans from a
far-away position, and set LOCATION, before each run.

No admin rights needed. You are only listening to Wi-Fi beacons.
"""

import csv
import os
import re
import subprocess
import sys
import time

OUT_PATH = "wifi_dataset.csv"
COLUMNS = ["RPi", "SSID", "Frequency (Hz)", "RSSI (dBm)", "Location", "Label"]

NUM_SCANS = 60      # about 5 minutes at 5s each
GAP_SECONDS = 5
LOCATION = "far"
LABEL = "0"
RPI = "1"


def signal_percent_to_dbm(percent):
    return round(percent / 2.0 - 100.0, 1)


def channel_to_freq_hz(channel, band_ghz):
    try:
        ch = int(channel)
    except (ValueError, TypeError):
        return ""
    if band_ghz and band_ghz.startswith("5"):
        mhz = 5000 + ch * 5
    elif ch == 14:
        mhz = 2484
    else:
        mhz = 2407 + ch * 5
    return int(mhz * 1_000_000)


def scan():
    try:
        raw = subprocess.check_output(
            ["netsh", "wlan", "show", "networks", "mode=bssid"],
            stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="ignore",
        )
    except Exception as e:
        print("Scan failed:", e)
        return []

    rows, ssid, cur = [], None, {}

    def flush():
        if ssid is not None and "signal" in cur:
            rows.append({"ssid": ssid, "bssid": cur.get("bssid", ""),
                         "signal": cur["signal"], "channel": cur.get("channel", ""),
                         "band": cur.get("band", "")})

    for line in raw.splitlines():
        s = line.strip()
        m = re.match(r"SSID\s+\d+\s*:\s*(.*)", s)
        if m:
            flush(); cur = {}; ssid = m.group(1).strip(); continue
        m = re.match(r"BSSID\s+\d+\s*:\s*(.+)", s)
        if m:
            flush(); cur = {"bssid": m.group(1).strip()}; continue
        m = re.match(r"Signal\s*:\s*(\d+)%", s)
        if m:
            cur["signal"] = int(m.group(1)); continue
        m = re.match(r"Channel\s*:\s*(\d+)", s)
        if m:
            cur["channel"] = m.group(1); continue
        m = re.match(r"Band\s*:\s*(.+)", s)
        if m:
            cur["band"] = m.group(1).strip(); continue
    flush()
    return rows


def main():
    new_file = not os.path.exists(OUT_PATH)
    total = 0
    print("Scanning. Leave this running. Press Ctrl+C to stop early.\n")
    try:
        with open(OUT_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if new_file:
                writer.writeheader()
            for i in range(NUM_SCANS):
                for net in scan():
                    writer.writerow({
                        "RPi": RPI,
                        "SSID": net["ssid"],
                        "Frequency (Hz)": channel_to_freq_hz(net["channel"], net["band"]),
                        "RSSI (dBm)": signal_percent_to_dbm(net["signal"]),
                        "Location": LOCATION,
                        "Label": LABEL,
                    })
                    total += 1
                f.flush()
                print(f"  scan {i+1}/{NUM_SCANS}, {total} rows saved", end="\r")
                if i < NUM_SCANS - 1:
                    time.sleep(GAP_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped early.")
    print(f"\nDone. Saved {total} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
