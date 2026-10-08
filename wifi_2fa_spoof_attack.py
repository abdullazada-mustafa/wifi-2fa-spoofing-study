"""
Copied-signal (beacon-spoofing) attack on the Wi-Fi 2FA models from
AlQahtani & Alshayeb, "Zero-Effort Two-Factor Authentication Using Wi-Fi
Radio Wave Transmission and Machine Learning" (extended in the IEEE Access
2024 paper "Leveraging Machine Learning for Wi-Fi-based Environmental
Continuous Two-Factor Authentication").

Data: self-collected Wi-Fi scans, in the same column format as the authors' dataset.
Columns: RPi, SSID, Frequency (Hz), RSSI (dBm), Location, Label
         Label: 1 = authentic (devices co-located), 0 = unauthorized

What this does:
  1. Loads the dataset.
  2. Retrains the paper's classifiers and reports accuracy (sanity check
     against the paper's ~0.92).
  3. Runs a "copied-signal" attack: takes the unauthorized samples (the
     attacker's far-away device), copies legitimate network names (SSIDs)
     onto them, and nudges their RSSI toward the authentic range, as a
     nearby attacker broadcasting look-alike beacons could do. Then
     measures how many of these forged samples each model wrongly ACCEPTS.


Usage:
  1. pip install pandas scikit-learn numpy
  2. Put wifi_dataset.csv next to this file.
  3. python wifi_2fa_spoof_attack.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

CSV_PATH = "wifi_dataset.csv"
RANDOM_STATE = 42

# Column names exactly as published on IEEE DataPort.
COL_RPI = "RPi"
COL_SSID = "SSID"
COL_FREQ = "Frequency (Hz)"
COL_RSSI = "RSSI (dBm)"
COL_LOC = "Location"
COL_LABEL = "Label"


def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]  # tolerate stray spaces
    # Drop rows with missing essentials.
    df = df.dropna(subset=[COL_SSID, COL_FREQ, COL_RSSI, COL_LABEL]).copy()
    df[COL_LABEL] = df[COL_LABEL].astype(int)
    return df


def encode(df, encoders=None, fit=False):
    """Turn the categorical text columns into numbers the models can use.
    SSID is categorical; frequency and RSSI are numeric. RPi and Location
    are encoded too, matching the paper's feature set."""
    df = df.copy()
    cat_cols = [COL_RPI, COL_SSID, COL_LOC]
    if encoders is None:
        encoders = {}
    for c in cat_cols:
        if c not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            encoders[c] = le
        else:
            le = encoders[c]
            known = set(le.classes_)
            # Unseen categories -> map to a sentinel (-1); the model just
            # treats it as an unfamiliar value.
            df[c] = df[c].astype(str).apply(
                lambda v: le.transform([v])[0] if v in known else -1
            )
    return df, encoders


def feature_cols(df):
    # Only the actual Wi-Fi signal features. Location and RPi are left out:
    # in a self-collected dataset each session has one location tag, so the
    # Location column equals the label and the model would just read it
    # instead of learning from the signals.
    cols = [COL_SSID, COL_FREQ, COL_RSSI]
    return [c for c in cols if c in df.columns]


def build_attack_set(train_df):
    """Craft forged samples: start from UNAUTHORIZED rows (attacker far
    away, label 0), then (a) copy a legitimate SSID seen in AUTHENTIC rows
    onto each, and (b) shift its RSSI toward the authentic mean, as a
    nearby attacker spoofing a stronger look-alike beacon would. These
    SHOULD still be rejected (true label 0); every acceptance is a break."""
    authentic = train_df[train_df[COL_LABEL] == 1]
    unauth = train_df[train_df[COL_LABEL] == 0].copy()
    if authentic.empty or unauth.empty:
        raise ValueError("Need both authentic and unauthorized rows.")

    legit_ssids = authentic[COL_SSID].unique()
    auth_rssi_mean = authentic[COL_RSSI].mean()

    rng = np.random.default_rng(RANDOM_STATE)
    unauth[COL_SSID] = rng.choice(legit_ssids, size=len(unauth))
    # Move each forged RSSI 70% of the way from its own value to the
    # authentic mean: a spoofer transmitting at a convincing strength.
    unauth[COL_RSSI] = unauth[COL_RSSI] + 0.7 * (auth_rssi_mean - unauth[COL_RSSI])
    return unauth


def main():
    print("Loading dataset ...")
    df = load_data(CSV_PATH)
    print(f"  {len(df)} samples "
          f"({(df[COL_LABEL]==1).sum()} authentic, "
          f"{(df[COL_LABEL]==0).sum()} unauthorized)\n")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[COL_LABEL]
    )

    train_enc, encoders = encode(train_df, fit=True)
    test_enc, _ = encode(test_df, encoders=encoders, fit=False)

    feats = feature_cols(df)
    X_train, y_train = train_enc[feats], train_enc[COL_LABEL]
    X_test, y_test = test_enc[feats], test_enc[COL_LABEL]

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN":           KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    }

    # Craft the attack once, from the training half, then encode it with
    # the same encoders.
    attack_df = build_attack_set(train_df)
    attack_enc, _ = encode(attack_df, encoders=encoders, fit=False)
    X_attack = attack_enc[feats]

    print("Clean accuracy (sanity check vs paper ~0.92):")
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"  {name:<14} {acc:.3f}")
    print()

    print("Copied-signal attack: share of forged samples wrongly ACCEPTED")
    print("(higher = easier to fool; these should all be rejected)")
    far_test = X_test[y_test == 0]
    for name, model in models.items():
        preds = model.predict(X_attack)           # 1 = accepted as authentic
        accepted = float((preds == 1).mean())
        base = float((model.predict(far_test) == 1).mean()) if len(far_test) else float("nan")
        print(f"  {name:<14} {accepted*100:5.1f}%  "
              f"({int((preds==1).sum())}/{len(preds)} forged samples let in)  "
              f"| baseline: {base*100:5.1f}% of real far samples let in")



if __name__ == "__main__":
    main()
