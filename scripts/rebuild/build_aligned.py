"""
Phase 2: build an ALIGNED WiFi(amplitude/Doppler) + IMU(phone, 9-ch) dataset from
the raw Labeled-MASD data, so we can test fusion with the PROPER CSI representation.

Alignment: each recording (User_X/{wifi,IMU}/aY) is a single activity, so WiFi and
IMU windows from the same recording share label + subject. We pair them by
PROPORTIONAL position (window j of each spans the same time-fraction of the
recording), which avoids per-sample timestamp matching while keeping the pairs
time-consistent enough for activity classification.

IMU = phone.tab: accelerometer(3) + gyroRotation(3) + locationHeading/µT mag(3) = 9
channels (matches our existing IMU). Validate by training IMU-alone on X_imu and
checking it reproduces our ~0.69-0.80 Hard number before trusting the fusion result.

Run on the server (reuses data/wifirb/raw_cache for the WiFi R/I; downloads phone.tab):
    python scripts/rebuild/build_aligned.py --out-dir data/aligned --window 500 --stride 250 --imu-win 150
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_wifi_proper import install_ssl_opener, fetch_metadata, download, representations

PHONE = {  # column prefixes (ignore the unit suffix in parentheses)
    "acc":  ["accelerometerAccelerationX", "accelerometerAccelerationY", "accelerometerAccelerationZ"],
    "gyro": ["gyroRotationX", "gyroRotationY", "gyroRotationZ"],
    "mag":  ["locationHeadingX", "locationHeadingY", "locationHeadingZ"],
}


def parse_phone(path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    pref = {c.split("(")[0]: c for c in df.columns}
    want = PHONE["acc"] + PHONE["gyro"] + PHONE["mag"]
    if not all(w in pref for w in want):
        return None
    arr = df[[pref[w] for w in want]].apply(pd.to_numeric, errors="coerce")
    arr = arr.ffill().bfill().to_numpy(np.float32)
    arr = arr[~np.isnan(arr).any(1)]
    return arr if len(arr) else None


def window_std(x, win, stride):
    n = (len(x) - win) // stride + 1
    return np.stack([x[i * stride:i * stride + win] for i in range(n)]) if n > 0 else None


def window_n(x, win, n):
    if len(x) < win or n < 1:
        return None
    starts = [0] if n == 1 else np.linspace(0, len(x) - win, n).astype(int)
    return np.stack([x[s:s + win] for s in starts])


def list_recs(meta):
    by = defaultdict(dict)
    for f in meta["data"]["latestVersion"]["files"]:
        p = f.get("directoryLabel", "").split("/")
        if len(p) == 3:
            by[(p[0], p[2])].setdefault(p[1], {})[f["dataFile"]["filename"]] = f["dataFile"]["id"]
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/aligned")
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--stride", type=int, default=250)
    ap.add_argument("--imu-win", type=int, default=150)
    ap.add_argument("--wifi-reps", default="amp,doppler")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    install_ssl_opener()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cache = Path("data/wifirb/raw_cache"); cache.mkdir(parents=True, exist_ok=True)  # reuse WiFi R/I
    reps = args.wifi_reps.split(",")

    recs = list_recs(fetch_metadata(out / "masd_meta.json"))
    items = sorted([(k, v) for k, v in recs.items() if "wifi" in v and "IMU" in v])
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} recordings with both wifi + IMU")

    accw = {r: [] for r in reps}; acci = []; labels, subjects = [], []
    for k, ((user, act), mods) in enumerate(items):
        w, im = mods["wifi"], mods["IMU"]
        if not ({"R_array_r1.npy", "I_array_r1.npy"} <= set(w)) or "phone.tab" not in im:
            continue
        Rp = cache / f"{user}_{act}_R.npy"; Ip = cache / f"{user}_{act}_I.npy"; Pp = cache / f"{user}_{act}_phone.tab"
        try:
            download(w["R_array_r1.npy"], Rp); download(w["I_array_r1.npy"], Ip); download(im["phone.tab"], Pp)
            R = np.load(Rp); I = np.load(Ip)
        except Exception as e:
            print(f"  dl/load fail {user}/{act}: {e}"); continue
        if R.ndim != 4 or R.shape[0] < args.window or R.shape != I.shape:
            print(f"  skip {user}/{act}: bad CSI {R.shape}"); continue
        imu = parse_phone(Pp)
        if imu is None or len(imu) < args.imu_win:
            print(f"  skip {user}/{act}: bad/short phone IMU"); continue
        rep_d = representations(R, I)
        wins0 = window_std(rep_d[reps[0]], args.window, args.stride)
        if wins0 is None:
            continue
        n = wins0.shape[0]
        iw = window_n(imu, args.imu_win, n)
        for r in reps:
            accw[r].append(window_std(rep_d[r], args.window, args.stride))
        acci.append(iw)
        labels += [int(act[1:]) - 1] * n; subjects += [int(user.split("_")[1])] * n
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(items)}  ({len(labels)} paired windows)")

    np.save(out / "labels.npy", np.array(labels, np.int64))
    np.save(out / "subjects.npy", np.array(subjects, np.int64))
    np.save(out / "X_imu.npy", np.concatenate(acci).astype(np.float32))
    for r in reps:
        np.save(out / f"X_wifi_{r}.npy", np.concatenate(accw[r]).astype(np.float32))
    print(f"\nsaved {len(labels)} ALIGNED windows | subjects {sorted(set(subjects))}")
    print(f"  X_imu (N,{args.imu_win},9) + X_wifi_{{{','.join(reps)}}} (N,{args.window},224)")
    print("Validate next: train IMU-alone on X_imu -> should reproduce ~0.69-0.80 Hard.")


if __name__ == "__main__":
    main()
