"""
Rebuild proper WiFi-CSI representations from the RAW Labeled-MASD data.

Why: our existing train_wifi.npy is the RAW SIGNED CSI (real/imag components,
~49% negative, max ~510), NOT the standard clean amplitude |R+jI|. This script
re-derives, from the raw complex CSI, three representations on IDENTICAL windows
so we can isolate the effect of the representation:
  - 'signed'   : real part, flattened (reproduces our current input as a control)
  - 'amp'      : clean amplitude sqrt(R^2 + I^2)               <-- the standard input
  - 'doppler'  : antenna-ratio magnitude+phase (cancels the per-receiver phase
                 offset using the 2x2 antennas) -> motion/Doppler-capable

It also records per-window SUBJECT (User_X) and ACTIVITY (aY) so we can build a
leave-one-subject-out (LOSO) evaluation, which the processed arrays never allowed.

Run on the server (needs internet + a few GB of scratch):
    python build_wifi_proper.py --out-dir data/wifirb --window 500 --stride 250

Resumable: downloaded raw files are cached under <out-dir>/raw_cache/ and skipped
if present. Re-run to continue an interrupted download.

CAVEAT TO VERIFY: activity label mapping is assumed aY -> class (Y-1). Confirm
against the existing labels (e.g. rebuild IMU the same way and compare per-class)
before trusting the numbers.
"""
import argparse
import json
import ssl
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

API = "https://dataplanet.ucsd.edu"
PID = "perma:83.ucsddata/FVZWII"


def install_ssl_opener():
    """dataplanet.ucsd.edu's chain isn't verifiable by the server's CA store (or
    certifi). Since this is a public, read-only research-data download, disable
    TLS verification so the download can proceed. (Shapes/values are sanity-checked
    after download, so a MITM would be caught.)"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    urllib.request.install_opener(opener)   # used by both urlopen and urlretrieve
    print("note: TLS certificate verification disabled for the public-data download.")


def fetch_metadata(cache: Path):
    if cache.exists():
        return json.load(open(cache))
    url = f"{API}/api/datasets/:persistentId/?persistentId={PID}"
    print("fetching dataset metadata ...")
    data = json.load(urllib.request.urlopen(url, timeout=120))
    cache.write_text(json.dumps(data))
    return data


def wifi_recordings(meta):
    """Return {(user, activity): {filename: file_id}} for wifi directories."""
    files = meta["data"]["latestVersion"]["files"]
    byrec = defaultdict(dict)
    for f in files:
        dl = f.get("directoryLabel", "")
        parts = dl.split("/")
        if len(parts) == 3 and parts[1] == "wifi":
            byrec[(parts[0], parts[2])][f["dataFile"]["filename"]] = f["dataFile"]["id"]
    return byrec


def download(file_id, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        return
    url = f"{API}/api/access/datafile/{file_id}"
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def representations(R, I):
    """R, I: (T, 56, 2, 2). Returns dict of (T, C) float32 arrays."""
    T = R.shape[0]
    H = R.astype(np.float64) + 1j * I.astype(np.float64)
    signed = R.reshape(T, -1).astype(np.float32)                      # (T,224) control
    amp = np.abs(H).reshape(T, -1).astype(np.float32)                 # (T,224) amplitude
    # antenna-ratio within each Tx: antenna rx0 * conj(rx1) -> cancels phase offset
    ratio = H[:, :, :, 0] * np.conj(H[:, :, :, 1])                    # (T,56,2)
    dopp = np.concatenate([np.abs(ratio), np.angle(ratio)], axis=2)   # (T,56,4)
    dopp = dopp.reshape(T, -1).astype(np.float32)                     # (T,224)
    return {"signed": signed, "amp": amp, "doppler": dopp}


def window(x, win, stride):
    n = (x.shape[0] - win) // stride + 1
    return np.stack([x[i * stride:i * stride + win] for i in range(max(n, 0))]) if n > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/wifirb")
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--stride", type=int, default=250)
    ap.add_argument("--reps", default="signed,amp,doppler")
    ap.add_argument("--limit", type=int, default=0, help="process only N recordings (debug)")
    args = ap.parse_args()

    install_ssl_opener()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cache = out / "raw_cache"; cache.mkdir(exist_ok=True)
    reps = args.reps.split(",")

    meta = fetch_metadata(out / "masd_meta.json")
    recs = wifi_recordings(meta)
    print(f"{len(recs)} wifi recordings found")

    acc = {r: [] for r in reps}
    labels, subjects, recids = [], [], []
    items = sorted(recs.items())
    if args.limit:
        items = items[:args.limit]

    for k, ((user, act), fs) in enumerate(items):
        need = ["R_array_r1.npy", "I_array_r1.npy"]
        if not all(n in fs for n in need):
            print(f"  skip {user}/{act}: missing R/I"); continue
        Rp = cache / f"{user}_{act}_R.npy"; Ip = cache / f"{user}_{act}_I.npy"
        try:
            download(fs["R_array_r1.npy"], Rp); download(fs["I_array_r1.npy"], Ip)
            R = np.load(Rp); I = np.load(Ip)
        except Exception as e:
            print(f"  download/load fail {user}/{act}: {e}")
            Rp.unlink(missing_ok=True); Ip.unlink(missing_ok=True); continue
        if R.ndim != 4 or R.shape[0] < args.window or R.shape != I.shape:
            print(f"  skip {user}/{act}: bad/short array R{R.shape} I{I.shape}")
            Rp.unlink(missing_ok=True); Ip.unlink(missing_ok=True); continue
        try:
            repr_d = representations(R, I)
        except Exception as e:
            print(f"  repr fail {user}/{act}: {e}"); continue
        subj = int(user.split("_")[1]); lab = int(act[1:]) - 1   # aY -> Y-1  (VERIFY)
        wins0 = window(repr_d[reps[0]], args.window, args.stride)
        if wins0 is None:
            continue
        nwin = wins0.shape[0]
        for r in reps:
            acc[r].append(window(repr_d[r], args.window, args.stride))
        labels += [lab] * nwin; subjects += [subj] * nwin; recids += [k] * nwin
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(items)} recordings  ({len(labels)} windows so far)")

    labels = np.array(labels, np.int64); subjects = np.array(subjects, np.int64)
    recids = np.array(recids, np.int64)
    np.save(out / "labels.npy", labels)
    np.save(out / "subjects.npy", subjects)
    np.save(out / "recording.npy", recids)
    for r in reps:
        X = np.concatenate(acc[r]).astype(np.float32)
        np.save(out / f"X_{r}.npy", X)
        print(f"saved {r}: {X.shape}  -> {out / ('X_'+r+'.npy')}")
    print(f"\n{len(labels)} windows | {len(set(subjects))} subjects | {len(set(labels))} activities")
    print("subjects:", sorted(set(subjects.tolist())))
    print("Next: build train/test (recording-level) + LOSO folds, add a loader modality, re-run WiFi baselines.")


if __name__ == "__main__":
    main()
