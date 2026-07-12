"""
Phase 2: fusion on the ALIGNED proper-WiFi + IMU data (data/aligned/).

Tests the paper's central claim under the CORRECT CSI representation: does fusion
(GMU / late concat) with amplitude/Doppler WiFi beat IMU-alone? Subject-held-out
split. First validate IMU-alone reproduces our ~0.69-0.80 Hard before trusting fusion.

    python scripts/train/run_aligned.py --model imu  --num-seeds 5                 # validation
    python scripts/train/run_aligned.py --model wifi --rep amp --num-seeds 5
    python scripts/train/run_aligned.py --model gmu  --rep amp --num-seeds 5       # the question
    python scripts/train/run_aligned.py --model late --rep amp --num-seeds 5
    # add --easy for 5-class; --rep doppler for the Doppler WiFi.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from src.models.deepconvlstm import DeepConvLSTM
from src.models.fusion import GMULateFusionModel, LateFusionModel
from src.training.trainer import Trainer
from src.utils.metrics import weighted_accuracy, macro_f1, per_class_f1

AL = Path("data/aligned")
TEST_SUBJECTS = [4, 9, 13, 18]


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def chunked_stats(X, idx, chunk=512):
    C = X.shape[2]; n = 0; s = np.zeros(C); ss = np.zeros(C)
    for i in range(0, len(idx), chunk):
        b = np.asarray(X[np.sort(idx[i:i+chunk])], np.float64)
        s += b.sum((0, 1)); ss += (b**2).sum((0, 1)); n += b.shape[0]*b.shape[1]
    m = s/n; return m.astype(np.float32), (np.sqrt(np.maximum(ss/n - m*m, 1e-12))+1e-8).astype(np.float32)


def down(x, n):  # (T,C)->(n,C) adaptive avg pool
    t = torch.from_numpy(np.ascontiguousarray(x)).transpose(0, 1).unsqueeze(0)
    return F.adaptive_avg_pool1d(t, n).squeeze(0).transpose(0, 1).contiguous().numpy()


class AlignedDS(Dataset):
    def __init__(self, mode, Xw, Xi, y, idx, wstat, istat, remap, wlen=150):
        self.mode, self.Xw, self.Xi, self.y, self.idx = mode, Xw, Xi, y, idx
        self.wm, self.ws = wstat; self.im, self.is_ = istat; self.remap = remap; self.wlen = wlen
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        i = int(self.idx[k]); lab = int(self.y[i]); lab = self.remap[lab] if self.remap else lab
        if self.mode == "imu":
            x = (np.asarray(self.Xi[i], np.float32) - self.im) / self.is_
            return torch.from_numpy(x), torch.tensor(lab)
        if self.mode == "wifi":
            x = (np.asarray(self.Xw[i], np.float32) - self.wm) / self.ws
            return torch.from_numpy(x), torch.tensor(lab)
        w = down((np.asarray(self.Xw[i], np.float32) - self.wm) / self.ws, self.wlen)
        m = (np.asarray(self.Xi[i], np.float32) - self.im) / self.is_
        return torch.from_numpy(w), torch.from_numpy(m), torch.tensor(lab)


def build(mode, ncls):
    if mode == "imu":  return DeepConvLSTM(in_channels=9, num_classes=ncls, conv_dropout=0.5, lstm_dropout=0.5, head_dropout=0.5)
    if mode == "wifi": return DeepConvLSTM(in_channels=224, num_classes=ncls, conv_dropout=0.5, lstm_dropout=0.5, head_dropout=0.5)
    if mode == "gmu":  return GMULateFusionModel(num_classes=ncls)
    if mode == "late": return LateFusionModel(num_classes=ncls)
    raise ValueError(mode)


def evaluate(model, loader, dev, ncls):
    model.eval(); yt, yp = [], []
    with torch.no_grad():
        for b in loader:
            if len(b) == 3:
                w, m, y = b; logits = model(w.to(dev), m.to(dev))
            else:
                x, y = b; logits = model(x.to(dev))
            yp.append(logits.argmax(1).cpu().numpy()); yt.append(y.numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return float(weighted_accuracy(yt, yp)), float(macro_f1(yt, yp)), per_class_f1(yt, yp, num_classes=ncls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["imu", "wifi", "gmu", "late"])
    ap.add_argument("--rep", default="amp", choices=["amp", "doppler"])
    ap.add_argument("--num-seeds", type=int, default=5)
    ap.add_argument("--easy", action="store_true")
    args = ap.parse_args()

    Xi = np.load(AL/"X_imu.npy", mmap_mode="r")
    Xw = np.load(AL/f"X_wifi_{args.rep}.npy", mmap_mode="r") if args.model != "imu" else Xi
    y = np.load(AL/"labels.npy"); subj = np.load(AL/"subjects.npy")
    keep = np.isin(y, [0,1,2,3,4]) if args.easy else np.ones(len(y), bool)
    remap = {c:i for i,c in enumerate([0,1,2,3,4])} if args.easy else None
    ncls = 5 if args.easy else 27
    name = f"aligned_{args.model}" + ("" if args.model=="imu" else f"_{args.rep}") + ("_easy" if args.easy else "")
    tr = np.where(keep & ~np.isin(subj, TEST_SUBJECTS))[0]; te = np.where(keep & np.isin(subj, TEST_SUBJECTS))[0]

    accs, f1s, pcs = [], [], []
    for s in range(args.num_seeds):
        seed_all(s)
        wstat = chunked_stats(Xw, tr) if args.model != "imu" else (np.zeros(1,np.float32), np.ones(1,np.float32))
        istat = chunked_stats(Xi, tr)
        full = AlignedDS(args.model, Xw, Xi, y, tr, wstat, istat, remap)
        nval = int(len(full)*0.2); full_tr, full_va = random_split(full, [len(full)-nval, nval], generator=torch.Generator().manual_seed(42))
        ted = AlignedDS(args.model, Xw, Xi, y, te, wstat, istat, remap)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = build(args.model, ncls)
        exp = Path("experiments")/name/f"seed_{s}"; exp.mkdir(parents=True, exist_ok=True)
        Trainer(model, DataLoader(full_tr, 32, shuffle=True, num_workers=4), DataLoader(full_va, 32, num_workers=4),
                torch.optim.Adam(model.parameters(), 1e-3), dev, exp, early_stop_patience=10).train(num_epochs=80)
        model.load_state_dict(torch.load(exp/"best_model.pt", map_location=dev)); model.to(dev)
        a, f, pc = evaluate(model, DataLoader(ted, 32, num_workers=4), dev, ncls)
        print(f"  seed {s}: acc {a:.4f} f1 {f:.4f}"); accs.append(a); f1s.append(f); pcs.append(pc)

    out = Path("experiments")/name; out.mkdir(parents=True, exist_ok=True)
    out.joinpath("multiseed_summary.json").write_text(json.dumps(dict(
        experiment=name, acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs, ddof=1)),
        f1_mean=float(np.mean(f1s)), f1_std=float(np.std(f1s, ddof=1)), test_subjects=TEST_SUBJECTS), indent=2))
    arr = np.array(pcs)
    with open(out/"per_class_f1.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["class","f1_mean","f1_std"])
        for c in range(ncls): w.writerow([c, float(arr[:,c].mean()), float(arr[:,c].std())])
    print(f"\n{name}: acc {np.mean(accs):.4f} +/- {np.std(accs,ddof=1):.4f}  f1 {np.mean(f1s):.4f}")


if __name__ == "__main__":
    main()
