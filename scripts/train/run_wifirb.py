"""
Train/evaluate WiFi baselines on the REBUILT CSI representations (data/wifirb/).

Answers the crux question: does a proper CSI representation (amplitude / Doppler)
beat the raw signed CSI we used originally? Same windows, same split -> the only
difference is the representation. Subject-held-out split (no within-subject leak);
a --loso flag does full leave-one-subject-out.

Examples (run on the server):
    # amp vs signed vs doppler, DeepConvLSTM, subject-held-out, 5 seeds:
    python scripts/train/run_wifirb.py --rep amp     --model deepconvlstm --num-seeds 5
    python scripts/train/run_wifirb.py --rep signed  --model deepconvlstm --num-seeds 5
    python scripts/train/run_wifirb.py --rep doppler --model deepconvlstm --num-seeds 5
    # backbone sweep on amplitude (subject-held-out; pairs with the DeepConvLSTM
    # amp/doppler rows in the paper's subject-independent table):
    python scripts/train/run_wifirb.py --rep amp --model resnet1d     --num-seeds 5
    python scripts/train/run_wifirb.py --rep amp --model csi_resnet2d --num-seeds 5
    # Easy 5-class only (classes 0-4):  add --easy
    # leave-one-subject-out:            add --loso  (19x cost; use --num-seeds 1)

Writes experiments/wifirb_<rep>_<model>[_easy][_loso]/multiseed_summary.json + per-class F1.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.models.deepconvlstm import DeepConvLSTM
from src.models.resnet1d import ResNet1D
from src.models.csi_resnet2d import CSIResNet2D
from src.training.trainer import Trainer
from src.utils.metrics import weighted_accuracy, macro_f1, per_class_f1

WB = Path("data/wifirb")
DEFAULT_TEST_SUBJECTS = [4, 9, 13, 18]   # ~21% held out, spread across the range


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def chunked_stats(X, idx, chunk=512):
    """Per-channel mean/std over X[idx] (mmap-friendly, two-pass)."""
    C = X.shape[2]; n = 0; s = np.zeros(C); ss = np.zeros(C)
    for i in range(0, len(idx), chunk):
        b = np.asarray(X[np.sort(idx[i:i+chunk])], dtype=np.float64)  # (b,T,C)
        s += b.sum((0, 1)); ss += (b**2).sum((0, 1)); n += b.shape[0]*b.shape[1]
    mean = s/n; std = np.sqrt(np.maximum(ss/n - mean**2, 1e-12))
    return mean.astype(np.float32), (std+1e-8).astype(np.float32)


class RB(Dataset):
    def __init__(self, X, y, idx, mean, std, remap=None):
        self.X, self.y, self.idx, self.mean, self.std, self.remap = X, y, idx, mean, std, remap
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        i = int(self.idx[k])
        x = (np.asarray(self.X[i], np.float32) - self.mean) / self.std
        lab = int(self.y[i]); lab = self.remap[lab] if self.remap else lab
        return torch.from_numpy(x), torch.tensor(lab, dtype=torch.long)


def build_model(name, num_classes):
    if name == "deepconvlstm":
        return DeepConvLSTM(in_channels=224, num_classes=num_classes,
                            conv_dropout=0.5, lstm_dropout=0.5, head_dropout=0.5)
    if name == "resnet1d":
        return ResNet1D(in_channels=224, num_classes=num_classes, head_dropout=0.5)
    if name == "csi_resnet2d":
        # 2D ResNet over the (T x subcarrier) amplitude image; stem 32 / dropout
        # 0.5 match configs/ablation/wifi2d_easy.yaml. forward() adds the channel.
        return CSIResNet2D(in_channels=1, num_classes=num_classes,
                           stem_channels=32, head_dropout=0.5)
    raise ValueError(name)


def evaluate(model, loader, device, num_classes):
    model.eval(); yt, yp = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)); yp.append(logits.argmax(1).cpu().numpy()); yt.append(y.numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return (float(weighted_accuracy(yt, yp)), float(macro_f1(yt, yp)),
            per_class_f1(yt, yp, num_classes=num_classes))


def run_split(X, y, train_idx, test_idx, args, num_classes, remap, seed):
    seed_all(seed)
    mean, std = chunked_stats(X, train_idx)
    full = RB(X, y, train_idx, mean, std, remap)
    nval = int(len(full)*0.2); ntr = len(full)-nval
    tr, va = random_split(full, [ntr, nval], generator=torch.Generator().manual_seed(42))
    te = RB(X, y, test_idx, mean, std, remap)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model, num_classes)
    exp = Path("experiments")/args.expname/f"seed_{seed}"; exp.mkdir(parents=True, exist_ok=True)
    Trainer(model, DataLoader(tr, 32, shuffle=True, num_workers=4),
            DataLoader(va, 32, num_workers=4), torch.optim.Adam(model.parameters(), 1e-3),
            dev, exp, early_stop_patience=10).train(num_epochs=80)
    model.load_state_dict(torch.load(exp/"best_model.pt", map_location=dev)); model.to(dev)
    return evaluate(model, DataLoader(te, 32, num_workers=4), dev, num_classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True, choices=["amp", "signed", "doppler"])
    ap.add_argument("--model", default="deepconvlstm", choices=["deepconvlstm", "resnet1d", "csi_resnet2d"])
    ap.add_argument("--num-seeds", type=int, default=5)
    ap.add_argument("--easy", action="store_true")
    ap.add_argument("--loso", action="store_true")
    ap.add_argument("--test-subjects", default=None, help="comma list (held-out split only)")
    args = ap.parse_args()

    X = np.load(WB/f"X_{args.rep}.npy", mmap_mode="r")
    y = np.load(WB/"labels.npy"); subj = np.load(WB/"subjects.npy")
    keep = np.isin(y, [0,1,2,3,4]) if args.easy else np.ones(len(y), bool)
    remap = {c: i for i, c in enumerate([0,1,2,3,4])} if args.easy else None
    num_classes = 5 if args.easy else 27
    args.expname = f"wifirb_{args.rep}_{args.model}" + ("_easy" if args.easy else "") + ("_loso" if args.loso else "")

    if args.loso:
        subs = sorted(set(subj[keep].tolist())); accs, f1s = [], []
        for t in subs:
            tr = np.where(keep & (subj != t))[0]; teidx = np.where(keep & (subj == t))[0]
            a, f, _ = run_split(X, y, tr, teidx, args, num_classes, remap, seed=0)
            print(f"  LOSO subject {t}: acc {a:.4f} f1 {f:.4f}"); accs.append(a); f1s.append(f)
        summ = dict(experiment=args.expname, mode="loso", n_folds=len(subs),
                    acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs, ddof=1)),
                    f1_mean=float(np.mean(f1s)), f1_std=float(np.std(f1s, ddof=1)), per_fold=list(zip(subs, accs, f1s)))
    else:
        tsub = [int(s) for s in args.test_subjects.split(",")] if args.test_subjects else DEFAULT_TEST_SUBJECTS
        tr = np.where(keep & ~np.isin(subj, tsub))[0]; teidx = np.where(keep & np.isin(subj, tsub))[0]
        accs, f1s, pcs = [], [], []
        for s in range(args.num_seeds):
            a, f, pc = run_split(X, y, tr, teidx, args, num_classes, remap, seed=s)
            print(f"  seed {s}: acc {a:.4f} f1 {f:.4f}"); accs.append(a); f1s.append(f); pcs.append(pc)
        summ = dict(experiment=args.expname, mode="held-out", test_subjects=tsub,
                    acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs, ddof=1)),
                    f1_mean=float(np.mean(f1s)), f1_std=float(np.std(f1s, ddof=1)))
        arr = np.array(pcs)
        with open(Path("experiments")/args.expname/"per_class_f1.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["class","f1_mean","f1_std"])
            for c in range(num_classes): w.writerow([c, float(arr[:,c].mean()), float(arr[:,c].std())])

    out = Path("experiments")/args.expname; out.mkdir(parents=True, exist_ok=True)
    (out/"multiseed_summary.json").write_text(json.dumps(summ, indent=2))
    print(f"\n{args.expname}: acc {summ['acc_mean']:.4f} +/- {summ['acc_std']:.4f}  f1 {summ['f1_mean']:.4f}")


if __name__ == "__main__":
    main()
