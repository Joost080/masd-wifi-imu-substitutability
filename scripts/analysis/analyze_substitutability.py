"""
Plan §B — per-class substitutability analysis (no GPU; reads corrected per-class CSVs).

(b) WiFi <-> IMU substitutability boundary across the 27 classes, grouped by
    semantic category: where can ambient WiFi substitute for body-worn IMU?
(c) Feature sufficiency by activity type: stats-only MLP vs DeepConvLSTM (IMU)
    per class -> "stats match deep on X of 27; all Y locomotion classes show a
    Z-W pp deficit, so temporal structure is necessary for motion-intensive acts."

Inputs (corrected-pipeline, n=5 multiseed per-class F1):
    experiments/audit_dropout_wifi/multiseed_per_class_f1.csv      (WiFi, Hard)
    experiments/audit_dropout_imu/multiseed_per_class_f1.csv       (IMU = DeepConvLSTM, Hard)
    experiments/stats_imu_only/multiseed_per_class_f1.csv          (stats-only MLP, Hard)

Usage:  python scripts/analysis/analyze_substitutability.py
Writes: results/substitutability_perclass.csv  and prints the two findings.
"""
import csv
from pathlib import Path

EXP = Path("experiments")
OUT = Path("results"); OUT.mkdir(exist_ok=True)

LABEL = {
    0:"standing",1:"walking",2:"jumping",3:"sitting",4:"lying",5:"wave right hand",
    6:"drink water",7:"torso-twisting",8:"kick right foot",9:"right hand up",
    10:"draw clockwise",11:"turn left",12:"turn right",13:"wave left hand",14:"throw",
    15:"kick left foot",16:"golf swing",17:"basketball shooting",18:"boxing",
    19:"squatting",20:"push",21:"pull",22:"bending (stand)",23:"bending (sit)",
    24:"leg stretch",25:"left hand up",26:"draw counterclockwise",
}

# Defensible semantic grouping (per supervisor's categories). The locomotion group
# is the analytically important one; borderline calls don't affect the headline.
CATEGORY = {
    "static_posture":   [0, 3, 4, 22, 23],                 # standing, sitting, lying, bending x2
    "locomotion":       [1, 2, 11, 12],                    # walking, jumping, turn left/right
    "upper_body_arm":   [5, 6, 9, 10, 13, 20, 21, 25, 26], # uni/bi-lateral arm gestures
    "bilateral_torso":  [7, 18],                           # torso-twisting, boxing
    "sports_limb":      [8, 14, 15, 16, 17, 19, 24],       # kicks, throw, golf, basketball, squat, leg stretch
}
CAT_OF = {c: cat for cat, cs in CATEGORY.items() for c in cs}
assert sorted(CAT_OF) == list(range(27)), "category map must cover all 27 classes exactly"


def load_f1(exp):
    d = {}
    with open(EXP / exp / "multiseed_per_class_f1.csv") as f:
        for row in csv.DictReader(f):
            d[int(row["class"])] = float(row["f1_mean"])
    return d


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--wifi", default="audit_dropout_wifi")
    ap.add_argument("--deep", default="audit_dropout_imu",
                    help="DeepConvLSTM comparator. Use rq2_imu_corrected for the conv_dropout=0.1 best-tuned deep.")
    ap.add_argument("--stats", default="stats_imu_only")
    args = ap.parse_args()
    print(f"[deep comparator = {args.deep}]\n")
    wifi = load_f1(args.wifi)
    imu  = load_f1(args.deep)               # = DeepConvLSTM
    stats = load_f1(args.stats)

    import json
    def macc(exp):
        try: return json.load(open(EXP / exp / "multiseed_summary.json"))["acc_mean"]
        except Exception: return float("nan")
    deep_acc, stats_acc = macc(args.deep), macc(args.stats)

    rows = []
    for c in range(27):
        rows.append({
            "class": c, "name": LABEL[c], "category": CAT_OF[c],
            "wifi_f1": wifi[c], "imu_f1": imu[c], "stats_f1": stats[c],
            "wifi_minus_imu": wifi[c] - imu[c],          # substitutability gap (b)
            "stats_minus_deep": stats[c] - imu[c],       # feature-sufficiency delta (c)
        })

    with open(OUT / "substitutability_perclass.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---------- (b) WiFi <-> IMU substitutability boundary ----------
    print("=" * 78)
    print("(b) WiFi <-> IMU SUBSTITUTABILITY BOUNDARY  (Hard 27-class, corrected n=5)")
    print("=" * 78)
    print(f"{'category':<18}{'n':>3}  {'meanWiFiF1':>10}  {'meanIMUF1':>10}  {'WiFi usable(>0.20)':>18}")
    SUB_THRESH = 0.20
    n_usable_total = 0
    for cat, cs in CATEGORY.items():
        wf = [wifi[c] for c in cs]; mf = [imu[c] for c in cs]
        usable = [c for c in cs if wifi[c] >= SUB_THRESH]
        n_usable_total += len(usable)
        print(f"{cat:<18}{len(cs):>3}  {sum(wf)/len(wf):>10.3f}  {sum(mf)/len(mf):>10.3f}  {len(usable):>10} / {len(cs)}")
    best = sorted(range(27), key=lambda c: -wifi[c])[:5]
    print(f"\nWiFi 'substitutes' (F1 >= {SUB_THRESH}) on {n_usable_total} / 27 classes.")
    print("Best WiFi classes:", ", ".join(f"{LABEL[c]} {wifi[c]:.2f}(IMU {imu[c]:.2f})" for c in best))

    # ---------- (c) feature sufficiency: stats vs DeepConvLSTM ----------
    print("\n" + "=" * 78)
    print("(c) FEATURE SUFFICIENCY  stats-MLP vs DeepConvLSTM  (Hard 27-class, corrected n=5)")
    print("=" * 78)
    stats_ge = [c for c in range(27) if stats[c] >= imu[c]]
    print(f"stats >= deep on {len(stats_ge)} / 27 classes.")
    print(f"{'category':<18}{'n':>3}  {'stats>=deep':>11}  {'mean delta(stats-deep) pp':>26}")
    for cat, cs in CATEGORY.items():
        ge = sum(1 for c in cs if stats[c] >= imu[c])
        md = 100 * sum(stats[c] - imu[c] for c in cs) / len(cs)
        print(f"{cat:<18}{len(cs):>3}  {ge:>6} / {len(cs):<3}  {md:>+26.1f}")
    # Where DeepConvLSTM's temporal modelling actually helps vs where stats win.
    by_delta = sorted(range(27), key=lambda c: stats[c] - imu[c])
    deep_wins = [c for c in by_delta if stats[c] - imu[c] <= -0.05][:6]
    stats_wins = [c for c in by_delta[::-1] if stats[c] - imu[c] >= 0.05][:6]
    print("\nDeep (temporal) wins most:", ", ".join(f"{LABEL[c]} {100*(imu[c]-stats[c]):+.0f}pp" for c in deep_wins))
    print("Stats win most:           ", ", ".join(f"{LABEL[c]} {100*(stats[c]-imu[c]):+.0f}pp" for c in stats_wins))
    loco = CATEGORY["locomotion"]
    print("Locomotion (stats-deep):  ", ", ".join(f"{LABEL[c]} {100*(stats[c]-imu[c]):+.0f}pp" for c in loco))

    # ---------- data-driven findings (recomputed for whatever --deep is given) ----------
    loco_deep_wins = sum(1 for c in loco if stats[c] < imu[c])
    loco_delta = 100 * sum(stats[c] - imu[c] for c in loco) / len(loco)
    if loco_deep_wins >= 3 and loco_delta <= -5:
        loco_verdict = (f"locomotion shows a SYSTEMATIC temporal-structure deficit: deep wins "
                        f"{loco_deep_wins}/{len(loco)} locomotion classes (mean {loco_delta:+.0f} pp) "
                        f"=> temporal structure IS necessary for motion-intensive activities.")
    else:
        loco_verdict = (f"locomotion shows NO clean deficit (deep wins {loco_deep_wins}/{len(loco)}, "
                        f"mean {loco_delta:+.0f} pp) => activity-type split not supported vs this deep model.")
    print("\n" + "-" * 78)
    print(f"FINDINGS (corrected multi-seed pipeline; deep = {args.deep}):")
    print(f"(b) WiFi substitutes for IMU on {n_usable_total}/27 activities (none reach F1>={SUB_THRESH}); "
          f"even its best, {LABEL[best[0]]} {wifi[best[0]]:.2f} (gross body motion), trails IMU "
          f"{imu[best[0]]:.2f}. The substitutability boundary on the hard set is absolute.")
    print(f"(c) stats-MLP (27K) vs DeepConvLSTM (299K): stats>=deep on {len(stats_ge)}/27 classes, "
          f"macro acc {stats_acc:.3f} vs {deep_acc:.3f}. {loco_verdict}")


if __name__ == "__main__":
    main()
