"""
Overnight orchestrator for the corrected-pipeline re-run (plan §F).

Runs the full multi-seed suite across all available GPUs in one batch, balancing
work so the whole thing finishes in a single night. Each (config, seed) is an
independent unit; units are packed onto one lane per GPU by longest-processing-
time-first (LPT), so the heavy ResNet-WiFi runs are spread across GPUs instead of
piling onto one. After training, a cheap eval-only pass writes each config's
multiseed_summary.*.

Run on the uni server after `git pull` AND after `python scripts/checks/smoke_test_a0.py` passes.

    # see the schedule + ETA without running anything:
    python scripts/train/run_overnight.py --gpus 0,1 --dry-run

    # actually run it (logs to logs/overnight/):
    python scripts/train/run_overnight.py --gpus 0,1

    # tight night on 2 GPUs: drop the ResNet-WiFi poles to 3 seeds:
    python scripts/train/run_overnight.py --gpus 0,1 --reduced-poles

    # resume after a crash (skips units whose test_metrics.json already exists):
    python scripts/train/run_overnight.py --gpus 0,1 --resume

Estimates below are per-seed minutes on an A10-class GPU (conservative). Edit the
MANIFEST freely; the scheduler and ETA adapt automatically.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

RESEARCH = Path(__file__).resolve().parents[2]
LOG_DIR = RESEARCH / "logs" / "overnight"

# group, config (relative to research/), seeds, est minutes/seed.
# Groups are cosmetic (for the printed schedule). "pole" marks the heavy ResNet runs
# that --reduced-poles drops to 3 seeds.
MANIFEST = [
    # --- IMU-side baselines (cheap: 150x9) -----------------------------------
    ("imu",    "configs/audit/imu.yaml",                 5,   5),
    ("imu",    "configs/audit/easy_imu.yaml",            5,   4),
    # --- hand-crafted-feature family (tiny MLPs) -----------------------------
    ("stats",  "configs/imu_stats_only.yaml",            5,   2),
    ("stats",  "configs/easy/stats_imu_only.yaml",       5,   2),
    ("stats",  "configs/imu_raw_stats.yaml",             5,   5),
    ("stats",  "configs/stats_imu_wifi.yaml",            5,   2),
    ("stats",  "configs/easy/stats_imu_wifi.yaml",       5,   2),
    ("stats",  "configs/gated_stats_fusion.yaml",        5,   2),
    ("stats",  "configs/easy/gated_stats_fusion.yaml",   5,   2),
    ("stats",  "configs/easy/wifi_stats_only.yaml",      5,   3),
    # --- fusion (now WiFi->150, faster than the old 500-step path) -----------
    ("fusion", "configs/audit/gmu_fusion.yaml",          5,   8),
    ("fusion", "configs/audit/easy_gmu_fusion.yaml",     5,   5),
    ("fusion", "configs/audit/gmu_perchannel.yaml",      5,   8),
    ("fusion", "configs/audit/easy_gmu_perchannel.yaml", 5,   5),
    ("fusion", "configs/audit/gmu_trajectory.yaml",      5,   9),   # track_gate -> extra val passes
    ("fusion", "configs/audit/gmu_moddrop.yaml",         5,   9),
    # --- WiFi DeepConvLSTM (heavy: 500x224) ----------------------------------
    ("wifi",   "configs/audit/wifi.yaml",                5,  18),
    ("wifi",   "configs/audit/easy_wifi.yaml",           5,  12),   # canonical audit Easy-WiFi (0.5/0.5/0.5)
    # --- ResNet-1D on WiFi (the compute poles) -------------------------------
    ("pole",   "configs/audit/wifi_resnet.yaml",         5, 110),
    ("pole",   "configs/easy/wifi_resnet.yaml",          5,  70),
    # --- §C: magnetometer ablation + ResNet-IMU + 2D-CNN ---------------------
    ("ablC",   "configs/ablation/imu6.yaml",             5,   5),
    ("ablC",   "configs/ablation/imu6_easy.yaml",        5,   4),
    ("ablC",   "configs/ablation/imu6_stats.yaml",       5,   2),
    ("ablC",   "configs/ablation/imu6_stats_easy.yaml",  5,   2),
    ("ablC",   "configs/ablation/imu6_resnet.yaml",      5,  18),   # IMU 150x6 ResNet (light)
    ("ablC",   "configs/ablation/wifi2d_easy.yaml",      5,  30),   # 2D CNN on CSI image, Easy
    # --- concat fusion (proposal RQ3 5-config), corrected pipeline at audit dropout --
    ("concat", "configs/audit/early_fusion.yaml",        5,   8),
    ("concat", "configs/audit/late_fusion.yaml",         5,   8),
    ("concat", "configs/audit/easy_early_fusion.yaml",   5,   5),
    ("concat", "configs/audit/easy_late_fusion.yaml",    5,   5),
    # --- Finding-3 spectral test + architecture-matrix fills -------------------
    ("fin3",   "configs/ablation/imu_stats_spectral.yaml", 5,  3),   # time+spectral stats-MLP
    ("fin3",   "configs/ablation/imu6_resnet_easy.yaml",   5, 10),   # ResNet-IMU Easy (cheap)
    ("fin3",   "configs/ablation/wifi2d_hard.yaml",        5, 70),   # 2D-CNN WiFi-Hard (pole)
]


def exp_name(config_path: Path) -> str:
    with open(config_path) as f:
        return yaml.safe_load(f)["experiment"]


def build_units(reduced_poles: bool, groups=None):
    """Expand the manifest into per-(config, seed) units with cost estimates.

    groups: optional set of group tags to keep (e.g. {"ablC"} to run only §C)."""
    units = []
    configs = {}   # config_path -> (exp_name, n_seeds)
    for group, cfg, n_seeds, est in MANIFEST:
        if groups is not None and group not in groups:
            continue
        if reduced_poles and group == "pole":
            n_seeds = 3
        cfg_path = RESEARCH / cfg
        name = exp_name(cfg_path)
        configs[cfg] = (name, n_seeds)
        for seed in range(n_seeds):
            units.append({"group": group, "config": cfg, "exp": name,
                          "seed": seed, "est": est})
    return units, configs


def lpt_schedule(units, n_lanes):
    """Longest-processing-time-first packing onto n_lanes. Returns list[list[unit]]."""
    lanes = [[] for _ in range(n_lanes)]
    loads = [0.0] * n_lanes
    for u in sorted(units, key=lambda x: -x["est"]):
        j = min(range(n_lanes), key=lambda i: loads[i])
        lanes[j].append(u)
        loads[j] += u["est"]
    return lanes, loads


def already_done(unit) -> bool:
    return (RESEARCH / "experiments" / unit["exp"] / f"seed_{unit['seed']}"
            / "test_metrics.json").exists()


def print_schedule(lanes, loads, gpus, max_hours):
    print("=" * 72)
    print("OVERNIGHT SCHEDULE  (LPT across GPUs)")
    print("=" * 72)
    total = sum(loads)
    for gpu, lane, load in zip(gpus, lanes, loads):
        n_pole = sum(1 for u in lane if u["group"] == "pole")
        print(f"  GPU {gpu}: {len(lane):2d} units, ~{load/60:5.2f} h  "
              f"({n_pole} ResNet-WiFi seeds)")
    makespan = max(loads) if loads else 0
    print("-" * 72)
    print(f"  total work : {total/60:6.2f} GPU-h across {len(gpus)} GPU(s)")
    print(f"  makespan   : ~{makespan/60:.2f} h  (slowest lane = wall-clock estimate)")
    verdict = "FITS" if makespan / 60 <= max_hours else "DOES NOT FIT"
    print(f"  budget     : {max_hours:.1f} h  ->  {verdict}")
    if makespan / 60 > max_hours:
        print("  hint       : add a GPU, raise --max-hours, or use --reduced-poles")
    print("=" * 72)


def run_unit(unit, gpu, env_base, logf):
    import os
    env = dict(env_base)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [sys.executable, "scripts/train/run_multiseed.py", "--configs", unit["config"],
           "--seeds", str(unit["seed"])]
    with open(logf, "a") as lf:
        lf.write(f"\n\n===== {unit['exp']} seed {unit['seed']} on GPU {gpu} "
                 f"@ {time.strftime('%H:%M:%S')} =====\n")
        lf.flush()
        return subprocess.run(cmd, cwd=RESEARCH, env=env, stdout=lf,
                              stderr=subprocess.STDOUT).returncode


def lane_worker(gpu, lane, env_base, resume, state):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logf = LOG_DIR / f"gpu{gpu}.log"
    for unit in lane:
        if resume and already_done(unit):
            with state["lock"]:
                state["skipped"] += 1
            continue
        t0 = time.time()
        rc = run_unit(unit, gpu, env_base, logf)
        dt = (time.time() - t0) / 60
        with state["lock"]:
            state["done"] += 1
            tag = "ok" if rc == 0 else f"FAIL(rc={rc})"
            print(f"  [{state['done']}/{state['total']}] GPU{gpu}  "
                  f"{unit['exp']} seed{unit['seed']}  {dt:5.1f}m  {tag}", flush=True)
            if rc != 0:
                state["failures"].append((unit["exp"], unit["seed"], rc))


def prime_caches():
    """Compute the train-partition normalization stats once, sequentially, before the
    parallel lanes start. Populates the on-disk cache (data/_normstats/) so the worker
    processes load instead of each re-gathering the WiFi partition. Torch is imported
    lazily here so --dry-run stays dependency-free."""
    from src.data.dataset import _train_stats, _train_feature_stats
    print("Priming normalization-stat caches (one-time gather)...")
    t0 = time.time()
    combos_raw = [("imu", None), ("wifi", None),
                  ("imu", [0, 1, 2, 3, 4]), ("wifi", [0, 1, 2, 3, 4])]
    combos_feat = [("imu", None), ("imu", [0, 1, 2, 3, 4]), ("wifi", [0, 1, 2, 3, 4])]
    for mod, cf in combos_raw:
        _train_stats(mod, cf)
        print(f"  raw  {mod:<4} cf={cf}  done")
    for mod, cf in combos_feat:
        _train_feature_stats(mod, cf)
        print(f"  feat {mod:<4} cf={cf}  done")
    print(f"  primed in {(time.time()-t0)/60:.1f} min\n")


def aggregate(configs, gpu, env_base):
    """Eval-only pass to write each config's multiseed_summary.* (cheap)."""
    import os
    print("\nAggregating multiseed summaries (eval-only)...")
    env = dict(env_base)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logf = LOG_DIR / "aggregate.log"
    for cfg, (name, n_seeds) in configs.items():
        cmd = [sys.executable, "scripts/train/run_multiseed.py", "--configs", cfg,
               "--num-seeds", str(n_seeds), "--skip-train"]
        with open(logf, "a") as lf:
            lf.write(f"\n===== aggregate {name} (n={n_seeds}) =====\n")
            lf.flush()
            rc = subprocess.run(cmd, cwd=RESEARCH, env=env, stdout=lf,
                                stderr=subprocess.STDOUT).returncode
        print(f"  {name:<34} {'ok' if rc == 0 else f'FAIL(rc={rc})'}")


def main():
    import os
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids, e.g. 0,1")
    ap.add_argument("--max-hours", type=float, default=10.0, help="one-night budget (h)")
    ap.add_argument("--reduced-poles", action="store_true",
                    help="run the ResNet-WiFi poles at 3 seeds instead of 5")
    ap.add_argument("--resume", action="store_true",
                    help="skip (config,seed) units whose test_metrics.json exists")
    ap.add_argument("--dry-run", action="store_true", help="print schedule + ETA, run nothing")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="skip the final eval-only summary pass")
    ap.add_argument("--no-prime", action="store_true",
                    help="skip the one-time normalization-stat cache priming")
    ap.add_argument("--groups", default=None,
                    help="comma-separated group tags to run (e.g. ablC for only §C). Default: all.")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    groups = set(g.strip() for g in args.groups.split(",")) if args.groups else None
    units, configs = build_units(args.reduced_poles, groups)
    lanes, loads = lpt_schedule(units, len(gpus))
    print_schedule(lanes, loads, gpus, args.max_hours)

    if args.dry_run:
        return

    if not args.no_prime:
        prime_caches()

    env_base = dict(os.environ)
    state = {"lock": threading.Lock(), "done": 0, "skipped": 0,
             "total": len(units), "failures": []}
    print(f"\nLaunching {len(units)} units across GPU(s) {','.join(gpus)} "
          f"@ {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    t_start = time.time()

    threads = [threading.Thread(target=lane_worker,
                                args=(gpu, lane, env_base, args.resume, state))
               for gpu, lane in zip(gpus, lanes)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if not args.no_aggregate:
        aggregate(configs, gpus[0], env_base)

    wall = (time.time() - t_start) / 3600
    print(f"\n{'='*72}")
    print(f"DONE in {wall:.2f} h wall-clock  "
          f"({state['done']} ran, {state['skipped']} skipped)")
    if state["failures"]:
        print(f"FAILURES ({len(state['failures'])}):")
        for exp, seed, rc in state["failures"]:
            print(f"  - {exp} seed {seed} (rc={rc})")
        sys.exit(1)
    print("All units completed cleanly.")
    print("=" * 72)


if __name__ == "__main__":
    main()
