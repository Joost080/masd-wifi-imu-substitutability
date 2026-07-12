#!/usr/bin/env bash
# Phase 2 overnight: build the aligned WiFi+IMU dataset, then train everything
# across both GPUs in parallel. Run inside tmux:  bash scripts/train/run_phase2.sh
set -e
cd "$(dirname "$0")/../.."

echo "=== [1/2] building aligned dataset (downloads phone.tab; reuses cached WiFi R/I) ==="
python scripts/rebuild/build_aligned.py --out-dir data/aligned

echo "=== [2/2] training on 2 GPUs in parallel ==="
gpu0() {
  export CUDA_VISIBLE_DEVICES=0
  python scripts/train/run_aligned.py --model imu  --num-seeds 5                  # VALIDATION (check this first)
  python scripts/train/run_aligned.py --model gmu  --rep amp     --num-seeds 5    # the question
  python scripts/train/run_aligned.py --model gmu  --rep doppler --num-seeds 5
  python scripts/train/run_aligned.py --model late --rep doppler --num-seeds 5
  python scripts/train/run_aligned.py --model imu  --easy --num-seeds 5
  python scripts/train/run_aligned.py --model gmu  --rep amp --easy --num-seeds 5
}
gpu1() {
  export CUDA_VISIBLE_DEVICES=1
  python scripts/train/run_aligned.py --model wifi --rep amp     --num-seeds 5
  python scripts/train/run_aligned.py --model late --rep amp     --num-seeds 5
  python scripts/train/run_aligned.py --model wifi --rep doppler --num-seeds 5
  python scripts/train/run_aligned.py --model wifi --rep amp --easy --num-seeds 5
  python scripts/train/run_aligned.py --model late --rep amp --easy --num-seeds 5
}
gpu0 > logs_phase2_gpu0.txt 2>&1 &
gpu1 > logs_phase2_gpu1.txt 2>&1 &
wait
echo "=== Phase 2 done. Upload experiments/aligned_* summaries. ==="
