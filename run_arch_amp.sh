#!/usr/bin/env bash
# Re-run the WiFi BACKBONE sweep on the PROPER AMPLITUDE representation, so the
# "WiFi fails under every backbone AND every representation" claim holds on
# amplitude too -- not only on the signed encoding the original sweep used.
#
# Reuses data/wifirb/X_amp.npy (already built by build_wifi_proper.py); no
# rebuild/download needed. Same SUBJECT-HELD-OUT split (test subjects 4,9,13,18)
# and pipeline as run_wifirb.py, so these numbers are directly comparable to the
# DeepConvLSTM amp/doppler rows in the paper's subject-independent table
# (DeepConvLSTM amp = .074 Hard / .425 Easy). Report them as extra rows there.
#
# Four jobs (2 per GPU), balanced heavy/light. Run inside tmux:  bash run_arch_amp.sh
set -e
cd "$(dirname "$0")"

echo "=== WiFi amplitude backbone sweep on 2 GPUs (n=5, subject-held-out) ==="

gpu0() {
  export CUDA_VISIBLE_DEVICES=0
  python run_wifirb.py --rep amp --model csi_resnet2d        --num-seeds 5   # 2D-CNN  Hard (heaviest)
  python run_wifirb.py --rep amp --model resnet1d    --easy  --num-seeds 5   # 1D-ResNet Easy (lightest)
}
gpu1() {
  export CUDA_VISIBLE_DEVICES=1
  python run_wifirb.py --rep amp --model csi_resnet2d --easy --num-seeds 5   # 2D-CNN  Easy
  python run_wifirb.py --rep amp --model resnet1d           --num-seeds 5    # 1D-ResNet Hard
}

gpu0 > logs_arch_amp_gpu0.txt 2>&1 &
gpu1 > logs_arch_amp_gpu1.txt 2>&1 &
wait

echo "=== done. Summaries:"
for e in wifirb_amp_resnet1d wifirb_amp_resnet1d_easy wifirb_amp_csi_resnet2d wifirb_amp_csi_resnet2d_easy; do
  f="experiments/$e/multiseed_summary.json"
  [ -f "$f" ] && python -c "import json;d=json.load(open('$f'));print(f\"  {d['experiment']:28s} acc {d['acc_mean']:.4f} +/- {d['acc_std']:.4f}  f1 {d['f1_mean']:.4f}\")"
done
echo "Compare against signed (subject-held-out, DeepConvLSTM): amp .074 Hard / .425 Easy."
