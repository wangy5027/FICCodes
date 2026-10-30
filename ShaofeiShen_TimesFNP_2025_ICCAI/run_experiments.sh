#!/usr/bin/env bash
# 顺序运行 gkx 主实验 (同一时间只跑一个训练进程), 然后汇总结果
# 从项目根目录执行: bash run_experiments.sh
set -e
PY="${PYTHON:-python}"
# 主配置: path_winz 口径, L=120, H=1
for M in base timesfnp; do
  echo "===== gkx $M path_winz L120 H1 ====="
  $PY src/train.py --dataset gkx --model $M --horizon 1 --target path_winz \
      --lookback 120 --lambda1 0.1 --lambda2 0.1 --epochs 40 --batch 64
done
$PY src/evaluate.py --aggregate
