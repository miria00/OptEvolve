#!/bin/bash
# Blackwell, epsilon training set: the screened chain against cuML on ONE idle GPU.
#   bw_screen_chain.sh <gpu> bench      six-variant benchmark (logreg_screen_gpu.py)
#   bw_screen_chain.sh <gpu> diagnose   bandwidth, cuML alone on the same GPU, then the automatic diagnose loop
# Refuses to start if any other process holds the GPU (other users share this box).
set -u
cd ~/OptEvolve/v2
PY=~/jaxenv_bw/bin/python
D=results/logreg_20260911/screen; mkdir -p $D
GPU=$1; ROLE=$2
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$UUID"; then
  echo "GPU $GPU is in use by another process; not starting"; exit 1
fi
export CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false
DATA=data/enet_libsvm/epsilon_train
if [ "$ROLE" = bench ]; then
  JAX_PLATFORMS=cuda $PY scripts/discovery/logreg_screen_gpu.py --data $DATA --reps 3 --out $D/bw_epsilon_train_screen_gpu$GPU.json
else
  JAX_PLATFORMS=cuda $PY scripts/diagnostics/logreg_bandwidth.py --data $DATA --out $D/bw_bandwidth_train_gpu$GPU.json
  echo "=== cuML alone on GPU $GPU, driver on CPU"
  JAX_PLATFORMS=cpu $PY scripts/baselines/logreg_headtohead.py --data $DATA --ours '' --baselines cuml --out $D/bw_epsilon_train_cuml_gpu$GPU.json
  echo "=== automatic diagnose loop"
  JAX_PLATFORMS=cuda $PY scripts/discovery/diagnose_loop_logreg.py --data $DATA --baseline-json $D/bw_epsilon_train_cuml_gpu$GPU.json \
    --bandwidth-json $D/bw_bandwidth_train_gpu$GPU.json --peak-gbps 1792 --out $D/bw_diagnose_loop_train_gpu$GPU.json
fi
echo BW_CHAIN_DONE
