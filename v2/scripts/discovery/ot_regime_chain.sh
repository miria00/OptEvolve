#!/bin/bash
# OT regime sweep: calibrated kappa at res 64, comparison against POT, then ask whether POT is still viable at 128.
cd /home/miria/OptEvolve/v2
while pgrep -f "^/home/miria/jaxenv2/bin/python scripts/baselines/(ot_headtohead|qp_scale_sweep)\.py" > /dev/null; do sleep 60; done
G="env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false /home/miria/jaxenv2/bin/python"
O=results/ot_regime_20260912
K64=$(python3 -c "
import json
d = json.load(open('$O/calib_res64.json'))
ok = [r for r in d['rows'] if r.get('passed')]
print(min(ok, key=lambda r: r['t_exec_s'])['kappa'] if ok else '')
" 2>/dev/null)
echo \"best kappa at res 64: ${K64:-none certified}\"
if [ -n "$K64" ]; then
  $G scripts/baselines/ot_headtohead.py --mode eval --res 64 --operator reduction --schemes pdhg --device gpu \
     --kappa "$K64" --classes CauchyDensity,GRFmoderate,WhiteNoise --pairs 1001-1002,1003-1004 \
     --tol 1e-4 --cap 200000 --K 256 --out $O/eval_res64.json
fi
echo "=== is POT still viable at resolution 128 (30 minute cap)?"
if timeout 1800 $G scripts/diagnostics/ot_pot_probe.py --res 128 --out $O/pot_probe_res128.json; then
  echo "POT finished at 128; running the full comparison"
  [ -n "$K64" ] && $G scripts/baselines/ot_headtohead.py --mode eval --res 128 --operator reduction --schemes pdhg \
     --device gpu --kappa "$K64" --classes CauchyDensity --pairs 1001-1002 --tol 1e-4 --cap 50000 --K 256 \
     --out $O/eval_res128.json
else
  echo "POT EXCEEDED 1800 s at resolution 128: that is the regime boundary; checking whether we still certify"
  [ -n "$K64" ] && $G scripts/discovery/kb_loop.py --family ot --instance CauchyDensity:128:1001-1002 \
     --kappa "$K64" --tol 1e-4 --max-evals 50000 --K 256 --out $O/ours_only_res128.json
fi
echo OT_REGIME_CHAIN_DONE
