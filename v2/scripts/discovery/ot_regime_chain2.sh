#!/bin/bash
# OT resolution ladder with the corrected primal-weight convention (r = kappa / omega0).
cd /home/miria/OptEvolve/v2
O=results/ot_regime_20260912
P="^/home/miria/jaxenv2/bin/python scripts/(baselines|discovery|diagnostics)/(qp_scale_sweep|ot_headtohead|ot_pot_probe|kb_loop)\.py"
while pgrep -f "$P" > /dev/null; do sleep 60; done
G="env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false /home/miria/jaxenv2/bin/python"
echo "=== calibration at resolution 64 (fixed convention) $(date -u +%H:%M)"
$G scripts/baselines/ot_headtohead.py --mode calibrate --res 64 --operator reduction --schemes pdhg --device gpu \
   --calib CauchyDensity:1009-1010 --kappas 3,10,30,100,300 --tol 1e-4 --cap 200000 --K 256 --out $O/calib_res64_fixed.json
K=$(python3 -c "
import json
d = json.load(open('$O/calib_res64_fixed.json'))
ok = [r for r in d['rows'] if r.get('passed')]
print(min(ok, key=lambda r: r['t_exec_s'])['kappa'] if ok else '')
" 2>/dev/null)
echo "=== best kappa at resolution 64: ${K:-none certified}"
if [ -n "$K" ]; then
  $G scripts/baselines/ot_headtohead.py --mode eval --res 64 --operator reduction --schemes pdhg --device gpu \
     --kappa "$K" --classes CauchyDensity,GRFmoderate,WhiteNoise --pairs 1001-1002,1003-1004 \
     --tol 1e-4 --cap 200000 --K 256 --out $O/eval_res64.json
  if [ -s $O/pot_probe_res128.json ]; then
    echo "=== POT is viable at 128; full comparison there"
    $G scripts/baselines/ot_headtohead.py --mode eval --res 128 --operator reduction --schemes pdhg --device gpu \
       --kappa "$K" --classes CauchyDensity --pairs 1001-1002 --tol 1e-4 --cap 50000 --K 256 --out $O/eval_res128.json
  else
    echo "=== POT did not finish at 128 within its cap; ours only, to see whether we still certify"
    $G scripts/discovery/kb_loop.py --family ot --instance CauchyDensity:128:1001-1002 --kappa "$K" \
       --tol 1e-4 --max-evals 50000 --K 256 --out $O/ours_only_res128.json
  fi
fi
echo OT_CHAIN2_DONE
