#!/bin/bash
TOTAL=525
[ -f .mon_start ] || { date +%s > .mon_start; find results/loso/mia_sample -name "fold_*_seed*.json" 2>/dev/null | wc -l > .mon_n0; }
while true; do
  N=$(find results/loso/mia_sample -name "fold_*_seed*.json" 2>/dev/null | wc -l)
  GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)
  python -c "
import time,sys
T0=int(open('.mon_start').read()); N0=int(open('.mon_n0').read())
N=$N; TOTAL=$TOTAL; el=time.time()-T0; dn=N-N0
if el>60 and dn>0:
    rate=dn*3600/el; eta=(TOTAL-N)/rate
    print(f'{time.strftime(\"%H:%M:%S\")}  {N}/{TOTAL}  ({rate:.0f} files/h, ETA {eta:.1f}h)  GPU: $GPU')
else:
    print(f'{time.strftime(\"%H:%M:%S\")}  {N}/{TOTAL}  (measuring...)  GPU: $GPU')
"
  sleep 120
done
