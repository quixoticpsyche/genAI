#!/bin/bash
cd /home/rit/Projects/genai
while pgrep -f "train.py --steps 20000 --val_every 500 --bs 256 --out run_a.pt" > /dev/null; do sleep 30; done
sleep 20
V=$(.venv/bin/python -c "import torch;print('%.1f'%torch.load('run_a.pt',map_location='cpu',weights_only=False)['val'])")
.venv/bin/python predict.py run_a.pt --chunk 250 --out "subs/sub_FINAL_runA_val${V}.csv"
echo "FINAL CSV READY  val ${V}  -> subs/sub_FINAL_runA_val${V}.csv"
