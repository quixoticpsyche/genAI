#!/bin/bash
cd /home/rit/Projects/genai
while pgrep -f "train.py --steps 20000 --val_every 500 --bs 256 --seed 1" > /dev/null; do sleep 60; done
sleep 20
PYTHONPATH=/home/rit/Projects/genai .venv/bin/python eval_ensemble.py run_a.pt run_b.pt
V=$(.venv/bin/python -c "import torch;print('%.1f'%torch.load('run_b.pt',map_location='cpu',weights_only=False)['val'])")
.venv/bin/python predict.py run_a.pt run_b.pt --chunk 250 --out "subs/sub_ENSEMBLE_AB.csv"
echo "ENSEMBLE CSV READY (runB solo val ${V}) -> subs/sub_ENSEMBLE_AB.csv"
