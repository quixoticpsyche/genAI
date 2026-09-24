#!/bin/bash
cd /home/rit/Projects/genai
while pgrep -f "train.py --steps 20000" > /dev/null; do sleep 60; done
# seed + synth-mix diversity makes the pair ensemble better than two identical runs
nohup .venv/bin/python -u train.py --steps 20000 --val_every 500 --bs 256 \
      --seed 1 --synth 0.62 --out run_b.pt > run_b.log 2>&1
