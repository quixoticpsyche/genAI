"""Emit a submission CSV whenever the running model improves enough to be worth
one of the day's limited submissions.

Gating on *relative* improvement (not a fixed clock) means each CSV that lands
is meaningfully better than the last one submitted, so no slot is spent on noise.
"""
import os, re, time, shutil, subprocess, sys, torch

LOG, CKPT, OUTDIR = 'run_a.log', 'run_a.pt', 'subs/'
MIN_REL_GAIN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.035   # 3.5% better
MIN_GAP_S    = float(sys.argv[2]) if len(sys.argv) > 2 else 540     # >=9 min apart
DEADLINE     = time.time() + 3.1 * 3600

last_val, last_t = float(sys.argv[3]) if len(sys.argv) > 3 else 75.8, 0.0

def cur_step():
    try:
        m = re.findall(r'^\[ *(\d+)\]', open(LOG).read(), re.M)
        return int(m[-1]) if m else 0
    except Exception:
        return 0

print(f'watching (baseline val {last_val:.2f}, need {MIN_REL_GAIN*100:.1f}% gain)', flush=True)
while time.time() < DEADLINE:
    time.sleep(20)
    if not os.path.exists(CKPT):
        continue
    try:
        val = torch.load(CKPT, map_location='cpu', weights_only=False)['val']
    except Exception:
        continue                      # checkpoint mid-write
    now = time.time()
    if val > last_val * (1 - MIN_REL_GAIN) or (now - last_t) < MIN_GAP_S:
        continue
    step = cur_step()
    snap = f'{OUTDIR}ck_s{step:05d}.pt'
    shutil.copy(CKPT, snap)
    out = f'{OUTDIR}sub_s{step:05d}_val{val:.1f}.csv'
    r = subprocess.run([sys.executable, 'predict.py', snap, '--chunk', '250', '--out', out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f'PREDICT FAILED step {step}: {r.stderr.strip()[-300:]}', flush=True)
        continue
    os.remove(snap)
    print(f'CSV READY  step {step:6d}  val {val:6.2f}  expected LB ~{val*1.064:5.1f}  ->  {out}', flush=True)
    last_val, last_t = val, now
print('watcher done', flush=True)
