"""Check the synthetic corruption pipeline reproduces the real train_corrupt statistics.

The sharpest single test is the per-image identity-MSE percentile curve: it encodes
how often each severity of degradation fires. Match that and the simulator is honest.
"""
import numpy as np, torch, sim

dev = 'cuda'
K = np.load('cache/train_clean.npy'); C = np.load('cache/train_corrupt.npy')
rs = np.random.RandomState(1); idx = rs.choice(len(K), 8000, replace=False)
Kr, Cr = K[idx].astype(np.float32), C[idx].astype(np.float32)

def stats(clean, corrupt):
    per = ((clean - corrupt) ** 2).mean(axis=(1, 2, 3))
    blk = (corrupt.sum(-1) == 0).mean(axis=(1, 2))
    tv = (np.abs(np.diff(corrupt, axis=1)).mean(axis=(1, 2, 3)) +
          np.abs(np.diff(corrupt, axis=2)).mean(axis=(1, 2, 3)))
    sat = (corrupt.max(-1) - corrupt.min(-1)).mean()
    return per, blk, tv, sat

def report(name, clean, corrupt):
    per, blk, tv, sat = stats(clean, corrupt)
    q = np.percentile(per, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    print(f'{name:10s} meanMSE {per.mean():7.1f} | pct ' +
          ' '.join(f'{v:7.1f}' for v in q))
    print(f'{"":10s} blkfrac {blk.mean():.4f} rot%({(blk>0.02).mean()*100:4.1f}) '
          f'TV {tv.mean():5.2f} sat {sat:5.1f}')
    return per

# real
per_real = report('REAL', Kr, Cr)
# synthetic, from the same clean images
kt = torch.from_numpy(Kr).permute(0, 3, 1, 2).to(dev) / 255.
torch.manual_seed(0)
outs = []
for i in range(0, len(kt), 2000):
    c, th = sim.corrupt(kt[i:i + 2000])
    outs.append((c * 255).permute(0, 2, 3, 1).cpu().numpy())
Cs = np.concatenate(outs)
per_syn = report('SYNTH', Kr, Cs)
print()
print('mean ratio synth/real %.2f' % (per_syn.mean() / per_real.mean()))
qr = np.percentile(per_real, [10, 25, 50, 75, 90, 99])
qs = np.percentile(per_syn, [10, 25, 50, 75, 90, 99])
print('percentile ratios (want ~1.0):', np.round(qs / np.maximum(qr, 1e-6), 2))
