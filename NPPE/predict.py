"""Write submission.csv from one or more checkpoints.

Averaging predictions (across seeds and across the hflip TTA pair) strictly
reduces MSE by Jensen, so it is a free gain - always ensemble.
Flattening is HWC row-major, verified byte-exact against sample_submission.csv.
"""
import argparse, numpy as np, torch, model

p = argparse.ArgumentParser()
p.add_argument('ckpts', nargs='+')
p.add_argument('--out', default='submission.csv')
p.add_argument('--tta', action='store_true', default=True)
p.add_argument('--source', default='cache/test_corrupt.npy')
p.add_argument('--ids', default='cache/test_ids.txt')
p.add_argument('--chunk', type=int, default=1000)
a = p.parse_args()

dev = 'cuda'
X = torch.from_numpy(np.load(a.source)).to(dev).permute(0, 3, 1, 2).float() / 255.
ids = open(a.ids).read().split()
assert len(ids) == len(X), (len(ids), len(X))

acc = torch.zeros_like(X)
nm = 0
for c in a.ckpts:
    ck = torch.load(c, map_location=dev)
    w = ck['args']['width']; nb = ck['args']['nb']
    net = model.Restorer(w=tuple(w), nb=nb).to(dev)
    net.load_state_dict(ck['ema']); net.eval()
    print(f'{c}: val {ck["val"]:.2f}')
    views = [lambda t: t, (lambda t: t.flip(-1))] if a.tta else [lambda t: t]
    for v in views:
        with torch.no_grad():
            for i in range(0, len(X), a.chunk):
                xb = v(X[i:i + a.chunk])
                with torch.autocast('cuda', torch.bfloat16):
                    pr, _ = net(xb)
                acc[i:i + a.chunk] += v(pr.float())
        nm += 1
pred = (acc / nm).clamp(0, 1)
flat = (pred * 255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).reshape(len(X), -1).cpu().numpy()
hdr = 'id,' + ','.join(f'pixel_{i}' for i in range(3072))
with open(a.out, 'w') as f:
    f.write(hdr + '\n')
    for i, r in zip(ids, flat):
        f.write(i + ',' + ','.join(map(str, r.tolist())) + '\n')
print(f'wrote {a.out}  ({nm} model-views averaged)')
