"""Measure the actual ensemble gain on the held-out split.

Reports each model solo and the running average, so the benefit of adding a
second model is a measurement rather than a rule of thumb.
"""
import sys, numpy as np, torch, model
dev = 'cuda'
K = torch.from_numpy(np.load('cache/train_clean.npy')).permute(0, 3, 1, 2).contiguous()
C = torch.from_numpy(np.load('cache/train_corrupt.npy')).permute(0, 3, 1, 2).contiguous()
g = torch.Generator().manual_seed(1234)
vi = torch.randperm(K.shape[0], generator=g)[:3000]
vC = C[vi].to(dev).float() / 255.; vK = K[vi].to(dev).float() / 255.
rot = ((vC.sum(1) == 0).float().mean((1, 2)) > 0.02)

def predict(net, src, tta=True):
    acc = torch.zeros_like(src)
    views = [lambda t: t, lambda t: t.flip(-1)] if tta else [lambda t: t]
    for v in views:
        with torch.no_grad():
            for i in range(0, len(src), 250):
                with torch.autocast('cuda', torch.bfloat16):
                    pr, _ = net(v(src[i:i + 250]))
                acc[i:i + 250] += v(pr.float())
    return acc / len(views)

def score(p):
    p = (p * 255).round().clamp(0, 255) / 255.
    per = (((p - vK) * 255) ** 2).mean((1, 2, 3))
    return per.mean().item(), per[rot].mean().item(), per[~rot].mean().item()

acc = torch.zeros_like(vC); n = 0
for c in sys.argv[1:]:
    ck = torch.load(c, map_location=dev)
    net = model.Restorer(w=tuple(ck['args']['width']), nb=ck['args']['nb']).to(dev)
    net.load_state_dict(ck['ema']); net.eval()
    p_notta = predict(net, vC, tta=False)
    p = predict(net, vC)
    acc += p; n += 1
    s_n, s_t, s_e = score(p_notta)[0], score(p)[0], score(acc / n)[0]
    a, r, nr = score(acc / n)
    print(f'{c:12s} solo {s_n:6.2f}  +hflipTTA {s_t:6.2f}  | ensemble({n}) {s_e:6.2f} '
          f'(rot {r:7.2f} / non {nr:6.2f})', flush=True)
print(f'\nBEST ENSEMBLE {score(acc / n)[0]:.2f}  -> expected LB ~{score(acc/n)[0]*1.064:.1f}')
