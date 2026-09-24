"""Train the restorer. All data lives on the GPU as uint8; batches are built by
indexing, so there is no dataloader in the loop.

Each batch is half real (corrupt, clean) pairs and half freshly-simulated pairs.
The synthetic half is what supplies exact rotation labels for the angle head and
what stops the model memorising the 37k fixed real pairs.
"""
import argparse, math, time, numpy as np, torch, torch.nn.functional as F
import sim, model

p = argparse.ArgumentParser()
p.add_argument('--steps', type=int, default=30000)
p.add_argument('--bs', type=int, default=256)
p.add_argument('--lr', type=float, default=2e-3)
p.add_argument('--synth', type=float, default=0.5, help='fraction of each batch that is simulated')
p.add_argument('--width', type=int, nargs=3, default=[128, 256, 512])
p.add_argument('--nb', type=int, default=2)
p.add_argument('--ema', type=float, default=0.9995)
p.add_argument('--val_every', type=int, default=1000)
p.add_argument('--out', type=str, default='ckpt.pt')
p.add_argument('--seed', type=int, default=0)
a = p.parse_args()

dev = 'cuda'
torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

K = torch.from_numpy(np.load('cache/train_clean.npy')).to(dev).permute(0, 3, 1, 2).contiguous()
C = torch.from_numpy(np.load('cache/train_corrupt.npy')).to(dev).permute(0, 3, 1, 2).contiguous()
N = K.shape[0]
g = torch.Generator().manual_seed(1234)            # fixed split, identical across runs
perm = torch.randperm(N, generator=g)
val_idx, tr_idx = perm[:3000].to(dev), perm[3000:].to(dev)
print(f'train {len(tr_idx)}  val {len(val_idx)}')

# validation buckets: rotated images carry ~53% of the error, so track them apart
vC, vK = C[val_idx].float() / 255., K[val_idx].float() / 255.
v_rot = ((vC.sum(1) == 0).float().mean((1, 2)) > 0.02)
print(f'val rotated fraction {v_rot.float().mean():.3f}')
print(f'val identity MSE {(((vC - vK) * 255) ** 2).mean():.1f}')

net = model.Restorer(w=tuple(a.width), nb=a.nb).to(dev).to(memory_format=torch.channels_last)
ema = model.Restorer(w=tuple(a.width), nb=a.nb).to(dev).to(memory_format=torch.channels_last)
ema.load_state_dict(net.state_dict())
for q in ema.parameters(): q.requires_grad_(False)
print('params %.1fM' % (sum(q.numel() for q in net.parameters()) / 1e6))

opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
warm = 500
def lr_at(i):
    if i < warm: return a.lr * i / warm
    t = (i - warm) / max(1, a.steps - warm)
    return a.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * t)))

n_syn = int(a.bs * a.synth); n_real = a.bs - n_syn

@torch.no_grad()
def evaluate(m):
    m.eval(); tot = []
    for i in range(0, len(vC), 500):
        x, y = vC[i:i + 500], vK[i:i + 500]
        with torch.autocast('cuda', torch.bfloat16):
            pr, _ = m(x.to(memory_format=torch.channels_last))
        pr = (pr.float() * 255).round().clamp(0, 255) / 255.     # score the submitted uint8
        tot.append((((pr - y) * 255) ** 2).mean((1, 2, 3)))
    per = torch.cat(tot); m.train()
    return per.mean().item(), per[v_rot].mean().item(), per[~v_rot].mean().item()

best = 1e9; t0 = time.time()
for it in range(1, a.steps + 1):
    for gp in opt.param_groups: gp['lr'] = lr_at(it)
    ri = tr_idx[torch.randint(len(tr_idx), (n_real,), device=dev)]
    xr, yr = C[ri].float() / 255., K[ri].float() / 255.
    si = tr_idx[torch.randint(len(tr_idx), (n_syn,), device=dev)]
    ys = K[si].float() / 255.
    with torch.no_grad():
        xs, ts = sim.corrupt(ys)
    x = torch.cat([xr, xs]); y = torch.cat([yr, ys])
    # horizontal flip is label-preserving for the pair (angle flips sign)
    if torch.rand(1).item() < 0.5:
        x, y, ts = x.flip(-1), y.flip(-1), -ts
    x = x.to(memory_format=torch.channels_last)

    with torch.autocast('cuda', torch.bfloat16):
        pred, th = net(x)
        loss_mse = F.mse_loss(pred.float(), y)
        loss_ang = F.smooth_l1_loss(th[n_real:].float(), ts, beta=1.0)
        loss = loss_mse + 0.02 * loss_ang
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()

    d = min(a.ema, (it + 1) / (it + 10))
    with torch.no_grad():
        for pe, pn in zip(ema.parameters(), net.parameters()): pe.lerp_(pn.detach(), 1 - d)
        for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)

    if it % 50 == 0 and it % a.val_every != 0:
        print(f'  {it:6d} {time.time()-t0:6.0f}s  {(time.time()-t0)/it*1000:5.0f}ms/step  '
              f'trainMSE255 {loss_mse.item()*255**2:7.1f} ang {loss_ang.item():5.2f}', flush=True)
    if it % a.val_every == 0 or it == a.steps:
        m_all, m_rot, m_non = evaluate(ema)
        r_all, _, _ = evaluate(net)
        print(f'[{it:6d}] {time.time()-t0:6.0f}s lr {lr_at(it):.2e} '
              f'trainMSE255 {loss_mse.item()*255**2:7.1f} ang {loss_ang.item():5.2f} | '
              f'EMA val {m_all:7.2f} (rot {m_rot:8.2f} / non {m_non:6.2f})  raw {r_all:7.2f}',
              flush=True)
        if m_all < best:
            best = m_all
            torch.save({'ema': ema.state_dict(), 'net': net.state_dict(),
                        'args': vars(a), 'val': m_all}, a.out)
print(f'BEST {best:.2f} -> {a.out}')
