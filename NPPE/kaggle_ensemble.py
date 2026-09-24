# ============================================================================
# Ensemble notebook - run AFTER two or more training notebooks have finished.
#
# Averaging predictions from independently-seeded models strictly reduces MSE
# (Jensen): the squared error of a mean is never worse than the mean of the
# squared errors, and it is strictly better whenever the models' errors are not
# perfectly correlated. Combined with hflip TTA this is typically 5-10% on top
# of the best single model, for zero extra training.
#
# Setup:
#   1. In each training notebook, /kaggle/working/ckpt.pt is saved automatically.
#      "Save Version" the notebook, then add its output as a Dataset here.
#   2. List the checkpoint paths in CKPTS below.
#   3. Run. Writes /kaggle/working/submission.csv.
#
# Everything stays inside Kaggle, so this remains a self-contained submission.
# ============================================================================
import os, math, csv, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def find_root():
    """Locate the competition data under /kaggle/input without hard-coding the slug."""
    if os.environ.get('ROOT'): return os.environ['ROOT']
    import glob
    for d in sorted(glob.glob('/kaggle/input/*/')):
        if os.path.exists(d + 'train.csv') and os.path.isdir(d + 'test_corrupt'):
            return d
    raise SystemExit('could not find competition data under /kaggle/input/ - set ROOT manually')
ROOT  = find_root()
OUT   = os.environ.get('OUT', '/kaggle/working/')
print('ROOT  =', ROOT)
CKPTS = [                      # <-- one entry per trained model
    '/kaggle/input/<RUN-A-DATASET>/ckpt.pt',
    '/kaggle/input/<RUN-B-DATASET>/ckpt.pt',
]
dev = 'cuda'
cap = torch.cuda.get_device_capability()
if   cap[0] >= 8: AMP, ADT = True, torch.bfloat16
elif cap[0] == 7: AMP, ADT = True, torch.float16
else:             AMP, ADT = False, torch.float32
EVB = 500

##### ------------------------------- model definition (must match training)
def warp(x, theta_deg):
    b = x.shape[0]
    t = theta_deg * math.pi / 180.0
    cos, sin = torch.cos(t), torch.sin(t)
    mat = torch.zeros(b, 2, 3, device=x.device, dtype=x.dtype)
    mat[:, 0, 0], mat[:, 0, 1] = cos, -sin
    mat[:, 1, 0], mat[:, 1, 1] = sin, cos
    grid = F.affine_grid(mat, x.shape, align_corners=False)
    out = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    ones = torch.ones(b, 1, *x.shape[2:], device=x.device, dtype=x.dtype)
    val = F.grid_sample(ones, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return out, val


class AngleHead(nn.Module):
    """Predicts rotation in degrees. Gets exact labels from the simulator."""
    def __init__(self, c=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, c, 3, 1, 1), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, c, 3, 2, 1), nn.GroupNorm(8, c), nn.SiLU(),        # 16
            nn.Conv2d(c, c * 2, 3, 1, 1), nn.GroupNorm(8, c * 2), nn.SiLU(),
            nn.Conv2d(c * 2, c * 2, 3, 2, 1), nn.GroupNorm(8, c * 2), nn.SiLU(),  # 8
            nn.Conv2d(c * 2, c * 2, 3, 1, 1), nn.GroupNorm(8, c * 2), nn.SiLU(),
        )
        self.fc = nn.Linear(c * 4, 1)
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        h = self.net(x)
        h = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
        return self.fc(h).squeeze(1) * 15.0        # scaled to the plausible range


class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ci); self.c1 = nn.Conv2d(ci, co, 3, 1, 1)
        self.n2 = nn.GroupNorm(8, co); self.c2 = nn.Conv2d(co, co, 3, 1, 1)
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.n = nn.GroupNorm(8, c); self.qkv = nn.Conv2d(c, c * 3, 1)
        self.proj = nn.Conv2d(c, c, 1); self.h = heads
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(b, 3, self.h, c // self.h, H * W).unbind(1)
        o = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2),
                                           v.transpose(-1, -2))
        return x + self.proj(o.transpose(-1, -2).reshape(b, c, H, W))


class UNet(nn.Module):
    def __init__(self, cin=8, w=(128, 256, 512), nb=2):
        super().__init__()
        w1, w2, w3 = w
        self.stem = nn.Conv2d(cin, w1, 3, 1, 1)
        self.e1 = nn.ModuleList([Block(w1, w1) for _ in range(nb)])
        self.d1 = nn.Conv2d(w1, w2, 3, 2, 1)
        self.e2 = nn.ModuleList([Block(w2, w2) for _ in range(nb)])
        self.a2 = Attn(w2)
        self.d2 = nn.Conv2d(w2, w3, 3, 2, 1)
        self.mid = nn.ModuleList([Block(w3, w3), Attn(w3), Block(w3, w3)])
        self.u2 = nn.Conv2d(w3, w2, 3, 1, 1)
        self.f2 = nn.ModuleList([Block(w2 * 2, w2)] + [Block(w2, w2) for _ in range(nb - 1)])
        self.b2 = Attn(w2)
        self.u1 = nn.Conv2d(w2, w1, 3, 1, 1)
        self.f1 = nn.ModuleList([Block(w1 * 2, w1)] + [Block(w1, w1) for _ in range(nb - 1)])
        self.out_n = nn.GroupNorm(8, w1); self.out = nn.Conv2d(w1, 3, 3, 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # start at identity

    def forward(self, x):
        h1 = self.stem(x)
        for b in self.e1: h1 = b(h1)
        h2 = self.d1(h1)
        for b in self.e2: h2 = b(h2)
        h2 = self.a2(h2)
        h3 = self.d2(h2)
        for m in self.mid: h3 = m(h3)
        g2 = self.u2(F.interpolate(h3, scale_factor=2, mode='nearest'))
        g2 = torch.cat([g2, h2], 1)
        for b in self.f2: g2 = b(g2)
        g2 = self.b2(g2)
        g1 = self.u1(F.interpolate(g2, scale_factor=2, mode='nearest'))
        g1 = torch.cat([g1, h1], 1)
        for b in self.f1: g1 = b(g1)
        return self.out(F.silu(self.out_n(g1)))


class Restorer(nn.Module):
    def __init__(self, w=(128, 256, 512), nb=2):
        super().__init__()
        self.angle = AngleHead()
        self.unet = UNet(cin=8, w=w, nb=nb)

    def forward(self, x):
        """x: corrupt in [0,1], [B,3,32,32]. -> (pred_clean, pred_theta)"""
        blk = (x.sum(1, keepdim=True) < 0.012).float()      # the rotation wedge
        th = self.angle(torch.cat([x, blk], 1))
        # warp with a detached angle: the head learns from its own exact
        # supervision, not from noisy grid_sample gradients
        wx, val = warp(x, -th.detach())
        inp = torch.cat([x, blk, wx, val], 1)
        return (x + self.unet(inp)).clamp(0, 1), th


##### ------------------------------------------------------------- data
# row order comes from sample_submission.csv so it cannot mis-align
ss = ROOT + 'sample_submission.csv'
if os.path.exists(ss):
    with open(ss) as fh: test_ids = [r[0] for r in csv.reader(fh)][1:]
else:
    test_ids = [x[:-4] for x in sorted(os.listdir(ROOT + 'test_corrupt')) if x.endswith('.png')]
T = np.empty((len(test_ids), 32, 32, 3), np.uint8)
def g(j): T[j] = np.asarray(Image.open(ROOT + 'test_corrupt/' + test_ids[j] + '.png').convert('RGB'))
with ThreadPoolExecutor(16) as ex: list(ex.map(g, range(len(test_ids))))
X = torch.from_numpy(T).to(dev).permute(0, 3, 1, 2).float() / 255.
print('test', X.shape)

##### -------------------------------------------------- validation check
# Score the ensemble on the same held-out split the training notebooks used,
# so you can confirm the ensemble really beats its best member before submitting.
rows = list(csv.DictReader(open(ROOT + 'train.csv')))
n = len(rows)
K = np.empty((n, 32, 32, 3), np.uint8); C = np.empty_like(K)
def f(j):
    r = rows[j]
    K[j] = np.asarray(Image.open(ROOT + 'train_clean/' + r['clean_filename']).convert('RGB'))
    C[j] = np.asarray(Image.open(ROOT + 'train_corrupt/' + r['corrupt_filename']).convert('RGB'))
with ThreadPoolExecutor(16) as ex: list(ex.map(f, range(n)))
gen = torch.Generator().manual_seed(1234)              # identical split to training
vi = torch.randperm(n, generator=gen)[:3000]
vC = torch.from_numpy(C[vi.numpy()]).to(dev).permute(0, 3, 1, 2).float() / 255.
vK = torch.from_numpy(K[vi.numpy()]).to(dev).permute(0, 3, 1, 2).float() / 255.

def predict(net, src):
    acc = torch.zeros_like(src)
    for v in (lambda t: t, lambda t: t.flip(-1)):      # hflip TTA
        with torch.no_grad():
            for i in range(0, len(src), EVB):
                with torch.autocast('cuda', ADT, enabled=AMP):
                    pr, _ = net(v(src[i:i + EVB]))
                acc[i:i + EVB] += v(pr.float())
    return acc / 2

def score(p):
    p = (p * 255).round().clamp(0, 255) / 255.
    return ((((p - vK) * 255) ** 2).mean()).item()

vacc = torch.zeros_like(vC); tacc = torch.zeros_like(X); nm = 0
CKPTS = [c for c in CKPTS if os.path.exists(c)]
assert CKPTS, 'no checkpoints found - attach the training notebook outputs as Datasets'
for c in CKPTS:
    try:    ck = torch.load(c, map_location=dev, weights_only=False)
    except TypeError: ck = torch.load(c, map_location=dev)
    net = Restorer().to(dev); net.load_state_dict(ck['ema']); net.eval()
    pv = predict(net, vC); vacc += pv
    tacc += predict(net, X); nm += 1
    print(f'{os.path.basename(os.path.dirname(c)):30s} solo val {score(pv):6.2f}   '
          f'running ensemble ({nm}) {score(vacc / nm):6.2f}', flush=True)
print(f'\nFINAL ENSEMBLE val {score(vacc / nm):.2f}  ({nm} models, hflip TTA)')

##### -------------------------------------------------------- submission
pred = (tacc / nm).clamp(0, 1)
flat = (pred * 255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).reshape(len(X), -1).cpu().numpy()
with open(OUT + 'submission.csv', 'w') as fh:
    fh.write('id,' + ','.join(f'pixel_{i}' for i in range(3072)) + '\n')
    for i, r in zip(test_ids, flat):
        fh.write(i + ',' + ','.join(map(str, r.tolist())) + '\n')
print('wrote submission.csv', flat.shape)
with open(OUT + 'submission.csv') as fh:
    rd = csv.reader(fh); hdr = next(rd); nrow = sum(1 for _ in rd)
assert hdr[0] == 'id' and len(hdr) == 3073 and nrow == len(test_ids)
print(f'VERIFIED: {nrow} rows x {len(hdr)} cols')
