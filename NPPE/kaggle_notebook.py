# ============================================================================
# Image Reconstruction Under Corruption - self-contained Kaggle notebook
# Paste as a single cell (or split at the ##### markers). GPU required.
#
# Approach: deterministic L2 restoration. The metric is MSE, whose Bayes-optimal
# predictor is the posterior mean E[clean|corrupt] - exactly what an L2-trained
# regressor converges to. A diffusion sampler would draw *from* that posterior,
# and a single sample carries ~2x the MSE of its mean, so regression is correct
# here, not a shortcut.
#
# Measured error budget on 8k training pairs (identity submission = 500 MSE):
#   rotated images      14.5% of set, mean MSE 1830  -> 53% of ALL error
#   non-rotated         85.5% of set, mean MSE  275
# Hence geometry is modelled explicitly rather than left to the UNet.
# ============================================================================
import os, sys, math, time, csv, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def find_root():
    """Locate the competition data under /kaggle/input without hard-coding the slug."""
    if os.environ.get('ROOT'): return os.environ['ROOT']
    import glob
    for pat in ('/kaggle/input/*/', '/kaggle/input/*/*/'):
        for d in sorted(glob.glob(pat)):
            if os.path.exists(d + 'train.csv') and os.path.isdir(d + 'test_corrupt'):
                return d
    raise SystemExit('could not find competition data under /kaggle/input/ - set ROOT manually')
ROOT = find_root()
OUT  = os.environ.get('OUT', '/kaggle/working/')
print('ROOT =', ROOT)
STEPS = int(os.environ.get('STEPS', 30000))    # ~9h on P100; drop to 6000 for a fast pass
SEED  = int(os.environ.get('SEED', 0))
# Kaggle kills a session at 12h. Stop training before that and still write a
# submission - an unfinished model scores, a killed session scores nothing.
TIME_BUDGET_H = float(os.environ.get('TIME_BUDGET_H', 10.5))
dev = 'cuda'
np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

##### ---------------------------------------------------------------- data
def load_split():
    rows = list(csv.DictReader(open(ROOT + 'train.csv')))
    n = len(rows)
    K = np.empty((n, 32, 32, 3), np.uint8); C = np.empty_like(K)
    def f(j):
        r = rows[j]
        K[j] = np.asarray(Image.open(ROOT + 'train_clean/' + r['clean_filename']).convert('RGB'))
        C[j] = np.asarray(Image.open(ROOT + 'train_corrupt/' + r['corrupt_filename']).convert('RGB'))
    with ThreadPoolExecutor(16) as ex: list(ex.map(f, range(n)))
    # Take the row order from sample_submission.csv rather than from listdir,
    # so the submission cannot silently mis-align with the expected order.
    ss = ROOT + 'sample_submission.csv'
    if os.path.exists(ss):
        with open(ss) as fh:
            ids = [r[0] for r in csv.reader(fh)][1:]
    else:
        ids = [x[:-4] for x in sorted(os.listdir(ROOT + 'test_corrupt')) if x.endswith('.png')]
    T = np.empty((len(ids), 32, 32, 3), np.uint8)
    def g(j): T[j] = np.asarray(Image.open(ROOT + 'test_corrupt/' + ids[j] + '.png').convert('RGB'))
    with ThreadPoolExecutor(16) as ex: list(ex.map(g, range(len(ids))))
    return K, C, T, ids

K_np, C_np, T_np, test_ids = load_split()
print('train', K_np.shape, 'test', T_np.shape)

##### ------------------------------------------------------- precision setup
cap = torch.cuda.get_device_capability()
if cap[0] >= 8:      AMP, ADT = True, torch.bfloat16      # Ampere+ (A100/L4)
elif cap[0] == 7:    AMP, ADT = True, torch.float16       # T4 / V100
else:                AMP, ADT = False, torch.float32      # P100 has no tensor cores
print('device', torch.cuda.get_device_name(0), 'amp', AMP, ADT)
try:
    scaler = torch.amp.GradScaler('cuda', enabled=(ADT is torch.float16))
except (AttributeError, TypeError):                      # older torch
    scaler = torch.cuda.amp.GradScaler(enabled=(ADT is torch.float16))

def safe_load(p):
    try:    return torch.load(p, map_location=dev, weights_only=False)
    except TypeError: return torch.load(p, map_location=dev)   # torch < 2.0

##### ----------------------------------------------- corruption simulator


def _rand(b, lo, hi, dev):
    return torch.rand(b, device=dev) * (hi - lo) + lo


def _bern(b, p, dev):
    return (torch.rand(b, device=dev) < p)


def rotate(x, theta_deg):
    """Rotate by theta degrees about centre, zero fill. Returns (rotated, validity)."""
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


def gauss_blur(x, sigma):
    """Per-image gaussian blur via a 5-tap separable kernel."""
    b = x.shape[0]
    r = torch.arange(-2, 3, device=x.device, dtype=x.dtype)
    k = torch.exp(-(r[None, :] ** 2) / (2 * sigma[:, None] ** 2 + 1e-8))
    k = k / k.sum(1, keepdim=True)                       # [B,5]
    xk = k.view(b, 1, 1, 5).expand(b, 3, 1, 5).reshape(b * 3, 1, 1, 5)
    yk = k.view(b, 1, 5, 1).expand(b, 3, 5, 1).reshape(b * 3, 1, 5, 1)
    v = x.reshape(1, b * 3, *x.shape[2:])
    v = F.conv2d(F.pad(v, (2, 2, 0, 0), mode='reflect'), xk, groups=b * 3)
    v = F.conv2d(F.pad(v, (0, 0, 2, 2), mode='reflect'), yk, groups=b * 3)
    return v.reshape(x.shape)


def rgb2hsv_shift(x, dh, ds):
    """Cheap hue rotation + saturation scale in YIQ-ish space (avoids full HSV)."""
    r, g, bl = x[:, 0], x[:, 1], x[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * bl
    i = 0.596 * r - 0.274 * g - 0.322 * bl
    q = 0.211 * r - 0.523 * g + 0.312 * bl
    c, s = torch.cos(dh)[:, None, None], torch.sin(dh)[:, None, None]
    sc = ds[:, None, None]
    i2, q2 = (i * c - q * s) * sc, (i * s + q * c) * sc
    return torch.stack([y + 0.956 * i2 + 0.621 * q2,
                        y - 0.272 * i2 - 0.647 * q2,
                        y - 1.106 * i2 + 1.703 * q2], 1)


def paint_boxes(x, active, sev, n_hi=5, s_lo=2, s_hi=10):
    """Solid-colour rectangular masks (CoarseDropout with random fill)."""
    b, _, H, W = x.shape
    dev = x.device
    out = x
    for j in range(n_hi):
        draw = active & (torch.rand(b, device=dev) < (0.25 + 0.6 * sev))
        span = (s_lo + (s_hi - s_lo) * sev).long().clamp(min=s_lo + 1)
        h = (torch.rand(b, device=dev) * (span - s_lo)).long() + s_lo
        w = (torch.rand(b, device=dev) * (span - s_lo)).long() + s_lo
        y0 = (torch.rand(b, device=dev) * (H - h).clamp(min=1)).long()
        x0 = (torch.rand(b, device=dev) * (W - w).clamp(min=1)).long()
        yy = torch.arange(H, device=dev)[None, :, None]
        xx = torch.arange(W, device=dev)[None, None, :]
        m = ((yy >= y0[:, None, None]) & (yy < (y0 + h)[:, None, None]) &
             (xx >= x0[:, None, None]) & (xx < (x0 + w)[:, None, None]))
        m = m & draw[:, None, None]
        # fill: mostly saturated random colour, sometimes black / grey
        col = torch.rand(b, 3, 1, 1, device=dev)
        kind = torch.rand(b, 1, 1, 1, device=dev)
        col = torch.where(kind < 0.25, torch.zeros_like(col), col)
        col = torch.where((kind >= 0.25) & (kind < 0.4),
                          torch.rand(b, 1, 1, 1, device=dev).expand(-1, 3, -1, -1), col)
        out = torch.where(m[:, None], col, out)
    return out


def impulse(x, active, dens):
    """Coloured salt-and-pepper: random pixels replaced by saturated colours."""
    b, _, H, W = x.shape
    dev = x.device
    hit = (torch.rand(b, 1, H, W, device=dev) < dens[:, None, None, None]) & active[:, None, None, None]
    col = torch.rand(b, 3, H, W, device=dev)
    col = torch.where(torch.rand(b, 3, 1, 1, device=dev) < 0.35, (col > 0.5).float(), col)
    return torch.where(hit, col, x)


def streaks(x, active, n=3):
    """Horizontal row corruption: a few rows shifted or replaced."""
    b, _, H, W = x.shape
    dev = x.device
    out = x
    for _ in range(n):
        draw = active & _bern(b, 0.5, dev)
        y0 = torch.randint(0, H - 1, (b,), device=dev)
        th = torch.randint(1, 3, (b,), device=dev)
        yy = torch.arange(H, device=dev)[None, :]
        m = ((yy >= y0[:, None]) & (yy < (y0 + th)[:, None])) & draw[:, None]
        shift = int(torch.randint(2, 7, (1,)).item())
        rolled = torch.roll(out, shift, dims=-1)
        out = torch.where(m[:, None, :, None], rolled, out)
    return out


DEFAULT_P = dict(rot=0.235, boxes=0.11, impulse=0.12, gnoise=0.45, blur=0.30,
                 down=0.12, bc=0.42, hue=0.18, streak=0.04, clean=0.03)

# Real per-image identity MSE spans ~4 orders of magnitude with a smooth ramp
# (10th pct 3.6, median 148, 90th pct 1379). Fixed magnitudes cannot produce
# that shape, so every op's strength is scaled by a per-image log-uniform
# severity s, which makes MSE roughly log-uniform too.
SEV_LO, SEV_HI = 0.085, 0.72


def corrupt(clean, p=None, sev=None):
    """clean: [B,3,32,32] float in [0,1]. -> (corrupt, theta_deg)"""
    P = dict(DEFAULT_P); P.update(p or {})
    x = clean
    b, dev = x.shape[0], x.device
    keep = _bern(b, P['clean'], dev)
    lo, hi = (sev or (SEV_LO, SEV_HI))
    s = torch.exp(_rand(b, math.log(lo), math.log(hi), dev))     # log-uniform severity
    s = s * (~keep).float()

    # ---- photometric (before geometry: real wedges are *exactly* 0, so no
    #      brightness bias may be added after the rotation) ----
    m = (_bern(b, P['bc'], dev) & ~keep).float()
    gain = 1 + m * s * _rand(b, -0.45, 0.25, dev)
    bias = m * s * _rand(b, -0.08, 0.28, dev)
    gam = 1 + m * s * _rand(b, -0.4, 0.55, dev)
    x = (x.clamp(0, 1) ** gam[:, None, None, None]) * gain[:, None, None, None] \
        + bias[:, None, None, None]
    x = x.clamp(0, 1)

    m = _bern(b, P['hue'], dev) & ~keep
    x = torch.where(m[:, None, None, None],
                    rgb2hsv_shift(x, s * _rand(b, -0.7, 0.7, dev),
                                  1 + s * _rand(b, -0.7, 0.7, dev)), x).clamp(0, 1)

    # ---- geometry ----
    do_rot = _bern(b, P['rot'], dev) & ~keep
    # rotation magnitude is independent of the other degradations in the real data:
    # observed angles cluster at 3-13 deg regardless of how noisy the image is
    sign = torch.where(_bern(b, 0.5, dev), 1.0, -1.0)
    theta = sign * _rand(b, 3.0, 14.5, dev) * do_rot.float()
    x, valid = rotate(x, theta)
    valid = (valid > 0.999).float()

    # ---- resolution loss ----
    m = _bern(b, P['blur'], dev) & ~keep
    xb = gauss_blur(x, 0.35 + 1.05 * s)
    x = torch.where(m[:, None, None, None], xb, x)

    m = _bern(b, P['down'], dev) & ~keep & (s > 0.45)
    xd = F.interpolate(F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False),
                       size=(32, 32), mode='bilinear', align_corners=False)
    x = torch.where(m[:, None, None, None], xd, x)

    # ---- painted-on degradations (after geometry: real noise sits over the wedge) ----
    x = paint_boxes(x, _bern(b, P['boxes'], dev) & ~keep, s)
    x = streaks(x, _bern(b, P['streak'], dev) & ~keep)

    m = (_bern(b, P['gnoise'], dev) & ~keep).float()
    x = x + torch.randn_like(x) * (m * s * 0.11)[:, None, None, None] * valid

    x = x + torch.randn_like(x) * ((~keep).float() * (0.004 + 0.010 * s))[:, None, None, None] * valid
    x = impulse(x, _bern(b, P['impulse'], dev) & ~keep, s * 0.055)

    x = x.clamp(0, 1)
    x = torch.where(keep[:, None, None, None], clean, x)
    theta = torch.where(keep, torch.zeros_like(theta), theta)
    x = (x * 255).round() / 255.0          # real data is stored as PNG
    return x, theta


##### ------------------------------------------------------------- model


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


##### ---------------------------------------------------------- training
K = torch.from_numpy(K_np).to(dev).permute(0, 3, 1, 2).contiguous()
C = torch.from_numpy(C_np).to(dev).permute(0, 3, 1, 2).contiguous()
N = K.shape[0]
g = torch.Generator().manual_seed(1234)
perm = torch.randperm(N, generator=g)
val_idx, tr_idx = perm[:3000].to(dev), perm[3000:].to(dev)
vC, vK = C[val_idx].float() / 255., K[val_idx].float() / 255.
v_rot = ((vC.sum(1) == 0).float().mean((1, 2)) > 0.02)
print(f'val identity MSE {(((vC - vK) * 255) ** 2).mean():.1f}  rotated {v_rot.float().mean():.3f}')

BS = int(os.environ.get('BS', 256))
EVB = int(os.environ.get('EVB', 500))    # eval chunk
LR, EMA_D, SYNTH = 2e-3, 0.9995, 0.5
net = Restorer().to(dev).to(memory_format=torch.channels_last)
ema = Restorer().to(dev).to(memory_format=torch.channels_last)
ema.load_state_dict(net.state_dict())
for q in ema.parameters(): q.requires_grad_(False)
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
print('params %.1fM' % (sum(q.numel() for q in net.parameters()) / 1e6))

def lr_at(i, warm=500):
    if i < warm: return LR * i / warm
    t = (i - warm) / max(1, STEPS - warm)
    return LR * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * t)))

@torch.no_grad()
def evaluate(m):
    m.eval(); tot = []
    for i in range(0, len(vC), EVB):
        x, y = vC[i:i + EVB], vK[i:i + EVB]
        with torch.autocast('cuda', ADT, enabled=AMP):
            pr, _ = m(x.to(memory_format=torch.channels_last))
        pr = (pr.float() * 255).round().clamp(0, 255) / 255.
        tot.append((((pr - y) * 255) ** 2).mean((1, 2, 3)))
    per = torch.cat(tot); m.train()
    return per.mean().item(), per[v_rot].mean().item(), per[~v_rot].mean().item()

X_test = torch.from_numpy(T_np).to(dev).permute(0, 3, 1, 2).float() / 255.

def write_submission(m, path=None):
    """Predict the test set with hflip TTA and write the CSV.

    Called periodically during training, not only at the end: a Kaggle session
    that dies mid-run then still leaves a valid, scoreable submission behind.
    """
    m.eval()
    acc = torch.zeros_like(X_test)
    for v in (lambda t: t, lambda t: t.flip(-1)):
        with torch.no_grad():
            for i in range(0, len(X_test), EVB):
                with torch.autocast('cuda', ADT, enabled=AMP):
                    pr, _ = m(v(X_test[i:i + EVB]))
                acc[i:i + EVB] += v(pr.float())
    m.train()
    flat = ((acc / 2).clamp(0, 1) * 255).round().clamp(0, 255).byte() \
             .permute(0, 2, 3, 1).reshape(len(X_test), -1).cpu().numpy()
    tmp = (path or (OUT + 'submission.csv')) + '.tmp'
    with open(tmp, 'w') as fh:
        fh.write('id,' + ','.join(f'pixel_{i}' for i in range(3072)) + '\n')
        for i, r in zip(test_ids, flat):
            fh.write(i + ',' + ','.join(map(str, r.tolist())) + '\n')
    os.replace(tmp, path or (OUT + 'submission.csv'))    # atomic: never a half file
    return flat.shape

n_syn = int(BS * SYNTH); n_real = BS - n_syn
best, t0 = 1e9, time.time()
deadline = t0 + TIME_BUDGET_H * 3600
last_sub = 0.0
print(f'training up to {STEPS} steps, hard stop after {TIME_BUDGET_H}h', flush=True)
for it in range(1, STEPS + 1):
    for gp in opt.param_groups: gp['lr'] = lr_at(it)
    ri = tr_idx[torch.randint(len(tr_idx), (n_real,), device=dev)]
    xr, yr = C[ri].float() / 255., K[ri].float() / 255.
    si = tr_idx[torch.randint(len(tr_idx), (n_syn,), device=dev)]
    ys = K[si].float() / 255.
    with torch.no_grad(): xs, ts = corrupt(ys)
    x = torch.cat([xr, xs]); y = torch.cat([yr, ys])
    if torch.rand(1).item() < 0.5: x, y, ts = x.flip(-1), y.flip(-1), -ts
    x = x.to(memory_format=torch.channels_last)
    with torch.autocast('cuda', ADT, enabled=AMP):
        pred, th = net(x)
        loss_mse = F.mse_loss(pred.float(), y)
        loss_ang = F.smooth_l1_loss(th[n_real:].float(), ts, beta=1.0)
        loss = loss_mse + 0.02 * loss_ang
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    scaler.step(opt); scaler.update()
    d = min(EMA_D, (it + 1) / (it + 10))
    with torch.no_grad():
        for pe, pn in zip(ema.parameters(), net.parameters()): pe.lerp_(pn.detach(), 1 - d)
        for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
    if it % 500 == 0 or it == STEPS:
        m_all, m_rot, m_non = evaluate(ema)
        print(f'[{it:6d}] {time.time()-t0:6.0f}s EMA val {m_all:7.2f} '
              f'(rot {m_rot:8.2f} / non {m_non:6.2f})', flush=True)
        if m_all < best:
            best = m_all
            torch.save({'ema': ema.state_dict(), 'val': m_all, 'step': it,
                        'seed': SEED}, OUT + 'ckpt.pt')
            # refresh the on-disk submission at most every 25 min (~30s to write)
            if time.time() - last_sub > 1500:
                write_submission(ema); last_sub = time.time()
                print(f'         submission.csv refreshed (val {best:.2f})', flush=True)
    if time.time() > deadline:
        print(f'TIME BUDGET reached at step {it} - stopping early', flush=True)
        break
print('BEST', best)

##### -------------------------------------------------------- submission
ck = safe_load(OUT + 'ckpt.pt')
best_net = Restorer().to(dev)
best_net.load_state_dict(ck['ema'])
print(f"loading best checkpoint: val {ck['val']:.2f} @ step {ck.get('step','?')}")
shape = write_submission(best_net)
print('wrote', OUT + 'submission.csv', shape)

# sanity-check the file we are about to submit
with open(OUT + 'submission.csv') as fh:
    rd = csv.reader(fh); hdr = next(rd); rows = sum(1 for _ in rd)
assert hdr[0] == 'id' and len(hdr) == 3073, hdr[:3]
assert rows == len(test_ids), (rows, len(test_ids))
print(f'VERIFIED: {rows} rows x {len(hdr)} cols, header ok')
