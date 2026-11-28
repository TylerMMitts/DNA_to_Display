# Deterministic training and output checks. The QC stage and the dashboard both
# call these, so a problem shows up as an alert while a run is still training.
#
# Each check returns issues as {'check', 'severity' (info/warning/fail),
# 'detail', 'action'}. The thresholds come from failures seen in this project:
#   - LiteVAE collapse (mice, epoch 14): loss jumped 250x in one epoch and the
#     decoder then drew one flat grey image; plain MSE ended above the all-white
#     baseline, and the latent spread shrank as the encoder gave up.
#   - LiteVAE white collapse (WiDiv v1): every image drawn white, plain MSE equal
#     to the all-white score from epoch 2.
#   - memorisation (209M root model): training genotypes 22% better with their
#     own genotype than another, held-out +0.1%.
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .project import ROOT, read_json


def find_runs(ds):
    """Training runs of this dataset: LiteVAE and diffusion, by their history files."""
    base = ds.root / 'results' / 'training'
    runs = []
    if not base.exists():
        return runs
    for d in sorted(base.iterdir()):
        if not d.is_dir() or ds.name not in d.name:
            continue
        if (d / 'training_history.json').exists():
            runs.append({'name': d.name, 'kind': 'litevae', 'dir': d, 'history': d / 'training_history.json'})
        elif (d / 'training_history.csv').exists():
            runs.append({'name': d.name, 'kind': 'diffusion', 'dir': d, 'history': d / 'training_history.csv'})
    return runs


def load_history(run):
    if run['kind'] == 'litevae':
        return pd.DataFrame(read_json(run['history'], []))
    try:
        return pd.read_csv(run['history'])
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()


def issue(check, severity, detail, action):
    return {'check': check, 'severity': severity, 'detail': detail, 'action': action}


def check_litevae(h):
    out = []
    if h.empty or 'train_loss' not in h:
        return out
    loss = h.train_loss.astype(float)
    if loss.isna().any() or np.isinf(loss).any():
        out.append(issue('non-finite loss', 'fail', 'the training loss became NaN or infinite',
                         'restart from the last good checkpoint with a lower learning rate'))
    med = loss.rolling(5, min_periods=3).median().shift(1)
    spikes = h.epoch[(loss > 10 * med)].tolist()
    if spikes:
        out.append(issue('loss spike', 'warning', f'training loss jumped over 10x its recent median at epochs {spikes[:6]}',
                         'check the spike guard log for skipped steps; look for unusual images in that batch'))
    if {'val_plain_mse', 'all_white_mse'} <= set(h.columns):
        white = float(h.all_white_mse.iloc[-1])
        best = float(h.val_plain_mse.min())
        last = float(h.val_plain_mse.iloc[-1])
        if best < 0.3 * white and last > 0.8 * white:
            ep = int(h.epoch.iloc[int(np.argmax(h.val_plain_mse.to_numpy() > 0.8 * white))])
            out.append(issue('collapse', 'fail',
                             f'reconstruction error went from {best:.4f} to {last:.4f} (all-white {white:.4f}) '
                             f'around epoch {ep}: the decoder now draws a flat image',
                             f'stop the job; restart from the checkpoint before epoch {ep}; remove flagged input images'))
        elif last > 0.8 * white and len(h) > 3:
            out.append(issue('never learned', 'fail', f'reconstruction error {last:.4f} is near the all-white '
                             f'score {white:.4f}', 'check the background handling and the learning-rate warm-up'))
    if 'root_latent_std' in h and len(h) > 5:
        s = h.root_latent_std.astype(float)
        if s.iloc[-1] < 0.5 * s.max():
            out.append(issue('latent shrinking', 'warning', f'latent spread fell from {s.max():.2f} to {s.iloc[-1]:.2f}',
                             'the decoder may be ignoring the latent; check reconstructions'))
    return out


def check_diffusion(h):
    out = []
    if h.empty or 'val_loss' not in h:
        return out
    v = h.val_loss.astype(float)
    if v.isna().any():
        out.append(issue('non-finite loss', 'fail', 'validation loss became NaN', 'restart from the last good checkpoint'))
        return out
    best_i = int(v.idxmin())
    if len(v) > 20 and v.iloc[-1] > v.min() * 1.1:
        out.append(issue('diverging', 'warning', f'validation loss {v.iloc[-1]:.5f} is 10% above its best '
                         f'{v.min():.5f} (epoch {int(h.epoch[best_i])})', 'use the best checkpoint; lower the learning rate'))
    if {'genotype_gain_heldout', 'genotype_gain_trained'} <= set(h.columns):
        gh, gt = float(h.genotype_gain_heldout.iloc[-1]), float(h.genotype_gain_trained.iloc[-1])
        if gh > 0 and gt / gh > 1.5:
            out.append(issue('memorisation', 'warning', f'genotype gain on training lines is {gt / gh:.1f}x the '
                             'held-out gain', 'prefer an earlier checkpoint or a smaller model'))
        if gh <= 0 and len(h) > 30:
            out.append(issue('no genotype signal', 'fail', 'held-out genotype gain is not above zero',
                             'the model ignores the genotype: check the encoding and that lines differ in the images'))
    if {'train_eval_loss', 'val_loss'} <= set(h.columns):
        gap = float(h.val_loss.iloc[-1] / h.train_eval_loss.iloc[-1] - 1)
        if gap > 0.05:
            out.append(issue('overfitting', 'warning', f'validation loss is {gap:.0%} above training loss',
                             'stop at the best epoch; more data or a smaller model'))
    if 'lr' in h and len(h) > 20:
        tail = v.iloc[-max(5, len(v) // 7):]
        if float(h.lr.iloc[-1]) < 1e-6 and (tail.max() - tail.min()) / tail.mean() < 0.002:
            out.append(issue('converged', 'info', 'the learning rate has decayed and validation loss is flat',
                             'training is finished; evaluate the best checkpoint'))
    return out


def summary(h, kind):
    if h.empty:
        return {}
    if kind == 'diffusion' and 'val_loss' in h:
        i = int(h.val_loss.idxmin())
        return {'epochs': int(h.epoch.max()), 'best_epoch': int(h.epoch[i]), 'best_val_loss': float(h.val_loss[i]),
                'gain_heldout': float(h.genotype_gain_heldout.iloc[-1]) if 'genotype_gain_heldout' in h else None}
    if kind == 'litevae' and 'val_loss' in h:
        i = int(h.val_loss.idxmin())
        return {'epochs': int(h.epoch.max()), 'best_epoch': int(h.epoch[i]), 'best_val_loss': float(h.val_loss[i])}
    return {'epochs': int(h.epoch.max()) if 'epoch' in h else len(h)}


# --- generated images ------------------------------------------------------------

def _thumbs(paths, size=32):
    return np.stack([np.asarray(Image.open(p).convert('L').resize((size, size)), dtype=np.float32).ravel() / 255
                     for p in paths])


def check_images(generated, real, train_real=None):
    """Generated images against real ones: flat, washed out, collapsed, copied."""
    out = []
    generated, real = list(generated), list(real)
    if not generated or not real:
        return out
    g, r = _thumbs(generated), _thumbs(real)
    flat = (g.std(axis=1) < 0.02).mean()
    if flat > 0.05:
        out.append(issue('flat images', 'fail', f'{flat:.0%} of generated images are nearly uniform',
                         'the decoder or the diffusion model has collapsed; see the LiteVAE checks'))
    gm, rm, rs = g.mean(axis=1), r.mean(axis=1).mean(), r.mean(axis=1).std()
    off = (np.abs(gm - rm) > 3 * max(rs, 1e-3)).mean()
    if off > 0.1:
        out.append(issue('brightness off', 'warning', f'{off:.0%} of generated images are far brighter or darker '
                         'than real ones', 'check for washed-out or inverted outputs'))
    if len(g) >= 8:
        d = np.sqrt(((g[:, None] - g[None]) ** 2).mean(axis=2))
        np.fill_diagonal(d, np.inf)
        dr = np.sqrt(((r[:, None] - r[None]) ** 2).mean(axis=2)) if len(r) >= 8 else None
        if dr is not None:
            np.fill_diagonal(dr, np.inf)
            if np.median(d.min(axis=1)) < 0.3 * np.median(dr.min(axis=1)):
                out.append(issue('mode collapse', 'warning', 'generated images are far more alike than real ones',
                                 'genotypes may be ignored; check the genotype gain'))
    if train_real is not None and len(train_real):
        t = _thumbs(train_real)
        dg = np.sqrt(((g[:, None] - t[None]) ** 2).mean(axis=2)).min(axis=1)
        rr = np.sqrt(((r[:, None] - t[None]) ** 2).mean(axis=2))
        rr[rr < 1e-6] = np.inf                 # a real image is not its own neighbour
        base = np.median(rr.min(axis=1))
        copied = (dg < 0.25 * base).mean()
        if copied > 0.05:
            out.append(issue('copies of training images', 'warning',
                             f'{copied:.0%} of generated images are near-copies of a training image',
                             'memorisation: use an earlier checkpoint or a smaller model'))
    return out


def latest_previews(run, n=12):
    d = Path(run['dir']) / 'previews'
    files = sorted(d.glob('*.png')) if d.exists() else []
    if not files:
        return []
    last_epoch = max(f.stem.rsplit('epoch', 1)[-1] for f in files)
    return _panels([f for f in files if f.stem.endswith(f'epoch{last_epoch}')], n)


def _panels(files, n):
    """The diffusion trainer saves each preview as a strip: the real image, then one
    panel per seed, separated by thin bars. The image checks need single generated
    images, so a strip is cut into its seed panels (the first, real panel is dropped)."""
    out = []
    for f in files:
        im = Image.open(f)
        w, h = im.size
        k = round(w / h)
        if k < 2:
            out.append(f)
            continue
        gap = (w - k * h) // max(k - 1, 1)
        cache = f.parent.parent / 'preview_panels'
        cache.mkdir(exist_ok=True)
        for i in range(1, k):
            p = cache / f'{f.stem}_seed{i - 1}.png'
            if not p.exists():
                im.crop((i * (h + gap), 0, i * (h + gap) + h, h)).save(p)
            out.append(p)
    return out[:n]
