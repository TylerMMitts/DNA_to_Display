# Copy check: is a generated root closer to one of its own line's real images than real
# images of the same line are to each other? Pixel RMSE at 64 px, real images in all 8
# rotations/flips (training used them), on 150 trained and all test lines.
#   d_own    generated image -> nearest real image of its own line
#   own_rank where its own line ranks (1-31) against 30 random other lines, each line scored by
#            its nearest real image; 1-in-31 (3.2%) ranked first is chance
#   d_other  median over those 30 lines of the nearest-image distance
#   d_rep    real image -> nearest other real image of the same line (replicate spread)
#     python mem_copy.py <run>_e<epoch> [...]
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, r'E:/D2D/DNA_to_Display\code'); sys.path.insert(0, r'E:/D2D/DNA_to_Display\code\widiv')
from widiv_paths import WIDIV_RESULTS_DIR, WIDIV_SCALED_DIR, WIDIV_SCALED_METADATA

S = 64
G = WIDIV_RESULTS_DIR / 'generated_all_lines'
md = pd.read_csv(WIDIV_SCALED_METADATA)


def load(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert('RGB').resize((S, S), Image.BILINEAR),
                                       dtype=np.float32) / 255).flatten()


def views(path):
    x = torch.from_numpy(np.asarray(Image.open(path).convert('RGB').resize((S, S), Image.BILINEAR),
                                    dtype=np.float32) / 255)
    out = []
    for k in range(4):
        r = torch.rot90(x, k, (0, 1))
        out += [r, torch.flip(r, (1,))]
    return torch.stack(out).flatten(1)


def rmse(a, b):      # a (n, d), b (m, d) -> (n, m)
    return torch.cdist(a, b) / np.sqrt(a.shape[1])


rows, summary = [], []
for run in sys.argv[1:]:
    traits = pd.read_csv(G / run / 'traits.csv')
    role = traits.groupby('genotype').role.first()
    rng = np.random.default_rng(0)
    pick = list(rng.choice(role.index[role == 'trained'], 150, replace=False)) + list(role.index[role == 'test'])
    counts = md.groupby('genotype').size()
    others = rng.choice(sorted(set(counts.index[counts >= 2]) - set(pick)), 30, replace=False)
    other_banks = [torch.cat([views(WIDIV_SCALED_DIR / f) for f in md[md.genotype == o].scaled_filename]) for o in others]
    for g in pick:
        real_files = md[md.genotype == g].scaled_filename.tolist()
        if len(real_files) < 2:
            continue
        rv = [views(WIDIV_SCALED_DIR / f) for f in real_files]
        own_bank = torch.cat(rv)
        rep = [float(rmse(load(WIDIV_SCALED_DIR / f)[None], torch.cat([v for j, v in enumerate(rv) if j != i])).min())
               for i, f in enumerate(real_files)]
        for f in traits[traits.genotype == g].image:
            x = load(G / run / 'images' / f)[None]
            d_own = float(rmse(x, own_bank).min())
            d_others = np.array([float(rmse(x, b).min()) for b in other_banks])
            rows.append({'run': run, 'genotype': g, 'role': role[g], 'image': f,
                         'd_own': d_own, 'd_other': float(np.median(d_others)),
                         'own_rank': int(1 + (d_others < d_own).sum()), 'd_rep_line': float(np.mean(rep))})
        for d in rep:
            rows.append({'run': run, 'genotype': g, 'role': role[g], 'image': 'real', 'd_rep': d})
    print('done', run, flush=True)

R = pd.DataFrame(rows)
gen = R[R.image != 'real'].copy()
real = R[R.image == 'real']
for run, gr in gen.groupby('run', sort=False):
    rep5 = np.percentile(real[real.run == run].d_rep, 5)
    for r, x in gr.groupby('role'):
        summary.append({'run': run, 'role': r, 'images': len(x),
                        'd_own': x.d_own.mean(), 'd_other': x.d_other.mean(),
                        'own_over_other': (x.d_own / x.d_other).mean(),
                        'own_line_ranked_first_pct': 100 * (x.own_rank == 1).mean(),
                        'mean_own_rank_of_31': x.own_rank.mean(),
                        'closer_than_replicates': (x.d_own < x.d_rep_line).mean(),
                        'near_copy_pct': 100 * (x.d_own < rep5).mean(),
                        'real_replicate_d': real[(real.run == run) & (real.role == r)].d_rep.mean()})
S_ = pd.DataFrame(summary)
pd.set_option('display.width', 220)
print(S_.round(4).to_string(index=False))
out = WIDIV_RESULTS_DIR / 'noqtl_diagnosis'
R.to_csv(out / f"copy_check_{'__'.join(sys.argv[1:])}.csv", index=False)
S_.to_csv(out / f"copy_check_summary_{'__'.join(sys.argv[1:])}.csv", index=False)
