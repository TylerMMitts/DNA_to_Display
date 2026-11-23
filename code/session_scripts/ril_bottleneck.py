# Where does genetic information get lost between the SNPs and the generated root?
# For each stage, a linear probe (kernel ridge, lambda by 5-fold CV on trained
# lines) predicts each observed phenotype; r on the 198 validation lines.
#   snps        all 10,000 markers (= GBLUP)
#   input       what the encoder's MLP receives (124 PCs, or PCs + gated windows)
#   bottleneck  the MLP's last hidden layer (256)
#   tokens      the 8 x 256 tokens the UNet cross-attends to
# Generated-root r comes from ril_eval.py's summary.csv (4 seeds per line).
# CPU only (the GPU is generating).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, TRAITS, load_tables
import numpy as np
import pandas as pd
import torch
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES
from widiv_snp_encoding import load_widiv_genotypes
from widiv_window_encoding import DosageWindowLayout, load_widiv_encoder

meta, split = load_tables()
val = pd.read_csv(OUT / 'validation_lines.csv').genotype.tolist()
tr = sorted(g for g in meta.index if g not in set(val))
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
row_of = {g: i for i, g in enumerate(lines)}
names = tr + val
D = dosage[[row_of[g] for g in names]]
n_tr = len(tr)
rng = np.random.default_rng(0)
folds = rng.permutation(np.arange(n_tr) % 5)
# Targets: observed phenotype (obs_) and, from the answer key, the true genetic
# value (g_) of every trait that has one; test-line rows are never read.
G = pd.read_csv('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/genetic_values.csv')
G = G[G.line.map(split) == 'train'].set_index('line')
TARGETS = [f'obs_{t}' for t in TRAITS] + [c for c in G.columns if c.startswith('g_')]
Y = np.column_stack([meta.loc[names, t[4:]].to_numpy(float) if t.startswith('obs_') else G.loc[names, t].to_numpy(float)
                     for t in TARGETS])


def probe(X):
    X = np.asarray(X, np.float64)
    X = (X - X[:n_tr].mean(0)) / (X[:n_tr].std(0) + 1e-8)
    K = X @ X.T / X.shape[1]
    Ktr = K[:n_tr, :n_tr]
    out = {}
    for j, t in enumerate(TARGETS):
        y = Y[:n_tr, j]
        best = (-2, None)
        for lam in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30):
            pred = np.empty_like(y)
            for f in range(5):
                a, b = folds != f, folds == f
                al = np.linalg.solve(Ktr[np.ix_(a, a)] + lam * np.eye(a.sum()), y[a] - y[a].mean())
                pred[b] = y[a].mean() + Ktr[np.ix_(b, a)] @ al
            r = np.corrcoef(pred, y)[0, 1]
            best = max(best, (r, lam))
        al = np.linalg.solve(Ktr + best[1] * np.eye(n_tr), y - y.mean())
        p = y.mean() + K[n_tr:, :n_tr] @ al
        out[t] = float(np.corrcoef(p, Y[n_tr:, j])[0, 1])
    return out


rows = [{'model': 'all', 'stage': 'snps', **probe(D.astype(np.float32))}]
print('snps mean r g', round(np.mean([v for k, v in rows[-1].items() if k.startswith('g_')]), 3), flush=True)
for model, path in (('pca', r'E:/D2D/DNA_to_Display/models/from_hellbender/diffusion_ril_v5_medium_random_vaev3_epoch_800.pt'),
                    ('windows', r'E:/D2D/DNA_to_Display/models/from_hellbender/diffusion_ril_v5_medium_random_windows200_pc32_sp1e-06_vaev3_epoch_800.pt')):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    enc, proj = load_widiv_encoder(ck)
    reps = {'input': [], 'bottleneck': [], 'tokens': []}
    with torch.no_grad():
        for i in range(0, len(names), 64):
            d = D[i:i + 64]
            if isinstance(proj, DosageWindowLayout):
                x = torch.as_tensor(d.astype(np.int8))
                z = enc.standardize(x)
                inp = torch.cat([z @ enc.pca_components.T,
                                 (enc.window_features(z) * enc.gates()[None, :, None]).flatten(1)], 1)
                tok = enc(x)
            else:
                inp = torch.as_tensor(proj.project_lines(d, names[i:i + 64]), dtype=torch.float32)
                tok = enc(inp)
            reps['input'].append(inp.numpy())
            reps['bottleneck'].append(enc.net[:-1](inp).numpy())
            reps['tokens'].append(tok.flatten(1).numpy())
    for stage, v in reps.items():
        X = np.concatenate(v)
        rows.append({'model': model, 'stage': stage, 'dim': X.shape[1], **probe(X)})
        print(rows[-1]['model'], rows[-1]['stage'], 'mean r g', round(np.mean([v for k, v in rows[-1].items() if k.startswith('g_')]), 3), flush=True)
    del ck, enc
pd.DataFrame(rows).to_csv(OUT / 'bottleneck_probe.csv', index=False)
