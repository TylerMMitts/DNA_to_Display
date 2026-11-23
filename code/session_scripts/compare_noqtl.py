# Did removing the QTL from the genotype input remove their effect from the images?
#
# For the full, literal (40 QTL SNPs removed) and strict (+-1 Mb removed, input made
# orthogonal to the QTL) models, on generated line means (seed-centred, as in gwas_widiv):
#  1. Joint regression, trait (plot-SD units) ~ the trait's 6 QTL dosages + true background
#     genetic value. Each QTL coefficient / its designed effect = share of that QTL's effect
#     the images reproduce; the background coefficient / 1 = the same for everything else.
#     Pooled QTL share = slope of fitted on designed effects through the origin, over the 30.
#     95% CIs from 500 bootstrap resamples of lines.
#  2. Hat decomposition (r with QTL part, background part), as before.
# Real line means are run through the same regression as the reference (expected ~1).
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'E:/D2D/DNA_to_Display\code'); sys.path.insert(0, r'E:/D2D/DNA_to_Display\code\widiv')
from widiv_data import load_split
from widiv_paths import (WIDIV_GENETIC_VALUES, WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_METADATA_DIR,
                         WIDIV_PLOT_PHENOTYPES, WIDIV_QTL, WIDIV_RESULTS_DIR, WIDIV_SPLIT)
from widiv_snp_encoding import load_widiv_genotypes

import os
EP = os.environ.get('NOQTL_EPOCH', '400')      # epoch 400 results stay at the folder root
OUT = WIDIV_RESULTS_DIR / 'gwas' / 'noqtl_comparison' / ('' if EP == '400' else f'e{EP}')
OUT.mkdir(parents=True, exist_ok=True)
T = ['stele_diam_um', 'mx_n', 'mx_diam_um', 'rca_pct', 'mcs_layers']
MODELS = {'full': f'random_vaev3_e{EP}', 'literal': f'random_noqtl0kb_vaev3_e{EP}',
          'strict': f'random_noqtl1000kb_orth_vaev3_e{EP}'}
B = 500

lines, snps, d = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
names = np.array(lines)
col = {s: i for i, s in enumerate(snps)}
qtl = pd.read_csv(WIDIV_QTL)
split = load_split(WIDIV_SPLIT).reindex(names).split_random.to_numpy()
gv = pd.read_csv(WIDIV_GENETIC_VALUES).set_index('genotype').loc[lines]
plot_sd = pd.read_csv(WIDIV_PLOT_PHENOTYPES)[T].std()


def line_means(path):
    g = pd.read_csv(path)
    if 'valid' in g:
        g = g[g.valid].copy()
    common = g.seed.map(g.seed.value_counts()) >= 30
    for t in T:
        g[t] = g[t] - np.where(common, g.groupby('seed')[t].transform('mean'), g[t].mean())
    return g.groupby('genotype')[T].mean()


tables = {'real': pd.read_csv(WIDIV_METADATA_DIR / 'synthetic_phenotypes_widiv_hard__genotype_means.csv')
          .set_index('genotype')[T]}
for label, run in MODELS.items():
    tables[label] = line_means(WIDIV_RESULTS_DIR / 'generated_all_lines' / run / 'traits.csv')

design = {}
for t in T:
    mine = qtl[qtl.trait == t]
    dos = np.stack([d[:, col[s]].astype(float) for s in mine.snp], 1)
    eff = mine.effect_per_alt_allele_sd.to_numpy()
    part = dos @ eff
    design[t] = (mine, dos, eff, part, gv[f'g_{t}'].to_numpy() - part)

sets = {'training': split == 'train', 'test': split == 'test'}
rng = np.random.default_rng(0)


def fit(idx, y, t):
    _, dos, eff, _, back = design[t]
    x = np.c_[np.ones(len(idx)), dos[idx], back[idx]]
    c = np.linalg.lstsq(x, y, rcond=None)[0]
    return c[1:-1], c[-1]


per_qtl, pooled, hat = [], [], []
for source, table in tables.items():
    for subset, mask in sets.items():
        fits = {}
        for t in T:
            idx = np.flatnonzero(mask & np.isin(names, table.index[table[t].notna()]))
            y = table.loc[names[idx], t].to_numpy(float) / plot_sd[t]
            mine, dos, eff, part, back = design[t]
            q, b = fit(idx, y, t)
            boot = [fit(idx[s], y[s], t) for s in (rng.integers(0, len(idx), len(idx)) for _ in range(B))]
            fits[t] = (q, b, boot)
            for k, r in enumerate(mine.itertuples()):
                per_qtl.append({'source': source, 'set': subset, 'trait': t, 'snp': r.snp, 'role': r.role,
                                'designed': r.effect_per_alt_allele_sd, 'fitted': q[k], 'share': q[k] / r.effect_per_alt_allele_sd,
                                'right_sign': bool(np.sign(q[k]) == np.sign(r.effect_per_alt_allele_sd))})
            hat.append({'source': source, 'set': subset, 'trait': t,
                        'r_qtl_part': np.corrcoef(y, part[idx])[0, 1], 'r_background': np.corrcoef(y, back[idx])[0, 1],
                        'background_share': b, 'background_share_lo': np.percentile([x[1] for x in boot], 2.5),
                        'background_share_hi': np.percentile([x[1] for x in boot], 97.5)})
        # Pooled over the 30 QTL: slope of fitted on designed, through the origin.
        eff_all = np.concatenate([design[t][2] for t in T])
        slope = lambda qs: float(eff_all @ qs / (eff_all @ eff_all))
        q_all = np.concatenate([fits[t][0] for t in T])
        boot_slopes = [slope(np.concatenate([fits[t][2][i][0] for t in T])) for i in range(B)]
        b_all = np.mean([fits[t][1] for t in T])
        boot_b = [np.mean([fits[t][2][i][1] for t in T]) for i in range(B)]
        pooled.append({'source': source, 'set': subset, 'qtl_share': slope(q_all),
                       'qtl_lo': np.percentile(boot_slopes, 2.5), 'qtl_hi': np.percentile(boot_slopes, 97.5),
                       'background_share': b_all, 'bg_lo': np.percentile(boot_b, 2.5), 'bg_hi': np.percentile(boot_b, 97.5),
                       'right_sign_of_30': int(np.sum(np.sign(q_all) == np.sign(eff_all)))})

per_qtl, pooled, hat = pd.DataFrame(per_qtl), pd.DataFrame(pooled), pd.DataFrame(hat)
per_qtl.to_csv(OUT / 'per_qtl.csv', index=False)
pooled.to_csv(OUT / 'pooled.csv', index=False)
hat.to_csv(OUT / 'hat.csv', index=False)
pd.set_option('display.width', 220)
print(pooled.round(3).to_string(index=False))
print(hat.pivot_table(index=['set', 'trait'], columns='source', values=['r_qtl_part', 'r_background']).round(3).to_string())
