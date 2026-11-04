# GWAS on WiDiv line means, real and generated, scored against the designed QTL.
#
# A mixed model with the genomic relationship matrix and 3 PCs as fixed effects, with
# variance components estimated once per trait by REML and held fixed across SNPs
# (population parameters previously determined), so the same kind of scan as the sender's
# GAPIT MLM baseline. "Found" uses the sender's rule: a Bonferroni hit (0.05 / 90,790) within
# 1 Mb of the QTL. Running it on the real line means first checks the method against
# gapit_mlm_found in qtl_and_gwas_baseline.
#
# Generated line means come from generate_all_lines.py. Each trait is centred within
# seed first: seeds share noise across lines, so a line whose image came from a
# different seed would otherwise have its mean moved by that seed's effect.
#
# Writes to results/widiv/gwas/<run>/:
#   qtl_recovery.csv    every designed QTL in every scan: min p within 1 Mb, found, effect
#   peaks.csv           every Bonferroni peak (hits within 1 Mb merged) and what it is near
#   summary.csv         per scan and trait: QTL found, peaks near no QTL, inflation
#   manhattan_<trait>.png
#
#     python code/widiv/gwas_widiv.py results/widiv/generated_all_lines/<run>/traits.csv

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from widiv_paths import (WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_METADATA_DIR, WIDIV_PLOT_PHENOTYPES, WIDIV_QTL,
                         WIDIV_RESULTS_DIR, WIDIV_SNP_MAP, WIDIV_SPLIT)
from widiv_data import load_split
from widiv_snp_encoding import load_widiv_genotypes

ALL_TRAITS = ['ccfn', 'stele_diam_um', 'mx_n', 'mx_diam_um', 'cort_cell_um', 'rca_pct', 'mcs_layers']
GENERATED_TRAITS = ['stele_diam_um', 'mx_n', 'mx_diam_um', 'rca_pct', 'mcs_layers']
LABEL = {'ccfn': 'Cortical cell files', 'stele_diam_um': 'Stele diameter', 'mx_n': 'Metaxylem number',
         'mx_diam_um': 'Metaxylem diameter', 'cort_cell_um': 'Cortical cell size', 'rca_pct': 'Aerenchyma',
         'mcs_layers': 'Sclerenchyma'}
FLANK = 1_000_000
N_PCS = 3


class Scan:
    # The parts of the mixed model shared by every trait on one set of lines.

    def __init__(self, rows, dosage):
        self.rows = rows
        self.x = dosage[rows].astype(np.float32)
        self.x -= self.x.mean(0)
        sd = self.x.std(0)
        self.polymorphic = sd > 0
        z = self.x[:, self.polymorphic] / sd[self.polymorphic]
        kinship = (z @ z.T).astype(np.float64) / z.shape[1]
        del z
        self.eigval, self.eigvec = np.linalg.eigh(kinship)
        self.eigval = np.clip(self.eigval, 0, None)
        pcs = self.eigvec[:, -N_PCS:]
        self.fixed = np.c_[np.ones(len(rows)), pcs]

    # Restricted likelihood over a grid of heritabilities, in the kinship's eigenbasis.
    def fit_variance(self, y):
        y_rot, x_rot = self.eigvec.T @ y, self.eigvec.T @ self.fixed
        best = (-np.inf, 0.0)
        n, q = len(y), self.fixed.shape[1]
        for h2 in np.linspace(0, 0.98, 50):
            v = h2 * self.eigval + (1 - h2)
            w = 1 / v
            xtx = (x_rot * w[:, None]).T @ x_rot
            beta = np.linalg.solve(xtx, (x_rot * w[:, None]).T @ y_rot)
            e = y_rot - x_rot @ beta
            s2 = float((e ** 2 * w).sum()) / (n - q)
            loglik = -0.5 * ((n - q) * np.log(s2) + np.log(v).sum() + np.linalg.slogdet(xtx)[1])
            if loglik > best[0]:
                best = (loglik, h2)
        return best[1]

    # -log10 p and the effect per alt allele for every SNP, on a trait with no missing lines.
    def run(self, y, chunk=4000):
        h2 = self.fit_variance(y)
        scale = np.sqrt(1 / (h2 * self.eigval + (1 - h2)))
        white = lambda a: (self.eigvec.T @ a) * (scale[:, None] if a.ndim == 2 else scale)
        q_, _ = np.linalg.qr(white(self.fixed))
        strip = lambda a: a - q_ @ (q_.T @ a)
        y_w = strip(white(y))
        df = len(y) - self.fixed.shape[1] - 1
        p = self.x.shape[1]
        logp, beta = np.full(p, np.nan), np.full(p, np.nan)
        for start in range(0, p, chunk):
            part = strip(white(self.x[:, start:start + chunk].astype(np.float64)))
            gg = (part ** 2).sum(0)
            ok = gg > 1e-8
            b = np.where(ok, part.T @ y_w / np.where(ok, gg, 1), np.nan)
            s2 = np.where(ok, ((y_w ** 2).sum() - b ** 2 * gg) / df, np.nan)
            t = b / np.sqrt(s2 / np.where(ok, gg, 1))
            logp[start:start + chunk] = -(stats.t.logsf(np.abs(t), df) + np.log(2)) / np.log(10)
            beta[start:start + chunk] = b
        return logp, beta, h2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('traits_csv', type=Path)
    args = parser.parse_args()
    run = args.traits_csv.parent.name
    out = WIDIV_RESULTS_DIR / 'gwas' / run
    out.mkdir(parents=True, exist_ok=True)

    lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
    names = np.array(lines)
    snp_map = pd.read_csv(WIDIV_SNP_MAP).set_index('snp').loc[snps]
    chrom, pos = snp_map['chr'].to_numpy(), snp_map['pos'].to_numpy()
    qtl = pd.read_csv(WIDIV_QTL)
    column = {s: i for i, s in enumerate(snps)}
    bonferroni = -np.log10(0.05 / len(snps))
    split = load_split(WIDIV_SPLIT).reindex(names).split_random.to_numpy()
    plot_sd = pd.read_csv(WIDIV_PLOT_PHENOTYPES)[ALL_TRAITS].std()

    real = pd.read_csv(WIDIV_METADATA_DIR / 'synthetic_phenotypes_widiv_hard__genotype_means.csv').set_index('genotype')
    gen = pd.read_csv(args.traits_csv)
    if 'valid' in gen:
        gen = gen[gen.valid].copy()
    # Within-seed centring needs enough lines per seed; replacement seeds that only a
    # few lines used are centred on the mean of all images instead.
    common = gen.seed.map(gen.seed.value_counts()) >= 30
    for t in GENERATED_TRAITS:
        gen[t] = gen[t] - np.where(common, gen.groupby('seed')[t].transform('mean'), gen[t].mean())
    gen = gen.groupby('genotype')[GENERATED_TRAITS].mean()

    sets = {'all': np.ones(len(names), bool), 'training': split == 'train', 'test': split == 'test'}
    scans = [('real', 'all', real, ALL_TRAITS), ('generated', 'all', gen, GENERATED_TRAITS),
             ('real', 'training', real, GENERATED_TRAITS), ('generated', 'training', gen, GENERATED_TRAITS),
             ('real', 'test', real, GENERATED_TRAITS), ('generated', 'test', gen, GENERATED_TRAITS)]
    recovery, peaks, summary, keep = [], [], [], {}
    for source, subset, table, traits in scans:
        cached, cached_rows = None, None
        for trait in traits:
            have = np.isin(names, table.index[table[trait].notna()])
            rows = np.flatnonzero(sets[subset] & have)
            if cached is None or not np.array_equal(rows, cached_rows):
                cached, cached_rows = Scan(rows, dosage), rows
            y = table.loc[names[rows], trait].to_numpy(np.float64)
            logp, beta, h2 = cached.run(y)
            keep[(source, subset, trait)] = logp
            chi = stats.chi2.isf(10.0 ** -logp[np.isfinite(logp)], 1)
            inflation = float(np.median(chi) / stats.chi2.ppf(0.5, 1))
            found = 0
            mine = qtl[qtl.trait == trait]
            for r in mine.itertuples():
                near = (chrom == r.chr) & (np.abs(pos - r.pos) <= FLANK)
                best = float(np.nanmax(logp[near]))
                hit = best > bonferroni
                found += hit
                c = column[r.snp]
                recovery.append({'source': source, 'set': subset, 'trait': trait, 'snp': r.snp, 'role': r.role,
                                 'pct_of_plot_variance': r.pct_of_plot_variance, 'min_p_within_1mb': 10.0 ** -best,
                                 'found': hit, 'p_at_qtl': 10.0 ** -logp[c], 'beta_per_allele': beta[c],
                                 'designed_per_allele': r.effect_per_alt_allele_sd * plot_sd[trait],
                                 'gapit_mlm_min_p': r.gapit_mlm_min_p, 'gapit_mlm_found': r.gapit_mlm_found})
            # Bonferroni hits merged into peaks, then classified by what they sit near.
            hits = np.flatnonzero(logp > bonferroni)
            order = hits[np.argsort(-logp[hits])]
            taken = []
            for h in order:
                if any(chrom[h] == chrom[k] and abs(pos[h] - pos[k]) <= FLANK for k in taken):
                    continue
                taken.append(h)
                own = ((mine.chr == chrom[h]) & ((mine.pos - pos[h]).abs() <= FLANK)).any()
                other = ((qtl.chr == chrom[h]) & ((qtl.pos - pos[h]).abs() <= FLANK)).any()
                peaks.append({'source': source, 'set': subset, 'trait': trait, 'snp': snps[h], 'chr': chrom[h],
                              'pos': pos[h], 'p': 10.0 ** -logp[h],
                              'near': 'own QTL' if own else 'other trait QTL' if other else 'no QTL'})
            near_none = sum(p['near'] == 'no QTL' for p in peaks[-len(taken):]) if taken else 0
            summary.append({'source': source, 'set': subset, 'trait': trait, 'lines': len(rows), 'h2': round(h2, 2),
                            'qtl_found': int(found), 'qtl_total': len(mine), 'peaks': len(taken),
                            'peaks_near_no_qtl': near_none, 'inflation': round(inflation, 2)})
            print(summary[-1], flush=True)
        del cached

    pd.DataFrame(recovery).to_csv(out / 'qtl_recovery.csv', index=False)
    pd.DataFrame(peaks).to_csv(out / 'peaks.csv', index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(out / 'summary.csv', index=False)

    # Manhattan plots: real above generated, for all lines and for test lines alone.
    sizes = pd.Series(pos).groupby(chrom).max()
    offset = sizes.cumsum().shift(fill_value=0)
    x = pos + offset.loc[chrom].to_numpy()
    SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
    BLUE, ORANGE = '#2a78d6', '#eb6834'
    plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 8, 'axes.edgecolor': GRID,
                         'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.facecolor': SURF, 'figure.facecolor': SURF, 'text.color': INK})
    for trait in GENERATED_TRAITS:
        panels = [('real', 'all'), ('generated', 'all'), ('real', 'test'), ('generated', 'test')]
        fig, axes = plt.subplots(4, 1, figsize=(13, 9.5), sharex=True)
        for ax, (source, subset) in zip(axes, panels):
            logp = keep[(source, subset, trait)]
            colours = np.where(chrom % 2 == 0, MUTED, '#c3c2b7')
            ax.scatter(x, logp, s=2, c=colours, linewidths=0, rasterized=True)
            for r in qtl[qtl.trait == trait].itertuples():
                ax.axvline(r.pos + offset.loc[r.chr], color=ORANGE, lw=0.9, alpha=0.8, zorder=0)
            ax.axhline(bonferroni, color=INK2, lw=0.8, ls='--')
            row = summary[(summary.source == source) & (summary.set == subset) & (summary.trait == trait)].iloc[0]
            ax.set_title(f"{source.capitalize()} line means, {subset} lines ({row.lines}): {row.qtl_found} of "
                         f"{row.qtl_total} designed QTL found, {row.peaks_near_no_qtl} peaks near no QTL, "
                         f"inflation {row.inflation}", loc='left', fontsize=8.5, color=BLUE if source == 'generated' else INK)
            ax.set_ylabel('-log10 p')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        axes[-1].set_xticks([offset.loc[c] + sizes.loc[c] / 2 for c in sizes.index])
        axes[-1].set_xticklabels(sizes.index)
        axes[-1].set_xlabel('Chromosome')
        fig.legend(handles=[Line2D([], [], color=ORANGE, label='designed QTL'),
                            Line2D([], [], color=INK2, ls='--', label='Bonferroni (0.05 / 90,790)')],
                   loc='upper right', ncol=2, frameon=False)
        fig.suptitle(f'{LABEL[trait]}: GWAS on real and generated line means', x=0.01, ha='left', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / f'manhattan_{trait}.png', dpi=150)
        plt.close(fig)
    print('wrote', out)


if __name__ == '__main__':
    main()
