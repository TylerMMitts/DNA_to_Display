# Genes behind each trait, and where they act in the generated image. Part of the
# analysis stage; evaluation only (nothing here is ever an input to a model).
#
#   1. Candidate genes per trait:
#        - the dataset's ground truth when it has one (genetic/known_genes.csv with
#          columns trait, marker), else
#        - a GWAS: for every genotype column, trait ~ covariates (+ 5 genotype PCs when the
#          markers are dense SNPs) + column, on the training lines only (not the
#          validation lines the maps are drawn on, not the sealed test lines), with real
#          trait values measured on the real images. Significance: Bonferroni 0.05.
#   2. The top genes (MAX_GENES, at least one per trait where significant) plus one
#      control column with no association.
#   3. For each: on MAP_LINES held-out validation lines x MAP_SEEDS, generate the line with
#      the gene in its low state and in its high state (knockout: off / on; SNP: the
#      block of markers in LD with the hit taken from a line with dosage 0 / 2), same
#      noise. The map is the mean signed change (high - low), so random pose jitter
#      cancels and what remains is the systematic effect; its strength is compared with
#      the control's. Every image is measured, so the generated trait change can be
#      compared with the GWAS effect.
# Writes analysis/gwas_<role>.csv, gwas_<role>.png, gene_maps.json, gene_maps/*.png and
# gene_maps_<role>.png; returns a summary for the stage report.
import json

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, stats

MAX_GENES = 6
MAP_LINES = 16
MAP_SEEDS = (0, 1)
LD_R = 0.8            # markers in a swapped block: |r| with the hit at least this
LD_SPAN = 200         # ... within this many columns of it on the same chromosome
GOLD, BLUE = np.array([241, 184, 45]) / 255, np.array([47, 91, 211]) / 255


def _training_lines(ds, ck):
    split = pd.read_csv(ds.split_path)
    split['line'] = split.line.astype(str)
    train = set(split.line[split.split_random == 'train'])
    return train - set(map(str, ck['val_genotypes'])) - set(map(str, ck.get('test_genotypes', [])))


def _residualise(c, *arrays):
    q, _ = np.linalg.qr(c)
    return [a - q @ (q.T @ a) for a in arrays]


def gwas(ds, ck, traits):
    """One row per trait x column: beta, se, t, p, carriers. Returns (table, info)."""
    from .analysis_extra import covariate_columns
    from .stages.genetics import load_genotypes
    lines, cols, x = load_genotypes(ds)
    cols = [str(c) for c in cols]
    real = pd.read_csv(ds.results_dir / 'traits' / 'real_line_means.csv', index_col=0)
    real.index = real.index.astype(str)
    use = sorted(_training_lines(ds, ck) & set(real.index) & set(map(str, lines)))
    idx = {str(l): i for i, l in enumerate(lines)}
    X = np.asarray(x[[idx[l] for l in use]], dtype=np.float64)
    cov = covariate_columns(ds, cols)
    markers = [j for j in range(len(cols)) if j not in set(cov)]
    C = [np.ones((len(use), 1))]
    if cov:
        cv = X[:, cov]
        cv = cv[:, cv.std(0) > 0]
        C.append(cv)
    M = X[:, markers]
    dense = float((M != 0).mean()) > 0.2
    if dense:            # population structure: top genotype PCs
        z = (M - M.mean(0)) / np.where(M.std(0) > 0, M.std(0), 1)
        u, sv, _ = np.linalg.svd(z[:, : min(z.shape[1], 20000)], full_matrices=False)
        C.append(u[:, :5] * sv[:5])
    C = np.hstack(C)
    k = np.linalg.matrix_rank(C)
    rows = []
    Mr, = _residualise(C, M)
    sxx = (Mr ** 2).sum(0)
    carriers = (M != np.median(M, axis=0)).sum(0)
    for t in traits:
        if t not in real:
            continue
        y = real.loc[use, t].to_numpy(float)
        ok = np.isfinite(y)
        if ok.sum() < 30:
            continue
        if ok.all():
            yr, = _residualise(C, y)
            Mr_t, sxx_t = Mr, sxx
        else:                       # some lines lack this trait: residualise on the rest
            yr, Mr_t = _residualise(C[ok], y[ok], M[ok])
            sxx_t = (Mr_t ** 2).sum(0)
        n = int(ok.sum())
        sxy = Mr_t.T @ yr
        with np.errstate(divide='ignore', invalid='ignore'):
            beta = sxy / sxx_t
            dof = n - k - 1
            s2 = ((yr ** 2).sum() - beta * sxy) / dof
            se = np.sqrt(s2 / sxx_t)
            tt = beta / se
        p = 2 * stats.t.sf(np.abs(tt), dof)
        sd = float(np.nanstd(y))
        for jj, j in enumerate(markers):
            if sxx_t[jj] <= 1e-9 or carriers[jj] < 3 or not np.isfinite(p[jj]):
                continue
            rows.append({'trait': t, 'marker': cols[j], 'column': j, 'beta': beta[jj], 'beta_sd': beta[jj] / sd,
                         'se': se[jj], 't': tt[jj], 'p': p[jj], 'carriers': int(carriers[jj]), 'n_lines': n})
    g = pd.DataFrame(rows)
    m = len(markers)
    chi = stats.chi2.isf(g.p.clip(1e-300), 1) if len(g) else np.array([])
    info = {'lines': len(use), 'markers_tested': m, 'covariates': [cols[j] for j in cov], 'genotype_pcs': 5 if dense else 0,
            'bonferroni_p': 0.05 / max(m, 1),
            'lambda_gc': float(np.median(chi) / stats.chi2.ppf(0.5, 1)) if len(chi) else None}
    if len(g):
        g['significant'] = g.p < info['bonferroni_p']
    return g, info, (lines, cols, x)


def known_genes(ds):
    p = ds.genetic_dir / 'known_genes.csv'
    if not p.exists():
        return None
    k = pd.read_csv(p)
    return k if {'trait', 'marker'} <= set(k.columns) else None


def choose(g, traits, known=None):
    """The genes to map: ground truth, or the strongest GWAS hit per trait, then the next
    strongest overall, up to MAX_GENES; plus a control with no association."""
    picks = []
    if known is not None:
        for _, r in known.iterrows():
            hit = g[(g.trait == r.trait) & (g.marker.astype(str) == str(r.marker))]
            picks.append(hit.iloc[0].to_dict() if len(hit) else {'trait': r.trait, 'marker': str(r.marker)})
        source = 'ground truth'
    else:
        sig = g[g.significant].sort_values('p')
        for t in traits:
            top = sig[sig.trait == t]
            if len(top) and top.iloc[0].marker not in {p['marker'] for p in picks}:
                picks.append(top.iloc[0].to_dict())
        for _, r in sig.iterrows():
            if len(picks) >= MAX_GENES:
                break
            if r.marker not in {p['marker'] for p in picks}:
                picks.append(r.to_dict())
        source = 'gwas'
        if not picks:      # nothing significant: map the strongest, say so
            for _, r in g.sort_values('p').drop_duplicates('marker').head(2).iterrows():
                picks.append({**r.to_dict(), 'suggestive_only': True})
    picks = picks[:MAX_GENES]
    # Control: the column with the weakest association over all traits.
    if len(g):
        worst = g.groupby('marker').p.min().sort_values(ascending=False)
        ctl = g[g.marker == worst.index[0]].sort_values('p').iloc[0].to_dict()
        ctl['control'] = True
        picks.append(ctl)
    return picks, source


def _block(cols, x, j, snp_map):
    """Columns swapped with column j: itself, plus those in LD with it on its chromosome."""
    if snp_map is None or cols[j] not in snp_map.index:
        return [j]
    chrom = snp_map.chr.get(cols[j])
    lo, hi = max(0, j - LD_SPAN), min(len(cols), j + LD_SPAN + 1)
    near = [i for i in range(lo, hi) if snp_map.chr.get(cols[i]) == chrom]
    v = x[:, j].astype(float)
    if v.std() == 0:
        return [j]
    cc = [i for i in near if x[:, i].std() > 0 and abs(np.corrcoef(v, x[:, i].astype(float))[0, 1]) >= LD_R]
    return sorted(set(cc) | {j})


def _states(x, j, block):
    """Low and high values of the block: the most common pattern among lines at the
    column's lowest / highest value (a knockout column: 0 and its carrier value)."""
    v = x[:, j]
    out = []
    for val in (v.min(), v.max()):
        pats = x[v == val][:, block]
        uniq, counts = np.unique(pats, axis=0, return_counts=True)
        out.append(uniq[counts.argmax()])
    return out


def maps(ds, ckpt, ck, picks, data, recipe, out, real_sd):
    from .generate import generate
    from .measure import measure_paths
    lines, cols, x = data
    idx = {str(l): i for i, l in enumerate(lines)}
    val = sorted(str(l) for l in ck['val_genotypes'] if str(l) in idx)
    snp_map = None
    if ds.snp_map_path.exists():
        sm = pd.read_csv(ds.snp_map_path)
        if {'snp', 'chr'} <= set(sm.columns):
            snp_map = sm.assign(snp=sm.snp.astype(str)).set_index('snp')
    img_dir = out / 'gene_maps' / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    genes, geno = [], {}
    for k, p in enumerate(picks):
        j = int(p.get('column', cols.index(p['marker']) if p['marker'] in cols else -1))
        if j < 0:
            continue
        block = _block(cols, x, j, snp_map)
        low, high = _states(x, j, block)
        # Held-out lines; for a knockout, lines that do not already carry it.
        targets = [l for l in val if x[idx[l], j] == low[block.index(j)]][:MAP_LINES]
        for l in targets:
            for tag, state in (('lo', low), ('hi', high)):
                row = np.array(x[idx[l]], dtype=float)
                row[block] = state
                geno[f'g{k}_{l}_{tag}'] = row
        step = float(high[block.index(j)] - low[block.index(j)])
        gwas_change = p['beta'] * step if p.get('beta') is not None else None
        genes.append({**p, 'k': k, 'column': j, 'block_size': len(block), 'targets': targets, 'low_to_high': step,
                      'gwas_change': gwas_change,
                      'gwas_change_sd_units': gwas_change / real_sd[p['trait']] if gwas_change is not None
                      and real_sd.get(p.get('trait')) else None})
    # One model load for every gene's images.
    generate(ds, ckpt, None, len(MAP_SEEDS), img_dir, genotypes=geno, seed_start=MAP_SEEDS[0])
    # Measure every image once.
    files = [f for f in sorted(img_dir.glob('*_seed*.png'))
             if np.asarray(Image.open(f).convert('L')).std() >= 3]       # blank (diverged) samples left out
    meas = measure_paths(files, recipe)
    meas['label'] = meas.image.str.rsplit('_seed', n=1).str[0]
    meas['seed'] = meas.image.str.rsplit('_seed', n=1).str[1].str.replace('.png', '', regex=False).astype(int)
    meas = meas.set_index(['label', 'seed'])
    mdir = out / 'gene_maps'
    rendered = []
    for gdict in genes:
        k = gdict['k']
        diffs, bases, dtr, skipped = [], [], [], 0
        trait = gdict.get('trait')
        for l in gdict['targets']:
            for s in MAP_SEEDS:
                a = np.asarray(Image.open(img_dir / f'g{k}_{l}_lo_seed{s}.png').convert('L'), float) / 255
                b = np.asarray(Image.open(img_dir / f'g{k}_{l}_hi_seed{s}.png').convert('L'), float) / 255
                if a.std() < 0.01 or b.std() < 0.01:     # a sample that diverged (blank image): left out
                    skipped += 1
                    continue
                diffs.append(b - a)
                bases.append(a)
                if trait in meas.columns and (f'g{k}_{l}_hi', s) in meas.index and (f'g{k}_{l}_lo', s) in meas.index:
                    dtr.append(meas.loc[(f'g{k}_{l}_hi', s), trait] - meas.loc[(f'g{k}_{l}_lo', s), trait])
        d = ndimage.gaussian_filter(np.mean(diffs, 0), 3)
        gdict['map_strength'] = float(np.abs(d).mean())
        gdict['blank_pairs_skipped'] = skipped
        rendered.append((gdict, np.mean(bases, 0), d))
        gdict['image_example'] = f'images/g{k}_{gdict["targets"][0]}_hi_seed{MAP_SEEDS[0]}.png' if gdict['targets'] else None
        if dtr:
            dtr = np.asarray(dtr, float)
            sd = real_sd.get(trait) or 1.0
            gdict.update({'generated_change': float(dtr.mean()), 'generated_change_sd_units': float(dtr.mean() / sd),
                          'generated_change_p': float(stats.ttest_1samp(dtr, 0).pvalue) if len(dtr) > 2 else None,
                          'pairs': int(len(dtr))})
            # Reproduced only when the generated change is clear (p < 0.05) and points the same way.
            pc = gdict['generated_change_p']
            clear = pc is not None and pc < 0.05
            if gdict.get('beta') is not None and np.isfinite(gdict.get('beta', np.nan)):
                same = bool(np.sign(dtr.mean()) == np.sign(gdict['beta']))
                gdict['sign_agrees'] = same and clear
                gdict['reproduced'] = ('agrees' if same else 'opposite') if clear else 'no clear change'
        gdict.pop('targets')
    # One colour scale for every map (the strongest gene sets it), so a weak map looks weak
    # and the control shows the noise level.
    vmax = max([float(np.percentile(np.abs(d), 99.5)) for _, _, d in rendered] + [1e-6])
    for gdict, base, d in rendered:
        gdict['map'] = _render(base, d, mdir / f"gene_{gdict['k']}.png", vmax)
    ctl = [g for g in genes if g.get('control')]
    if ctl and ctl[0]['map_strength'] > 0:
        for g in genes:
            g['strength_vs_control'] = g['map_strength'] / ctl[0]['map_strength']
    return genes


def _render(base, d, path, vmax=None):
    """Mean generated image in grey, the signed change painted over it: gold where the
    image got lighter with the gene on, blue where it got darker."""
    vmax = vmax or max(float(np.percentile(np.abs(d), 99.5)), 1e-6)
    h = np.clip(d / vmax, -1, 1)
    rgb = np.repeat(base[..., None], 3, -1) * 0.9 + 0.1
    a = np.abs(h)[..., None] ** 0.8
    col = np.where(h[..., None] > 0, GOLD, BLUE)
    Image.fromarray(np.uint8(np.clip(rgb * (1 - a) + col * a, 0, 1) * 255)).save(path)
    return f'gene_maps/{path.name}'


def _figures(g, info, genes, labels, out, role):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    traits = list(dict.fromkeys(g.trait)) if len(g) else []
    if traits:
        fig, axes = plt.subplots(len(traits), 1, figsize=(10, 1.5 * len(traits)), dpi=110, sharex=True, squeeze=False)
        thr = -np.log10(info['bonferroni_p'])
        for ax, t in zip(axes[:, 0], traits):
            s = g[g.trait == t]
            lp = -np.log10(s.p.clip(1e-300))
            ax.scatter(s.column, lp, s=4, c=np.where(lp > thr, '#c98a00', '#8a8a8a'))
            ax.axhline(thr, ls='--', lw=0.7, c='#444')
            ax.set_ylabel(labels.get(t, t), fontsize=7, rotation=0, ha='right', va='center')
            ax.tick_params(labelsize=6)
        axes[-1, 0].set_xlabel('genotype column', fontsize=7)
        fig.suptitle(f"GWAS on {info['lines']} training lines (dashed: Bonferroni)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f'gwas_{role}.png')
        plt.close(fig)
    if genes:
        n = len(genes)
        fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.9), dpi=120, squeeze=False)
        for ax, gd in zip(axes[0], genes):
            ax.imshow(Image.open(out / gd['map']))
            ax.set_xticks([]); ax.set_yticks([])
            title = 'control (no association)' if gd.get('control') else f"{gd['marker']}\n{labels.get(gd.get('trait'), gd.get('trait'))}"
            sub = f"p {gd['p']:.1e}" if gd.get('p') is not None and not gd.get('control') else ''
            if gd.get('strength_vs_control') and not gd.get('control'):
                sub += f" · {gd['strength_vs_control']:.1f}x control"
            ax.set_title(f'{title}\n{sub}', fontsize=7)
        fig.suptitle('Where switching the gene on changes the generated image (gold lighter, blue darker)', fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f'gene_maps_{role}.png')
        plt.close(fig)


def run(ds, ckpt, role='validation'):
    import torch
    out = ds.results_dir / 'analysis'
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg = ds.config().get('traits') or {}
    traits = [t['measured_name'] for t in cfg.get('traits', [])]
    labels = {t['measured_name']: t.get('label') or t['measured_name'] for t in cfg.get('traits', [])}
    g, info, data = gwas(ds, ck, traits)
    g.to_csv(out / f'gwas_{role}.csv', index=False)
    known = known_genes(ds)
    picks, source = choose(g, traits, known)
    real = pd.read_csv(ds.results_dir / 'traits' / 'real_line_means.csv', index_col=0)
    real_sd = {t: float(real[t].std()) for t in traits if t in real}
    for p in picks:                 # the GWAS effect in SD units of the real trait, for comparison
        if p.get('beta') is not None and p.get('trait') in real_sd:
            p['beta_sd'] = p['beta'] / real_sd[p['trait']] if real_sd[p['trait']] else None
    genes = maps(ds, ckpt, ck, picks, data, cfg.get('recipe'), out, real_sd)
    _figures(g, info, genes, labels, out, role)
    top = (g.sort_values('p').groupby('trait').head(5)[['trait', 'marker', 'beta', 'beta_sd', 'p', 'carriers', 'significant']]
           if len(g) else pd.DataFrame())
    result = {'source': source, **info,
              'significant_hits': {t: int(g[(g.trait == t) & g.significant].shape[0]) for t in traits} if len(g) else {},
              'top_hits': top.to_dict('records'), 'genes': genes, 'labels': labels,
              'note': 'GWAS on real trait values of training lines only; maps are the mean signed change when the gene '
                      'is switched on in held-out lines, same noise. What the model is sensitive to, not proven biology.'}
    from .project import _clean
    (out / 'gene_maps.json').write_text(json.dumps(_clean(result), indent=1, default=float))
    mapped = [x for x in genes if not x.get('control')]
    agree = [x['sign_agrees'] for x in mapped if 'sign_agrees' in x]     # clear change, same direction
    return {'gene_maps_source': source, 'genes_mapped': len(mapped),
            'gwas_significant_total': int(g.significant.sum()) if len(g) else 0,
            'gene_maps_sign_agreement': f'{sum(agree)} of {len(agree)}' if agree else None,
            'gwas_lambda_gc': info.get('lambda_gc')}
