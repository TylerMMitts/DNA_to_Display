# The rest of the results analysis, beyond trait accuracy. Everything here reuses the
# images and measurements the analysis stage already made, so it costs seconds:
#
#   reliability    per trait: how much of the spread in generated values is between lines
#                  rather than between seeds of one line (ICC), and what that means for
#                  the reliability of a line mean of n seeds
#   distributions  generated against real images of the same lines, per image: mean, SD,
#                  Kolmogorov-Smirnov distance, histograms
#   line matching  from a generated line's trait profile, can the right line be picked out
#                  of all evaluated lines? top-1 / top-5 against chance
#   memorisation   training history (genotype gain on trained vs held-out lines) and a
#                  nearest-neighbour copy test against the training images
#   baselines      when the genotype table carries covariates (sex, batch, site...): ridge
#                  on covariates only, markers only, and everything, so it is clear how much
#                  of the accuracy is genetic
import re

import numpy as np
import pandas as pd
from PIL import Image

from . import qc_checks as Q

COVARIATE_NAME = re.compile(r'^(sex|batch|cov|env|site|year|loc|age|plate|block|rep|trial)[_:\-]', re.I)
CHROM = re.compile(r'^(chr)?([0-9]+|x|y|z|w|mt|m|pt|un|[0-9]+[a-z]?)$', re.I)


def covariate_columns(ds, cols):
    """Covariates in the genotype table: named like SEX_/BATCH_..., or mapped to a
    'chromosome' that is not one (e.g. chr 'batch' in the SNP map)."""
    chrom = {}
    if ds.snp_map_path.exists():
        m = pd.read_csv(ds.snp_map_path)
        if {'snp', 'chr'} <= set(m.columns):
            chrom = dict(zip(m.snp.astype(str), m.chr.astype(str)))
    return [i for i, c in enumerate(cols)
            if COVARIATE_NAME.match(c) or (c in chrom and not CHROM.match(chrom[c].strip()))]


def reliability(gen, traits):
    rows = []
    for t in traits:
        if t not in gen:
            continue
        g = gen[['line', t]].dropna()
        per = g.groupby('line')[t]
        n = float(per.size().mean())
        within = float(per.var(ddof=1).mean())
        between = float(per.mean().var(ddof=1)) - within / n      # variance of true line means
        between = max(between, 0.0)
        icc = between / (between + within) if between + within > 0 else np.nan
        rows.append({'trait': t, 'lines': int(per.ngroups), 'seeds_per_line': round(n, 1),
                     'line_sd': np.sqrt(between), 'seed_sd': np.sqrt(within), 'icc_one_seed': icc,
                     'reliability_line_mean': n * icc / (1 + (n - 1) * icc) if icc == icc else np.nan,
                     'seeds_for_0.9': int(np.ceil(9 * (1 - icc) / icc)) if icc and icc == icc and icc > 0 else None})
    return pd.DataFrame(rows)


def distributions(gen, real, traits, bins=24):
    from scipy.stats import ks_2samp
    rows, hist = [], {}
    for t in traits:
        if t not in gen or t not in real:
            continue
        a, b = real[t].dropna().to_numpy(float), gen[t].dropna().to_numpy(float)
        if len(a) < 5 or len(b) < 5:
            continue
        lo, hi = np.percentile(np.r_[a, b], [1, 99])
        edges = np.linspace(lo, hi, bins + 1)
        ha, _ = np.histogram(np.clip(a, lo, hi), edges)
        hb, _ = np.histogram(np.clip(b, lo, hi), edges)
        pa, pb = ha / ha.sum(), hb / hb.sum()
        rows.append({'trait': t, 'real_images': len(a), 'generated_images': len(b),
                     'real_mean': a.mean(), 'generated_mean': b.mean(), 'real_sd': a.std(ddof=1),
                     'generated_sd': b.std(ddof=1), 'mean_shift_in_real_sd': (b.mean() - a.mean()) / a.std(ddof=1),
                     'ks_distance': float(ks_2samp(a, b).statistic), 'histogram_overlap': float(np.minimum(pa, pb).sum())})
        hist[t] = {'edges': edges.tolist(), 'real': pa.tolist(), 'generated': pb.tolist()}
    return pd.DataFrame(rows), hist


def line_matching(gen_means, truth_means):
    """Standardise each trait on the truth, then for every line rank all lines' true
    profiles by distance to its generated profile. Rank 1 = picked out correctly."""
    lines = gen_means.index.intersection(truth_means.index)
    g, t = gen_means.loc[lines], truth_means.loc[lines]
    keep = [c for c in t.columns if t[c].std() > 0 and g[c].std() > 0]
    if len(lines) < 10 or not keep:
        return {}
    zt = (t[keep] - t[keep].mean()) / t[keep].std()
    zg = (g[keep] - g[keep].mean()) / g[keep].std()           # generated on its own scale: shape, not offset
    d = np.sqrt(((zg.to_numpy()[:, None] - zt.to_numpy()[None]) ** 2).sum(-1))
    ranks = (d < np.diag(d)[:, None]).sum(1) + 1
    n = len(lines)
    return {'lines': n, 'traits_used': keep, 'top1': float((ranks == 1).mean()), 'top5': float((ranks <= 5).mean()),
            'median_rank': float(np.median(ranks)), 'chance_top1': 1 / n, 'chance_top5': min(5 / n, 1.0),
            'chance_median_rank': (n + 1) / 2, 'rank_hist': np.bincount(np.minimum(ranks, 51) - 1, minlength=51).tolist()}


def _run_for(ds, ckpt_name):
    runs = [r for r in Q.find_runs(ds) if r['kind'] == 'diffusion']
    named = [r for r in runs if r['name'] in ckpt_name]
    return (named or runs or [None])[0]


def memorisation(ds, ckpt_name, gen_paths, target_lines, test_lines, out_png, n_train=1500, n_gen=400, seed=0):
    out = {}
    run = _run_for(ds, ckpt_name)
    if run is not None:
        h = Q.load_history(run)
        cols = [c for c in ('epoch', 'genotype_gain_trained', 'genotype_gain_heldout', 'val_loss',
                            'val_loss_wrong_genotype', 'train_eval_loss') if c in h]
        if len(h) and 'genotype_gain_heldout' in h:
            gt, gh = float(h.genotype_gain_trained.iloc[-1]), float(h.genotype_gain_heldout.iloc[-1])
            out['history'] = h[cols].replace({np.nan: None}).to_dict('list')
            out['gain_trained'], out['gain_heldout'] = gt, gh
            out['gain_ratio'] = gt / gh if gh > 0 else None
        if {'train_eval_loss', 'val_loss'} <= set(h.columns) and len(h):
            out['val_over_train_loss'] = float(h.val_loss.iloc[-1] / h.train_eval_loss.iloc[-1])
    meta_p = ds.scaled_metadata_path()
    if meta_p.exists() and gen_paths:
        m = pd.read_csv(meta_p)
        m['genotype'] = m.genotype.astype(str)
        rng = np.random.default_rng(seed)
        train = m[~m.genotype.isin(set(target_lines) | set(test_lines))]
        held = m[m.genotype.isin(set(target_lines))]
        pick = lambda df, k: df.scaled_filename.to_numpy()[rng.permutation(len(df))[:k]]
        tr_files = [ds.scaled_dir() / f for f in pick(train, n_train)]
        held_files = [ds.scaled_dir() / f for f in pick(held, n_gen)]
        gen_files = [gen_paths[i] for i in rng.permutation(len(gen_paths))[:n_gen]]
        if len(tr_files) >= 20 and len(held_files) >= 10:
            T, G, R = Q._thumbs(tr_files), Q._thumbs(gen_files), Q._thumbs(held_files)
            dist = lambda A: np.sqrt(((A[:, None] - T[None]) ** 2).mean(-1))
            dg_all, dr = dist(G), dist(R).min(1)
            dg, nn = dg_all.min(1), dg_all.argmin(1)
            thr = float(np.percentile(dr, 5))
            out['copy_test'] = {
                'training_images_sampled': len(T), 'generated_images': len(G), 'heldout_real_images': len(R),
                'median_nn_generated': float(np.median(dg)), 'median_nn_heldout_real': float(np.median(dr)),
                'nn_ratio': float(np.median(dg) / np.median(dr)),
                'near_copy_threshold': thr, 'near_copy_share': float((dg < thr).mean()),
                'expected_share_if_no_copying': 0.05,
                'hist_edges': np.linspace(0, float(max(dg.max(), dr.max())), 31).tolist()}
            e = out['copy_test']['hist_edges']
            out['copy_test']['hist_generated'] = (np.histogram(dg, e)[0] / len(dg)).tolist()
            out['copy_test']['hist_heldout_real'] = (np.histogram(dr, e)[0] / len(dr)).tolist()
            # The six closest generated-to-training pairs, side by side.
            order = np.argsort(dg)[:6]
            cell = 160
            sheet = Image.new('RGB', (2 * cell + 8, len(order) * (cell + 8) - 8), (20, 20, 20))
            for k, i in enumerate(order):
                sheet.paste(Image.open(gen_files[i]).convert('RGB').resize((cell, cell)), (0, k * (cell + 8)))
                sheet.paste(Image.open(tr_files[nn[i]]).convert('RGB').resize((cell, cell)), (cell + 8, k * (cell + 8)))
            sheet.save(out_png)
            out['closest_pairs_figure'] = out_png.name
            out['closest_pairs'] = [{'generated': _name(gen_files[i]), 'training': _name(tr_files[nn[i]]),
                                     'distance': float(dg[i])} for i in order]
    verdict = []
    if out.get('gain_ratio') and out['gain_ratio'] > 1.5:
        verdict.append(f"genotype gain on training lines is {out['gain_ratio']:.1f}x the held-out gain")
    ct = out.get('copy_test')
    if ct and ct['near_copy_share'] > 0.10:
        verdict.append(f"{ct['near_copy_share']:.0%} of generated images are closer to a training image than "
                       "95% of real held-out images are")
    out['verdict'] = 'possible memorisation: ' + '; '.join(verdict) if verdict else 'no sign of memorisation'
    return out


def _name(p):
    return getattr(p, 'name', str(p))


def covariate_baselines(ds, truth, pairs, target_lines, r_generated):
    from sklearn.linear_model import RidgeCV
    from .stages.genetics import load_genotypes, standardise
    lines, cols, x = load_genotypes(ds)
    cov = covariate_columns(ds, cols)
    if not cov:
        return pd.DataFrame(), []
    markers = [i for i in range(len(cols)) if i not in set(cov)]
    z, _ = standardise(x)
    idx = {l: i for i, l in enumerate(lines)}
    split = pd.read_csv(ds.split_path)
    test = set(split.line[split.split_random == 'test'].astype(str))
    rows = []
    for measured, tc in pairs:
        if tc not in truth:
            continue
        y = truth[tc].dropna()
        fit = [l for l in y.index if l in idx and l not in test and l not in set(target_lines)]
        tg = [l for l in target_lines if l in idx and l in y.index]
        if len(fit) < 30 or len(tg) < 10:
            continue
        row = {'trait': measured, 'r_generated': r_generated.get(measured)}
        for name, sub in (('r_covariates_only', cov), ('r_markers_only', markers), ('r_all_columns', list(range(len(cols))))):
            if not sub:
                continue
            m = RidgeCV(alphas=np.logspace(-1, 6, 15)).fit(z[[idx[l] for l in fit]][:, sub], y[fit].to_numpy())
            row[name] = float(np.corrcoef(m.predict(z[[idx[l] for l in tg]][:, sub]), y[tg])[0, 1])
        rows.append(row)
    return pd.DataFrame(rows), [cols[i] for i in cov]
