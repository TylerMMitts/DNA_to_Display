# Making contribution maps trait-specific: four map types from the same gene swaps, each
# scored against the designed QTL (answer key, scoring only). Best RIL model (windows e1200),
# validation lines only, inference only.
#   abs        mean |edited - original| colour change (the current maps)
#   signed     mean (edited - original) colour change, each pair oriented so the swap always
#              adds the allele coded 2 at the region's centre marker (random jitter cancels)
#   tissue     signed change of each tissue label (gained +, lost -), same orientation
#   tissue_norm  as tissue, but each root aligned on its OWN centre and radius (size removed)
# Per compartment strength: abs/signed -> mean |map| inside that compartment's region of the
# average root; tissue maps -> mean |signed map of that tissue| over the root.
# z = (strength - control mean) / control sd (controls scored leave-one-out). The map points to
# the compartment with the highest z; it is called when that z > 3.
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / 'ril_poster_heroes.py').read().split('# 1. Mosaic.')[0])
from PIL import Image
from widiv_paths import WIDIV_RESULTS_DIR

GM = WIDIV_RESULTS_DIR / 'gene_maps'
GM.mkdir(exist_ok=True)
N_LINES, SEEDS2, CANON = 48, (30, 2), 256
COMP = ['vessels', 'stele', 'aerenchyma', 'outer', 'cortex']
AK = 'E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/'
q = pd.read_csv(AK + 'designed_qtl_markers.csv')
qt = pd.read_csv(AK + 'qtl_and_scan_baseline__qtl.csv')
pct = {r.qtl: r.pct_genetic_variance for _, r in qt.iterrows()}
pct['lac_q'] = 70.0
EXPECT = {'mx_n': {'vessels', 'stele'}, 'stele_diam_um': {'vessels', 'stele'}, 'mx_diam_um': {'vessels'},
          'cortex_width_um': {'cortex', 'outer'}, 'rca_pct': {'aerenchyma'}, 'n_lacunae': {'aerenchyma'}}
snp_list = list(snps)
ch, cMv = smap.chr.to_numpy(), smap.pos_cM.to_numpy()
regions = []
for _, r in q.iterrows():
    regions.append({'key': r.qtl, 'marker': r.snp, 'chr': int(r.chr), 'pos': float(smap.loc[r.snp, 'pos_cM']),
                    'width': 2.0 if r.qtl.startswith('rca_linked') else 10.0, 'trait': r.trait,
                    'expected': sorted(set().union(*[EXPECT[t] for t in r.trait.split(';')])), 'pct_var': pct.get(r.qtl, np.nan)})
for i, (c, p) in enumerate([(2, 20.0), (3, 120.0), (4, 40.0), (5, 20.0), (7, 40.0), (10, 20.0)]):
    centre = smap.index[(ch == c)][np.argmin(np.abs(cMv[ch == c] - p))]
    regions.append({'key': f'control{i + 1}', 'marker': centre, 'chr': c, 'pos': p, 'width': 10.0, 'trait': 'none',
                    'expected': [], 'pct_var': 0.0})
STELE_L = [WT.IDX[l] for l in (5, 6, 7, 8, 9)]
OUTER_L = [WT.IDX[l] for l in (1, 2, 3)]


def align(im, cy, cx, R):
    c = crop(im, cy, cx, 1.3 * R)
    return np.asarray(Image.fromarray(c).resize((CANON, CANON), Image.NEAREST))


def onehot(rgb):
    cls = WT.classify(rgb)
    white = cls == WT.WHITE
    root = ndimage.binary_fill_holes(ndimage.binary_closing(~white, iterations=6))
    stele = ndimage.binary_opening(ndimage.binary_fill_holes(np.isin(cls, STELE_L)), iterations=1)
    vessels = white & ndimage.binary_erosion(stele, iterations=1)
    outer = (np.isin(cls, OUTER_L) | (root & ~ndimage.binary_erosion(root, iterations=3))) & ~stele
    aer = white & root & ~stele & ~outer
    cortex = root & ~stele & ~outer & ~aer
    return np.stack([vessels, stele & ~vessels, aer, outer, cortex]).astype(np.float32)   # COMP order


lineset = val[:80]
base_cache = {}
store = {}
tmpl_acc, tmpl_n = np.zeros((5, CANON, CANON), np.float32), 0
for reg in regions:
    blk = np.flatnonzero((ch == reg['chr']) & (np.abs(cMv - reg['pos']) <= reg['width']))
    col = snp_list.index(reg['marker'])
    lines_q = [g for g in lineset if dosage[row_of[g], col] != 1][:N_LINES]
    D0 = dosage[[row_of[g] for g in lines_q]]
    D1 = D0.copy()
    for k in range(len(lines_q)):
        hom = blk[D1[k, blk] != 1]
        D1[k, hom] = 2 - D1[k, hom]
    sign = np.where(D0[:, col] == 0, 1.0, -1.0)          # +1: the swap adds allele 2 at the centre marker
    acc = {m: np.zeros((CANON, CANON), np.float32) for m in ('abs', 'signed')}
    tis = {m: np.zeros((5, CANON, CANON), np.float32) for m in ('tissue', 'tissue_norm')}
    n = 0
    for sd in SEEDS2:
        need = [g for g in lines_q if (g, sd) not in base_cache]
        if need:
            for g, im in zip(need, generate(dosage[[row_of[g] for g in need]], need, [sd] * len(need))):
                base_cache[(g, sd)] = im
        for k, (g, e) in enumerate(zip(lines_q, generate(D1, lines_q, [sd] * len(lines_q)))):
            b = base_cache[(g, sd)]
            cy, cx, R = root_box(b)
            ab, ae = align(b, cy, cx, R), align(e, cy, cx, R)
            ey, ex, eR = root_box(e)
            ae_own = align(e, ey, ex, eR)
            d = (ae.astype(np.float32) - ab.astype(np.float32)).mean(-1) / 255
            acc['abs'] += np.abs(d)
            acc['signed'] += sign[k] * d
            hb = onehot(ab)
            tis['tissue'] += sign[k] * (onehot(ae) - hb)
            tis['tissue_norm'] += sign[k] * (onehot(ae_own) - hb)
            if reg['key'].startswith('control'):
                tmpl_acc += hb; tmpl_n += 1
            n += 1
    store[reg['key']] = {**{m: v / n for m, v in acc.items()}, **{m: v / n for m, v in tis.items()}}
    print(reg['key'], 'pairs', n, flush=True)

# Compartment regions of the average root (for abs/signed maps).
tmpl = tmpl_acc / tmpl_n
root_mask = tmpl.sum(0) > 0.5
region = np.where(root_mask, tmpl.argmax(0), -1)
np.savez_compressed(GM / 'gene_maps_methods.npz', template=tmpl,
                    **{f'{k}__{m}': v for k, d in store.items() for m, v in d.items()})


def strengths(d, method):
    if method in ('abs', 'signed'):
        return [float(np.abs(d[method])[region == j].mean()) for j in range(5)]
    return [float(np.abs(d[method][j])[root_mask].mean()) for j in range(5)]


rows = []
for method in ('abs', 'signed', 'tissue', 'tissue_norm'):
    S = {r['key']: strengths(store[r['key']], method) for r in regions}
    ctrl = [r['key'] for r in regions if r['key'].startswith('control')]
    for r in regions:
        pool = [c for c in ctrl if c != r['key']]          # leave-one-out for controls
        cm = np.array([S[c] for c in pool])
        z = (np.array(S[r['key']]) - cm.mean(0)) / cm.std(0, ddof=1)
        top = COMP[int(np.argmax(z))]
        rows.append({'method': method, 'key': r['key'], 'trait': r['trait'], 'pct_var': r['pct_var'],
                     'expected': ';'.join(r['expected']), 'points_to': top, 'top_z': float(z.max()),
                     'called': bool(z.max() > 3), 'correct': (top in r['expected']) if r['expected'] else np.nan,
                     **{f'z_{c}': float(v) for c, v in zip(COMP, z)}})
R_ = pd.DataFrame(rows)
R_.to_csv(GM / 'gene_map_methods.csv', index=False)
summ = []
for method, d in R_.groupby('method', sort=False):
    g = d[~d.key.str.startswith('control')]
    c = d[d.key.str.startswith('control')]
    summ.append({'method': method, 'correct_all': f"{int(g.correct.sum())}/{len(g)}",
                 'called': int(g.called.sum()), 'correct_called': int(g[g.called].correct.sum()),
                 'cortex_genes_correct': f"{int(g[g.trait == 'cortex_width_um'].correct.sum())}/8",
                 'structure_genes_correct': f"{int(g[g.trait != 'cortex_width_um'].correct.sum())}/6",
                 'controls_called (false +)': int(c.called.sum())})
S_ = pd.DataFrame(summ)
S_.to_csv(GM / 'gene_map_methods_summary.csv', index=False)
pd.set_option('display.width', 250)
print(S_.to_string(index=False))
print(R_[R_.method.isin(['tissue', 'tissue_norm'])][['method', 'key', 'pct_var', 'expected', 'points_to', 'top_z', 'called', 'correct'] +
      [f'z_{c}' for c in COMP]].round(2).to_string(index=False))
print('done')
