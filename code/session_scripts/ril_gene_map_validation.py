# Are genotype-to-phenotype contribution maps reliable? Ground truth = the 14 designed QTL.
# Best RIL model (windows e1200), validation lines only, inference only.
# For every designed QTL and 6 control regions (no designed QTL within 20 cM):
#   swap the region (±10 cM; ±2 cM for the linked aerenchyma pair) to the other parent on up to
#   48 unseen lines x 2 seeds (same noise before/after), align each pair on the unedited root,
#   and record |edited - original| per pixel, split by tissue compartment. Compartments come
#   from the tissue-colour classification (the same one the trait measurements use), taking a
#   pixel's compartment from either image by priority vessels > aerenchyma > stele > outer > cortex.
# Score per compartment: z = (mean change - control mean) / control sd across the 6 controls.
# A map "points to" its highest-z compartment; correct if that compartment holds the QTL's
# trait (answer key, used only for scoring). Writes results/ril_v5/gene_maps/.
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
q = pd.read_csv('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/designed_qtl_markers.csv')
qt = pd.read_csv('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/qtl_and_scan_baseline__qtl.csv')
pct = {r.qtl: r.pct_genetic_variance for _, r in qt.iterrows()}
pct['lac_q'] = 70.0
EXPECT = {'mx_n': {'vessels', 'stele'}, 'stele_diam_um': {'vessels', 'stele'}, 'mx_diam_um': {'vessels'},
          'cortex_width_um': {'cortex', 'outer'}, 'rca_pct': {'aerenchyma'}, 'n_lacunae': {'aerenchyma'}}
regions = []
for _, r in q.iterrows():
    traits = r.trait.split(';')
    exp = set().union(*[EXPECT[t] for t in traits])
    regions.append({'key': r.qtl, 'marker': r.snp, 'chr': int(r.chr), 'pos': float(smap.loc[r.snp, 'pos_cM']),
                    'width': 2.0 if r.qtl.startswith('rca_linked') else 10.0, 'trait': r.trait, 'expected': sorted(exp),
                    'pct_var': pct.get(r.qtl, np.nan)})
for i, (c, p) in enumerate([(2, 20.0), (3, 120.0), (4, 40.0), (5, 20.0), (7, 40.0), (10, 20.0)]):
    regions.append({'key': f'control{i + 1}', 'marker': None, 'chr': c, 'pos': p, 'width': 10.0, 'trait': 'none',
                    'expected': [], 'pct_var': 0.0})
snp_list = list(snps)
ch, cMv = smap.chr.to_numpy(), smap.pos_cM.to_numpy()
WHITEI = WT.WHITE
STELE_L = [WT.IDX[l] for l in (5, 6, 7, 8, 9)]
OUTER_L = [WT.IDX[l] for l in (1, 2, 3)]


def align(im, cy, cx, R):
    c = crop(im, cy, cx, 1.3 * R)
    return np.asarray(Image.fromarray(c).resize((CANON, CANON), Image.NEAREST))


def compartments(rgb):
    cls = WT.classify(rgb)
    white = cls == WHITEI
    root = ndimage.binary_fill_holes(ndimage.binary_closing(~white, iterations=6))
    stele = ndimage.binary_fill_holes(np.isin(cls, STELE_L))
    stele = ndimage.binary_opening(stele, iterations=1)
    vessels = white & ndimage.binary_erosion(stele, iterations=1)
    outer = (np.isin(cls, OUTER_L) | (root & ~ndimage.binary_erosion(root, iterations=3))) & ~stele
    aer = white & root & ~stele & ~outer
    cortex = root & ~stele & ~outer & ~aer
    return {'vessels': vessels, 'stele': stele & ~vessels, 'aerenchyma': aer, 'outer': outer, 'cortex': cortex}


lineset = val[:80]
base_cache, rows, maps = {}, [], {}
for reg in regions:
    blk = np.flatnonzero((ch == reg['chr']) & (np.abs(cMv - reg['pos']) <= reg['width']))
    col = snp_list.index(reg['marker']) if reg['marker'] else None
    lines_q = [g for g in lineset if col is None or dosage[row_of[g], col] != 1][:N_LINES]
    D0 = dosage[[row_of[g] for g in lines_q]]
    D1 = D0.copy()
    for k in range(len(lines_q)):
        hom = blk[D1[k, blk] != 1]
        D1[k, hom] = 2 - D1[k, hom]
    S, A = {c: 0.0 for c in COMP}, {c: 0 for c in COMP}
    acc, n = np.zeros((CANON, CANON), np.float32), 0
    for sd in SEEDS2:
        need = [g for g in lines_q if (g, sd) not in base_cache]
        if need:
            for g, im in zip(need, generate(dosage[[row_of[g] for g in need]], need, [sd] * len(need))):
                base_cache[(g, sd)] = im
        for g, e in zip(lines_q, generate(D1, lines_q, [sd] * len(lines_q))):
            b = base_cache[(g, sd)]
            cy, cx, R = root_box(b)
            ab, ae = align(b, cy, cx, R), align(e, cy, cx, R)
            d = np.abs(ae.astype(np.float32) - ab.astype(np.float32)).mean(-1) / 255
            cb, ce = compartments(ab), compartments(ae)
            taken = np.zeros_like(d, bool)
            for c in COMP:                       # priority order
                m = (cb[c] | ce[c]) & ~taken
                S[c] += float(d[m].sum()); A[c] += int(m.sum())
                taken |= m
            acc += d
            n += 1
    maps[reg['key']] = acc / n
    row = {k: reg[k] for k in ('key', 'marker', 'chr', 'pos', 'width', 'trait', 'pct_var')}
    row.update(expected=';'.join(reg['expected']), n_pairs=n, n_markers=int(len(blk)),
               **{f'mean_{c}': S[c] / max(A[c], 1) for c in COMP}, mean_root=sum(S.values()) / max(sum(A.values()), 1))
    rows.append(row)
    print(reg['key'], {c: round(row[f'mean_{c}'], 4) for c in COMP}, flush=True)

R_ = pd.DataFrame(rows)
ctrl = R_[R_.key.str.startswith('control')]
for c in COMP + ['root']:
    mu, sdv = ctrl[f'mean_{c}'].mean(), ctrl[f'mean_{c}'].std(ddof=1)
    R_[f'z_{c}'] = (R_[f'mean_{c}'] - mu) / sdv
R_['top_compartment'] = R_[[f'z_{c}' for c in COMP]].idxmax(axis=1).str.replace('z_', '')
R_['top_z'] = R_[[f'z_{c}' for c in COMP]].max(axis=1)
R_['detected'] = R_.top_z > 3
R_['correct'] = [t in e.split(';') if e else np.nan for t, e in zip(R_.top_compartment, R_.expected)]
R_.to_csv(GM / 'gene_map_validation.csv', index=False)
np.savez_compressed(GM / 'gene_maps_all.npz', **maps)
pd.set_option('display.width', 250)
print(R_[['key', 'trait', 'pct_var', 'expected', 'top_compartment', 'top_z', 'detected', 'correct'] +
         [f'z_{c}' for c in COMP]].round(2).to_string(index=False))
d = R_[~R_.key.str.startswith('control')]
print(f"\ndetected (top z > 3): {int(d.detected.sum())}/{len(d)}; correct among detected: "
      f"{int(d[d.detected].correct.sum())}/{int(d.detected.sum())}; correct overall: {int(d.correct.sum())}/{len(d)}")
print('done')
