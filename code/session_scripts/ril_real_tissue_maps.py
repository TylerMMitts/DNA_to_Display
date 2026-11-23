# The QTL effects as they are in the ORIGINAL data, no model involved.
#  1. Effect sizes: the designed effect of each QTL (true genetic values, answer key) and the
#     same effect measured on the original images' phenotypes, both from one OLS of the trait
#     on the 14 designed markers (+ the A x A product) over the 1,979 non-test lines.
#  2. Tissue maps from the original images: every non-test line's real 512 px image, centred on
#     its root at the fixed 5.38 um/px scale, cut into the same five tissues as the generated
#     maps; per pixel, OLS of each tissue on the same markers + the 6 control markers. The
#     coefficient of a marker is the tissue gained (+) / lost (-) going from dosage 0 to 2 -
#     the same orientation as the generated swap maps.
# Test lines are dropped on read and never touched.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import load_tables
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import widiv_traits as WT
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_SNP_MAP, WIDIV_SCALED_DIR, WIDIV_RESULTS_DIR
from widiv_snp_encoding import load_widiv_genotypes

GM = WIDIV_RESULTS_DIR / 'gene_maps'
AK = 'E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/'
CANON = 256
COMP = ['vessels', 'stele', 'aerenchyma', 'outer', 'cortex']
meta, split = load_tables()                                 # non-test lines only
lines_all, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
row_of = {g: i for i, g in enumerate(lines_all)}
snp_list = list(snps)
smap = pd.read_csv(WIDIV_SNP_MAP).set_index('snp').loc[snp_list]
ch, cMv = smap.chr.to_numpy(), smap.pos_cM.to_numpy()
q = pd.read_csv(AK + 'designed_qtl_markers.csv')
QTL = q.qtl.tolist()
ctrl_markers = []
for c, p in [(2, 20.0), (3, 120.0), (4, 40.0), (5, 20.0), (7, 40.0), (10, 20.0)]:  # as the generated maps
    ctrl_markers.append(smap.index[(ch == c)][np.argmin(np.abs(cMv[ch == c] - p))])
keys = QTL + [f'control{i + 1}' for i in range(6)]
marks = q.snp.tolist() + ctrl_markers
lines = [g for g in meta.index if g in row_of and (WIDIV_SCALED_DIR / f'{g}.png').exists()]
assert not set(lines) & set(split[split == 'test'].index)
print(len(lines), 'non-test lines with images', flush=True)

D = dosage[np.ix_([row_of[g] for g in lines], [snp_list.index(m) for m in marks])].astype(np.float64)
cc = D - 1.0                                                 # -1 / 0 / +1
ia, ie = QTL.index('mx_diam_add'), QTL.index('mx_diam_epi')
X = np.column_stack([np.ones(len(lines)), D / 2, cc[:, ia] * cc[:, ie]])   # main: 0 -> 2 effect
cols = ['intercept'] + keys + ['AxA']
XtXi = np.linalg.inv(X.T @ X)

# ---- 1. effect sizes ------------------------------------------------------------------
gv = pd.read_csv(AK + 'genetic_values.csv').set_index('line').loc[lines]
TR = ['stele_diam_um', 'mx_n', 'mx_diam_um', 'cortex_width_um', 'rca_pct', 'n_lacunae']
rows = []
for t in TR:
    bt = XtXi @ X.T @ gv[f'g_{t}'].to_numpy()
    bm = XtXi @ X.T @ meta.loc[lines, t].to_numpy() if t in meta.columns else np.full(len(cols), np.nan)
    for j, k in enumerate(cols[1:], 1):
        rows.append({'trait': t, 'term': k, 'designed_effect': bt[j], 'original_images_effect': bm[j]})
E = pd.DataFrame(rows)
# Aerenchyma genetic values are on the liability scale (answer key README); put the designed
# effect on the image % scale with the linear fit of the phenotype on the genetic value.
gr, pr = gv['g_rca_pct'].to_numpy(), meta.loc[lines, 'rca_pct'].to_numpy()
rca_scale = float(np.cov(pr, gr)[0, 1] / np.var(gr, ddof=1))
E['designed_effect_image_units'] = np.where(E.trait == 'rca_pct', E.designed_effect * rca_scale, E.designed_effect)
print('rca liability -> % slope', round(rca_scale, 2))
E.to_csv(GM / 'qtl_effect_sizes_original.csv', index=False)
print(E[E.term.isin(QTL + ['AxA'])].pivot(index='term', columns='trait', values='designed_effect').round(2).to_string())

# ---- 2. tissue maps from the original images -------------------------------------------
STELE_L = [WT.IDX[l] for l in (5, 6, 7, 8, 9)]
OUTER_L = [WT.IDX[l] for l in (1, 2, 3)]


def root_box(im):
    # root = everything the tissue classifier does not call background, closed and filled
    # (a darkness threshold misses the pale outer cells of the simulated roots)
    m = ndimage.binary_opening(WT.classify(im) != WT.WHITE, iterations=1)
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=10))
    lab, nn = ndimage.label(m)
    if nn == 0:
        return 256, 256, 64
    big = 1 + int(np.argmax(ndimage.sum(np.ones_like(lab), lab, range(1, nn + 1))))
    ys, xs = np.nonzero(lab == big)
    return (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2, max(ys.max() - ys.min(), xs.max() - xs.min()) / 2


def crop(im, cy, cx, half):
    pad = np.pad(im, ((400, 400), (400, 400), (0, 0)), constant_values=255)
    cy, cx, half = int(round(cy)) + 400, int(round(cx)) + 400, int(round(half))
    return pad[cy - half:cy + half, cx - half:cx + half]


def onehot(rgb):      # identical to the generated-map tissues
    cls = WT.classify(rgb)
    white = cls == WT.WHITE
    root = ndimage.binary_fill_holes(ndimage.binary_closing(~white, iterations=6))
    stele = ndimage.binary_opening(ndimage.binary_fill_holes(np.isin(cls, STELE_L)), iterations=1)
    vessels = white & ndimage.binary_erosion(stele, iterations=1)
    outer = (np.isin(cls, OUTER_L) | (root & ~ndimage.binary_erosion(root, iterations=3))) & ~stele
    aer = white & root & ~stele & ~outer
    cortex = root & ~stele & ~outer & ~aer
    return np.stack([vessels, stele & ~vessels, aer, outer, cortex]).astype(np.float32)


load = lambda g: np.asarray(Image.open(WIDIV_SCALED_DIR / f'{g}.png').convert('RGB'))
boxes = {g: root_box(load(g)) for g in lines}
HALF = 1.3 * float(np.median([b[2] for b in boxes.values()]))     # one fixed frame for every root
print('frame half-size px', round(HALF, 1), 'radius px 5/50/95%', np.percentile([b[2] for b in boxes.values()], [5, 50, 95]).round(1), flush=True)
P = 5 * CANON * CANON
XtY = np.zeros((X.shape[1], P))
YY = np.zeros(P)
for i, g in enumerate(lines):
    cy, cx, _ = boxes[g]
    c = np.asarray(Image.fromarray(crop(load(g), cy, cx, HALF)).resize((CANON, CANON), Image.NEAREST))
    y = onehot(c).ravel().astype(np.float64)
    XtY += np.outer(X[i], y)
    YY += y * y
    if i % 200 == 0:
        print(i, flush=True)
B = XtXi @ XtY                                               # (k, P)
rss = np.maximum(YY - np.einsum('kp,kp->p', B, XtY), 0)
se = np.sqrt(rss / (len(lines) - X.shape[1]))[None] * np.sqrt(np.diag(XtXi))[:, None]
tmpl = B[0].reshape(5, CANON, CANON) + (X[:, 1:].mean(0) @ B[1:]).reshape(5, CANON, CANON)   # mean root
np.savez_compressed(GM / 'qtl_maps_original_images.npz', template=tmpl.astype(np.float32), half=HALF,
                    **{f'{k}__tissue': B[j].reshape(5, CANON, CANON).astype(np.float32) for j, k in enumerate(cols) if j},
                    **{f'{k}__t': (B[j] / np.maximum(se[j], 1e-9)).reshape(5, CANON, CANON).astype(np.float32)
                       for j, k in enumerate(cols) if j})

# Score like the generated maps: per-tissue strength, z against the 6 controls (leave-one-out).
root_mask = tmpl.sum(0) > 0.5
EXPECT = {'mx_n': {'vessels', 'stele'}, 'stele_diam_um': {'vessels', 'stele'}, 'mx_diam_um': {'vessels'},
          'cortex_width_um': {'cortex', 'outer'}, 'rca_pct': {'aerenchyma'}, 'n_lacunae': {'aerenchyma'}}
S = {k: [float(np.abs(B[j + 1].reshape(5, CANON, CANON)[t])[root_mask].mean()) for t in range(5)] for j, k in enumerate(keys)}
gen = np.load(GM / 'gene_maps_methods.npz')
out = []
for k in keys:
    pool = np.array([S[c] for c in keys if c.startswith('control') and c != k])
    z = (np.array(S[k]) - pool.mean(0)) / pool.std(0, ddof=1)
    top = COMP[int(np.argmax(z))]
    tr = q.set_index('qtl').trait.get(k, 'none')
    exp = set().union(*[EXPECT[t] for t in tr.split(';')]) if tr != 'none' else set()
    real_map = B[keys.index(k) + 1].reshape(5, CANON, CANON)
    gen_map = gen[f'{k}__tissue']
    r_maps = float(np.corrcoef(real_map[:, root_mask].ravel(), gen_map[:, root_mask].ravel())[0, 1])
    out.append({'key': k, 'trait': tr, 'expected': ';'.join(sorted(exp)), 'points_to': top, 'top_z': float(z.max()),
                'called': bool(z.max() > 3), 'correct': (top in exp) if exp else np.nan,
                'r_original_vs_generated_map': r_maps, **{f'z_{c}': float(v) for c, v in zip(COMP, z)}})
O = pd.DataFrame(out)
O.to_csv(GM / 'qtl_maps_original_images_scores.csv', index=False)
pd.set_option('display.width', 250)
print(O.round(2).to_string(index=False))
g_ = O[~O.key.str.startswith('control')]
print('original images: correct', int(g_.correct.sum()), '/ 14; called', int(g_.called.sum()),
      '; controls called', int(O[O.key.str.startswith('control')].called.sum()))
print('done')
