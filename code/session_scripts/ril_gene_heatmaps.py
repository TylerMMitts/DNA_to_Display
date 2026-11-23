# Genotype-to-phenotype maps: where in the root each gene acts. Best RIL model (windows e1200),
# validation lines only. For each gene: swap its chromosome region to the other parent's
# version on up to 48 unseen lines (2 seeds, same noise before/after), align each pair on the
# unedited root (centre and radius), and average |edited - original| per pixel. A region with
# no designed gene is the control. Answer key used only to pick the designed gene positions.
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / 'ril_poster_heroes.py').read().split('# 1. Mosaic.')[0])
from PIL import Image

N_LINES, SEEDS2, CANON = 48, (30, 2), 256
GENES = [  # (key, marker, block cM, title, subtitle)
    ('shared', '1_814', 10, 'Shared gene', 'stele size and vessel number'),
    ('mxdiam', '5_2477', 10, 'Vessel-width gene', 'metaxylem diameter'),
    ('cortex', '3_255', 10, 'Cortex gene', 'cortex width (25% of variance)'),
    ('rca', '9_2156', 2, 'Aerenchyma gene', 'air channels in the cortex'),
    ('lacuna', '6_2889', 10, 'Lacuna gene', 'trait never measured or given'),
    ('control', None, 10, 'Control region', 'no designed gene (chr 10)'),
]
snp_list = list(snps)
cM = smap.pos_cM.to_numpy(); ch = smap.chr.to_numpy()


def block_for(marker, width, control=False):
    if control:
        c0, p0 = 10, 20.0
    else:
        c0, p0 = smap.loc[marker, 'chr'], smap.loc[marker, 'pos_cM']
    return np.flatnonzero((ch == c0) & (np.abs(cM - p0) <= width)), (snp_list.index(marker) if marker else None)


def align(im, cy, cx, R):
    half = 1.3 * R
    c = crop(im, cy, cx, half)
    return np.asarray(Image.fromarray(c).resize((CANON, CANON), Image.LANCZOS)).astype(np.float32)


lineset = val[:80]
base_cache = {}
maps, templates, info = {}, [], {}
for key, marker, width, title, subtitle in GENES:
    blk, col = block_for(marker, width, control=(key == 'control'))
    lines_q = [g for g in lineset if col is None or dosage[row_of[g], col] != 1][:N_LINES]
    D0 = dosage[[row_of[g] for g in lines_q]]
    D1 = D0.copy()
    for r in range(len(lines_q)):
        hom = blk[D1[r, blk] != 1]
        D1[r, hom] = 2 - D1[r, hom]
    acc, n = np.zeros((CANON, CANON), np.float32), 0
    for sd in SEEDS2:
        need = [g for g in lines_q if (g, sd) not in base_cache]
        if need:
            for g, im in zip(need, generate(dosage[[row_of[g] for g in need]], need, [sd] * len(need))):
                base_cache[(g, sd)] = im
        ed = generate(D1, lines_q, [sd] * len(lines_q))
        for g, e in zip(lines_q, ed):
            b = base_cache[(g, sd)]
            cy, cx, R = root_box(b)
            ab, ae = align(b, cy, cx, R), align(e, cy, cx, R)
            acc += np.abs(ae - ab).mean(-1) / 255
            n += 1
            if key == 'shared':
                templates.append(ab)
    maps[key] = acc / n
    info[key] = {'marker': marker, 'block_cM': width, 'n_pairs': n, 'n_markers_swapped': int(len(blk)),
                 'mean_change': float(maps[key].mean())}
    print(key, info[key], flush=True)

np.savez_compressed(OUT / 'gene_heatmaps.npz', template=np.mean(templates, 0), **maps)
json.dump(info, open(OUT / 'gene_heatmaps.json', 'w'), indent=1)
print('done')
