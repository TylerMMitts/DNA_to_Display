# Builds the WiDiv metadata files in the same layout as the MEMA ones:
#   image_metadata.csv  the 11 standard columns, one row per image
#   genotype_list.csv   every line with a genotype
#   kinship_matrix.csv  SNP correlation between lines, for picking least-related
#                       partners in the memorisation tests
#   plot_table.csv      per image: line, block, splits, mask present, phenotypes
#
# Images keep their delivered names (<line>_B<block>.png) rather than the MEMA
# {genotype}_{rootnode}_{replication}_{rootnumber}.JPG pattern: the masks and
# the sender's measure_traits.py pair files by those names, and PNG stays
# lossless where JPG would blur the one-pixel tissue boundaries. Reads only the
# delivered files; never changes or removes them.

import numpy as np
import pandas as pd
from PIL import Image

from widiv_paths import (WIDIV_GENOTYPE_LIST, WIDIV_GENOTYPES, WIDIV_IMAGE_METADATA,
                         WIDIV_IMAGES_DIR, WIDIV_KINSHIP_MATRIX, WIDIV_MASKS_DIR,
                         WIDIV_PIXEL_SIZE_UM, WIDIV_PLOT_PHENOTYPES, WIDIV_PLOT_TABLE,
                         WIDIV_SPLIT)


def main():
    plots = pd.read_csv(WIDIV_PLOT_PHENOTYPES)
    for c in ('has_image', 'aerenchyma_capped'):
        plots[c] = plots[c].astype(str).str.upper().eq('TRUE')
    split = pd.read_csv(WIDIV_SPLIT).set_index('genotype')

    plots = plots[plots.has_image].reset_index(drop=True)
    rows = []
    for i, r in plots.iterrows():
        name = f'{r.id}.png'
        with Image.open(WIDIV_IMAGES_DIR / name) as im:
            w, h = im.size
        has_mask = (WIDIV_MASKS_DIR / name).exists()
        notes = []
        if not has_mask:
            notes.append('no mask delivered')
        if r.aerenchyma_capped:
            notes.append('aerenchyma capped')
        rows.append({
            'image_id': i + 1, 'new_filename': name, 'original_filename': name,
            'genotype': r.genotype, 'rootnode': 'synthetic', 'replication': int(r.block),
            'rootnumber': 1, 'pixel_size_um': WIDIV_PIXEL_SIZE_UM,
            'image_width_px': w, 'image_height_px': h,
            'quality_notes': '; '.join(notes) if notes else 'ok',
            'has_mask': has_mask,
        })
    meta = pd.DataFrame(rows)
    meta.drop(columns='has_mask').to_csv(WIDIV_IMAGE_METADATA, index=False)

    table = plots.drop(columns='has_image').rename(columns={'id': 'plot_id'})
    table.insert(1, 'filename', meta.new_filename)
    table.insert(4, 'split_random', table.genotype.map(split.split_random))
    table.insert(5, 'split_cluster', table.genotype.map(split.split_cluster))
    table.insert(6, 'cluster', table.genotype.map(split.cluster))
    table.insert(7, 'has_mask', meta.has_mask)
    table.insert(8, 'canvas_px', meta.image_width_px)
    table.to_csv(WIDIV_PLOT_TABLE, index=False)

    geno = pd.read_csv(WIDIV_GENOTYPES, index_col=0)
    pd.DataFrame({'genotype_id': geno.index}).to_csv(WIDIV_GENOTYPE_LIST, index=False)
    X = geno.to_numpy(np.float32)
    X -= X.mean(0)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    pd.DataFrame(X @ X.T, index=geno.index, columns=geno.index).round(5).to_csv(WIDIV_KINSHIP_MATRIX)

    print(f'{len(meta)} images, {meta.genotype.nunique()} imaged lines, '
          f'{int(meta.has_mask.sum())} with masks; {len(geno)} genotyped lines')
    print('canvas px vs root diameter (um), r =',
          round(float(np.corrcoef(table.canvas_px, table.root_diam_um)[0, 1]), 3),
          '| canvas / root diameter in px, median',
          round(float((table.canvas_px / (table.root_diam_um / WIDIV_PIXEL_SIZE_UM)).median()), 3))
    for s in ('split_random', 'split_cluster'):
        print(s, table.groupby(s).agg(images=('plot_id', 'size'), lines=('genotype', 'nunique'),
                                      masks=('has_mask', 'sum')).to_dict('index'))
    print('wrote', WIDIV_IMAGE_METADATA.parent)


if __name__ == '__main__':
    main()
