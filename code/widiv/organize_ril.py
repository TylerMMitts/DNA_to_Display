# Builds the RIL (ril_v5) metadata in the layout the WiDiv pipeline reads, so
# prepare_widiv_images.py, train_litevae_widiv.py and train_widiv.py run on it
# unchanged with WIDIV_DATASET=ril_v5 (widiv_paths.py):
#   synthetic_phenotypes_ril_v5__<sheet>.csv  each workbook sheet as CSV (the
#                       venv has no Excel reader, so the .xlsx is read here with
#                       the standard library)
#   genetic/snp_map.csv marker_map.csv as snp, chr, pos (bp), pos_cM - the
#                       column names the window encoding reads
#   image_metadata.csv  the 11 standard columns, one row per image
#   genotype_list.csv   every line with a genotype
#   kinship_matrix.csv  marker correlation between lines
#   plot_table.csv      per image: line, split, mask present, canvas size, phenotypes
#
# The RIL has one image per line, named <line>.png, and only split_random; its
# designed QTL and true genetic values are held back by the sender. Reads only
# the delivered files; never changes or removes them.
#
#     python code/widiv/organize_ril.py

import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

# This script only ever organizes the RIL set.
os.environ.setdefault('WIDIV_DATASET', 'ril_v5')
if os.environ['WIDIV_DATASET'] != 'ril_v5':
    raise SystemExit("organize_ril.py organizes the RIL set; unset WIDIV_DATASET or set it to ril_v5")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from PIL import Image

from widiv_paths import (WIDIV_GENETIC_DIR, WIDIV_GENOTYPE_LIST, WIDIV_GENOTYPES, WIDIV_IMAGE_METADATA,
                         WIDIV_IMAGES_DIR, WIDIV_KINSHIP_MATRIX, WIDIV_MASKS_DIR, WIDIV_METADATA_DIR,
                         WIDIV_PIXEL_SIZE_UM, WIDIV_PLOT_PHENOTYPES, WIDIV_PLOT_TABLE, WIDIV_SNP_MAP,
                         WIDIV_SPLIT)
from widiv_data import load_split

WORKBOOK = WIDIV_METADATA_DIR / 'synthetic_phenotypes_ril_v5_final_2026-11-05.xlsx'
MARKER_MAP = WIDIV_GENETIC_DIR / 'marker_map.csv'
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}


# Column letters of a cell reference ("AB12") as a 0-based index.
def column_index(ref):
    letters = re.match(r'[A-Z]+', ref).group()
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n - 1


# Every sheet of an .xlsx as a DataFrame, first row as the header.
def read_xlsx(path):
    with zipfile.ZipFile(path) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', NS):
                shared.append(''.join(t.text or '' for t in si.iter(f"{{{NS['m']}}}t")))
        rels = {r.get('Id'): r.get('Target') for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        sheets = {}
        for sheet in ET.fromstring(z.read('xl/workbook.xml')).find('m:sheets', NS):
            target = rels[sheet.get(f"{{{NS['r']}}}id")].lstrip('/')
            target = target if target.startswith('xl/') else 'xl/' + target
            rows = []
            for row in ET.fromstring(z.read(target)).iter(f"{{{NS['m']}}}row"):
                values = {}
                for c in row.findall('m:c', NS):
                    kind, v = c.get('t'), c.find('m:v', NS)
                    if kind == 's':
                        value = shared[int(v.text)]
                    elif kind == 'inlineStr':
                        value = ''.join(t.text or '' for t in c.iter(f"{{{NS['m']}}}t"))
                    elif kind == 'b':
                        value = v.text == '1'
                    elif v is None:
                        value = None
                    else:
                        value = v.text
                    values[column_index(c.get('r'))] = value
                rows.append([values.get(i) for i in range(max(values) + 1)] if values else [])
            width = max(len(r) for r in rows)
            rows = [r + [None] * (width - len(r)) for r in rows]
            frame = pd.DataFrame(rows[1:], columns=rows[0])
            # Numbers come back as text; convert every column that is wholly numeric.
            for col in frame.columns:
                converted = pd.to_numeric(frame[col], errors='coerce')
                if converted.notna().sum() == frame[col].notna().sum():
                    frame[col] = converted
            sheets[sheet.get('name')] = frame
    return sheets


def main():
    sheets = read_xlsx(WORKBOOK)
    for name, frame in sheets.items():
        out = (WIDIV_PLOT_PHENOTYPES if len(sheets) == 1
               else WIDIV_METADATA_DIR / f'synthetic_phenotypes_ril_v5__{name}.csv')
        frame.to_csv(out, index=False)
        print(f"sheet {name!r}: {frame.shape[0]} rows x {frame.shape[1]} columns -> {out.name}")
    if not WIDIV_PLOT_PHENOTYPES.exists():
        raise SystemExit(f"the workbook has {len(sheets)} sheets ({list(sheets)}); point "
                         f"WIDIV_PLOT_PHENOTYPES in widiv_paths.py at the per-line one")
    pheno = pd.read_csv(WIDIV_PLOT_PHENOTYPES)
    line_col = next(c for c in ('line', 'genotype', 'id') if c in pheno.columns)
    pheno = pheno.rename(columns={line_col: 'genotype'})
    print('phenotype columns:', list(pheno.columns))

    snp_map = pd.read_csv(MARKER_MAP).rename(columns={'marker': 'snp', 'pos_bp': 'pos'})
    snp_map[['snp', 'chr', 'pos', 'pos_cM']].to_csv(WIDIV_SNP_MAP, index=False)
    print(f"snp_map.csv: {len(snp_map)} markers on {snp_map.chr.nunique()} chromosomes, "
          f"{snp_map.groupby('chr').pos_cM.max().sum():.0f} cM in total")

    split = load_split(WIDIV_SPLIT)
    rows, empty_images, empty_masks = [], [], []
    for genotype in pheno.genotype:
        name = f'{genotype}.png'
        image, mask = WIDIV_IMAGES_DIR / name, WIDIV_MASKS_DIR / name
        if not image.exists():
            continue
        # A few files were delivered empty (0 bytes); they count as not delivered.
        if image.stat().st_size == 0:
            empty_images.append(name)
            continue
        with Image.open(image) as im:
            w, h = im.size
        has_mask = mask.exists() and mask.stat().st_size > 0
        if mask.exists() and not has_mask:
            empty_masks.append(name)
        rows.append({'genotype': genotype, 'filename': name, 'width': w, 'height': h,
                     'has_mask': has_mask})
    found = pd.DataFrame(rows)
    if empty_images or empty_masks:
        print(f"delivered empty, treated as missing: images {empty_images}, masks {empty_masks}")
    missing = len(pheno) - len(found)
    meta = pd.DataFrame({
        'image_id': np.arange(1, len(found) + 1), 'new_filename': found.filename,
        'original_filename': found.filename, 'genotype': found.genotype, 'rootnode': 'synthetic',
        'replication': 1, 'rootnumber': 1, 'pixel_size_um': WIDIV_PIXEL_SIZE_UM,
        'image_width_px': found.width, 'image_height_px': found.height,
        'quality_notes': np.where(found.has_mask, 'ok', 'no mask delivered'),
    })
    meta.to_csv(WIDIV_IMAGE_METADATA, index=False)

    table = found[['genotype', 'filename']].copy()
    table.insert(0, 'plot_id', table.genotype)
    table['block'] = 1
    table['split_random'] = table.genotype.map(split.split_random)
    table['has_mask'] = found.has_mask.to_numpy()
    table['canvas_px'] = found.width.to_numpy()
    table = table.merge(pheno, on='genotype', how='left')
    table.to_csv(WIDIV_PLOT_TABLE, index=False)

    geno = pd.read_csv(WIDIV_GENOTYPES, index_col=0)
    lines = geno.index.tolist()
    pd.DataFrame({'genotype_id': lines}).to_csv(WIDIV_GENOTYPE_LIST, index=False)
    X = geno.to_numpy(np.float32)
    n_missing = int(np.isnan(X).sum())
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    het = float((X == 1).mean())
    X -= X.mean(0)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    pd.DataFrame(X @ X.T, index=lines, columns=lines).round(5).to_csv(WIDIV_KINSHIP_MATRIX)

    print(f"{len(meta)} images ({missing} phenotype rows without an image), "
          f"{int(found.has_mask.sum())} with masks; {len(lines)} genotyped lines x {geno.shape[1]:,} markers, "
          f"{n_missing} missing calls, {het:.1%} heterozygous")
    print('canvas px:', int(table.canvas_px.min()), '-', int(table.canvas_px.max()))
    if 'root_diam_um' in table:
        print('canvas / root diameter in px, median',
              round(float((table.canvas_px / (table.root_diam_um / WIDIV_PIXEL_SIZE_UM)).median()), 3))
    print('split_random:', table.split_random.value_counts().to_dict(),
          '| imaged lines missing from the split file:', int(table.split_random.isna().sum()))
    print('wrote', WIDIV_METADATA_DIR)


if __name__ == '__main__':
    main()
