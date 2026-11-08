# Every input and output location for the WiDiv synthetic root dataset.
#
# WiDiv is kept apart from the MEMA root data the same way the seed and
# synthetic-stele datasets are: its own dataset folder, its own results folder,
# its own model folders. Its genotypes are real SNPs coded 0/1/2 rather than
# founder codes, so nothing written for the MEMA SNP table can be read in place
# of it by a script left at its defaults. paths.py itself is left untouched; this
# file builds on its root folders.

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import DATASET_DIR, MODELS_DIR, RESULTS_DIR  # noqa: E402

# The same pipeline serves every synthetic RootSlice dataset delivered in this
# format, chosen by the WIDIV_DATASET environment variable:
#   'widiv'  (default) the WiDiv diversity panel, 1,051 lines x 90,790 SNPs,
#            about four images per line, two splits
#   'ril_v5' a biparental RIL population (2026-11-05), 2,500 lines x 10,000
#            markers, one image per line, a random split only; its designed QTL
#            and true genetic values are held back by the sender
# Every folder below is named after the dataset, so a RIL run never reads or
# writes a WiDiv file.
WIDIV_DATASET = os.environ.get('WIDIV_DATASET', 'widiv')
if WIDIV_DATASET not in ('widiv', 'ril_v5'):
    raise SystemExit(f"WIDIV_DATASET must be 'widiv' or 'ril_v5', not {WIDIV_DATASET!r}")
RIL = WIDIV_DATASET == 'ril_v5'

WIDIV_DIR = DATASET_DIR / WIDIV_DATASET

# As delivered (README.txt in metadata/ describes every file). WiDiv: one PNG per
# plot, named <line>_B<block>.png. RIL: one PNG per line, named <line>.png. Each
# has a tissue mask of the same name and size. Images are 0.5 um per pixel, so
# canvas size follows root size.
WIDIV_IMAGES_DIR = WIDIV_DIR / 'images'
WIDIV_MASKS_DIR = WIDIV_DIR / 'masks'
WIDIV_PIXEL_SIZE_UM = 0.5

WIDIV_GENETIC_DIR = WIDIV_DIR / 'genetic'
# WiDiv: 1,051 lines x 90,790 SNPs. RIL: 2,500 lines x 10,000 markers.
WIDIV_GENOTYPES = WIDIV_GENETIC_DIR / 'genotypes_012.csv.gz'
# snp, chr, pos (bp). The RIL's marker_map.csv is rewritten to this layout by organize_ril.py.
WIDIV_SNP_MAP = WIDIV_GENETIC_DIR / 'snp_map.csv'

WIDIV_METADATA_DIR = WIDIV_DIR / 'metadata'
WIDIV_SPLIT = WIDIV_METADATA_DIR / 'train_test_split.csv'
# The delivered split columns. The RIL is one cross, so it has no cluster split.
WIDIV_SPLITS = ('split_random',) if RIL else ('split_random', 'split_cluster')
if RIL:
    # Held back by the sender, who scores the generated roots against them.
    WIDIV_GENETIC_VALUES = None
    WIDIV_QTL = None
    # The delivered workbook's one sheet as CSV (organize_ril.py). One image per
    # line, so the per-image table is also the line means.
    WIDIV_PLOT_PHENOTYPES = WIDIV_METADATA_DIR / 'synthetic_phenotypes_ril_v5__phenotypes.csv'
    WIDIV_GENOTYPE_MEANS = WIDIV_PLOT_PHENOTYPES
else:
    WIDIV_GENETIC_VALUES = WIDIV_METADATA_DIR / 'genetic_values.csv'
    # The two delivered workbooks, one CSV per sheet (exported so no Excel reader is needed).
    WIDIV_PLOT_PHENOTYPES = WIDIV_METADATA_DIR / 'synthetic_phenotypes_widiv_hard__plot_phenotypes.csv'
    WIDIV_GENOTYPE_MEANS = WIDIV_METADATA_DIR / 'synthetic_phenotypes_widiv_hard__genotype_means.csv'
    WIDIV_QTL = WIDIV_METADATA_DIR / 'qtl_and_gwas_baseline__qtl.csv'

# Written by organize_widiv.py, in the same layout as the MEMA metadata.
WIDIV_IMAGE_METADATA = WIDIV_METADATA_DIR / 'image_metadata.csv'
WIDIV_GENOTYPE_LIST = WIDIV_METADATA_DIR / 'genotype_list.csv'
WIDIV_KINSHIP_MATRIX = WIDIV_METADATA_DIR / 'kinship_matrix.csv'
# One row per image: line, block, both splits, whether a mask exists, phenotypes.
WIDIV_PLOT_TABLE = WIDIV_METADATA_DIR / 'plot_table.csv'

WIDIV_RESULTS_DIR = RESULTS_DIR / WIDIV_DATASET

# What the models train on, written by prepare_widiv_images.py. Every image is
# placed on one common canvas before downsampling, so all of them share one
# scale: canvas size equals about 1.07 x root diameter, so resizing each image
# on its own would draw every root the same size and erase root and stele
# diameter. 5,500 px holds the largest image (5,473 px). At 512 px that is
# 5.37 um per pixel, which keeps cortical cells (about 20 um) visible; at 256 px
# they alias into a checkerboard (results/widiv/resolution_test/). The RIL uses
# the same canvas, so its images share WiDiv's 5.37 um per pixel and every
# size-based setting carries over; prepare_widiv_images.py stops if an image
# is larger than the canvas.
WIDIV_CANVAS_PX = 5500
WIDIV_IMAGE_SIZE = 512
WIDIV_SCALED_DIR = WIDIV_RESULTS_DIR / f'scaled_{WIDIV_IMAGE_SIZE}'
WIDIV_SCALED_MASKS_DIR = WIDIV_RESULTS_DIR / f'scaled_{WIDIV_IMAGE_SIZE}_masks'
WIDIV_SCALED_METADATA = WIDIV_RESULTS_DIR / f'scaled_{WIDIV_IMAGE_SIZE}_metadata.csv'

# The genotype table parsed once into a matrix. Reading 90,790 columns of CSV
# takes far longer than loading the cache, which is rebuilt whenever the CSV changes.
WIDIV_GENOTYPE_CACHE = RESULTS_DIR / 'cache' / f'{WIDIV_DATASET}_genotypes.npz'

# Every scaled image encoded once by the frozen WiDiv LiteVAE, in all eight
# quarter-turn and flip views, written by train_widiv.py. One file per LiteVAE
# checkpoint, named after it, so a retrained autoencoder never reuses old latents.
WIDIV_LATENT_CACHE_DIR = RESULTS_DIR / 'cache' / f'{WIDIV_DATASET}_latents'

# WiDiv gets its own autoencoder: the root LiteVAE never saw rendered images
# (18-21 dB inside the root on them) and is fixed at 256 px.
#
# v3 adds a background prior: latents outside the root are pulled to a fixed
# value, so root and background differ in latent value (train_litevae_widiv.py).
# v2 (linear output, background-weighted loss) trained every diffusion model up
# to the layout-channel runs; models/litevae_widiv/ holds the first run, which
# collapsed to drawing every image white. Both are kept; nothing reads them by default.
#
# The RIL gets its own autoencoder trained with the same v3 recipe (512 px,
# background-weighted loss, background prior), hence litevae_ril_v5_v3.
WIDIV_LITEVAE_DIR = MODELS_DIR / f'litevae_{WIDIV_DATASET}_v3'
WIDIV_LITEVAE_MODEL = WIDIV_LITEVAE_DIR / f'{WIDIV_LITEVAE_DIR.name}_best.pt'
WIDIV_LITEVAE_V2_MODEL = MODELS_DIR / 'litevae_widiv_v2' / 'litevae_widiv_v2_best.pt'
