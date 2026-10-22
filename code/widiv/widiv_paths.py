# Every input and output location for the WiDiv synthetic root dataset.
#
# WiDiv is kept apart from the MEMA root data the same way the seed and
# synthetic-stele datasets are: its own dataset folder, its own results folder,
# its own model folders. Its genotypes are real SNPs coded 0/1/2 rather than
# founder codes, so nothing written for the MEMA SNP table can be read in place
# of it by a script left at its defaults. paths.py itself is left untouched; this
# file builds on its root folders.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import DATASET_DIR, MODELS_DIR, RESULTS_DIR  # noqa: E402

WIDIV_DIR = DATASET_DIR / 'widiv'

# As delivered (README.txt in metadata/ describes every file). One PNG per plot,
# named <line>_B<block>.png, with a tissue mask of the same name and size.
# Images are 0.5 um per pixel, so canvas size follows root size.
WIDIV_IMAGES_DIR = WIDIV_DIR / 'images'
WIDIV_MASKS_DIR = WIDIV_DIR / 'masks'
WIDIV_PIXEL_SIZE_UM = 0.5

WIDIV_GENETIC_DIR = WIDIV_DIR / 'genetic'
WIDIV_GENOTYPES = WIDIV_GENETIC_DIR / 'genotypes_012.csv.gz'   # 1,051 lines x 90,790 SNPs
WIDIV_SNP_MAP = WIDIV_GENETIC_DIR / 'snp_map.csv'

WIDIV_METADATA_DIR = WIDIV_DIR / 'metadata'
WIDIV_SPLIT = WIDIV_METADATA_DIR / 'train_test_split.csv'
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

WIDIV_RESULTS_DIR = RESULTS_DIR / 'widiv'

# What the models train on, written by prepare_widiv_images.py. Every image is
# placed on one common canvas before downsampling, so all of them share one
# scale: canvas size equals about 1.07 x root diameter, so resizing each image
# on its own would draw every root the same size and erase root and stele
# diameter. 5,500 px holds the largest image (5,473 px). At 512 px that is
# 5.37 um per pixel, which keeps cortical cells (about 20 um) visible; at 256 px
# they alias into a checkerboard (results/widiv/resolution_test/).
WIDIV_CANVAS_PX = 5500
WIDIV_IMAGE_SIZE = 512
WIDIV_SCALED_DIR = WIDIV_RESULTS_DIR / f'scaled_{WIDIV_IMAGE_SIZE}'
WIDIV_SCALED_MASKS_DIR = WIDIV_RESULTS_DIR / f'scaled_{WIDIV_IMAGE_SIZE}_masks'
WIDIV_SCALED_METADATA = WIDIV_RESULTS_DIR / f'scaled_{WIDIV_IMAGE_SIZE}_metadata.csv'

# The genotype table parsed once into a matrix. Reading 90,790 columns of CSV
# takes far longer than loading the cache, which is rebuilt whenever the CSV changes.
WIDIV_GENOTYPE_CACHE = RESULTS_DIR / 'cache' / 'widiv_genotypes.npz'

# WiDiv gets its own autoencoder: the root LiteVAE never saw rendered images
# (18-21 dB inside the root on them) and is fixed at 256 px.
WIDIV_LITEVAE_DIR = MODELS_DIR / 'litevae_widiv'
WIDIV_LITEVAE_MODEL = WIDIV_LITEVAE_DIR / 'litevae_widiv_best.pt'
