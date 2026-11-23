# IMPC mouse X-rays (BCM centre): `WIDIV_DATASET=impc_bcm`

Whole-body dorso-ventral X-rays of knockout and wild-type mice from the International
Mouse Phenotyping Consortium (www.mousephenotype.org, CC BY 4.0; cite the IMPC portal
and its publication), phenotyping centre BCM (Baylor College of Medicine). Downloaded
2026-11-16 with `code/impc/fetch_impc_xray.py`.

| | |
|---|---|
| Mice | 8,835 (1,880 wild-type controls), C57BL/6N background, about 14 weeks old |
| Knocked-out genes | 718, about 10 mice each (about 5 per sex) |
| X-rays | 8,819 downloaded (16 missing on the IMPC server); 8,803 used (16 from imaging setups too rare to calibrate) |
| Ground truth | the IMPC's own significant calls: 78 X-ray calls on 33 genes (tibia length, skull, teeth, vertebral fusion, pelvis), 513 body-composition (DEXA) calls, 5,657 in all on 609 genes |
| Measurements | per mouse: tibia length, rib / vertebra / digit counts, and 23 categorical X-ray observations |

## Files

```
dataset/impc_bcm/
├── images/<observation_id>.jpg        X-rays as published (2.6 GB, laptop only)
├── genetic/
│   ├── genotypes_012.csv.gz           mouse x 817 columns: 718 genes (0 wt / 1 het / 2 hom),
│   │                                  SEX_male, 97 BATCH_<YYYY-MM> imaging months
│   ├── snp_map.csv                    column, chromosome, position
│   └── encoder_columns.csv            every column (the qtl_only encoder reads all)
├── metadata/
│   ├── impc_xray_images.csv           IMPC record per image
│   ├── impc_xray_measurements.csv     IMPC X-ray measurements, long format
│   ├── impc_gene_positions.csv
│   ├── mouse_table.csv                one row per mouse: gene, zygosity, sex, dates,
│   │                                  measurements, split  (scoring only)
│   ├── train_test_split.csv           line (= mouse M00001...), split_random
│   └── genotype_list.csv
└── answer_key/impc_significant_calls.csv   scoring only, never a model input
results/impc_bcm/
├── scaled_512/ + scaled_512_metadata.csv   training images (1.1 GB) and their QC
├── scale_groups.csv                        imaging-setup scale factors
├── pipeline_check/                          check.json, features.csv
└── size_analysis/size_analysis.csv          model-size comparison
```

## How the data is shaped for the pipeline

- **One "line" per mouse.** Each mouse has its own genotype row. The split is at the mouse level: 15% of every gene × zygosity × sex group is test, and 10% of the rest is validation.
  - Held-out mice share their genotype with training mice. A model cannot generate a gene it never saw, so this dataset tests new mice of known genotypes, not unseen genotypes.
  - Every test mouse's condition occurs in training.
- **Encoder: `qtl_only` over all 817 columns.** This is a learned embedding per gene, sex and imaging month. The PCA and window encoders are built for SNP panels and do not fit one knockout per mouse.
- **Images: negatives on white,** so the root pipeline's white-background logic applies unchanged. Each mouse is rotated head-up (median correction 1.1°) and cropped to one physical scale (`code/impc/prepare_impc_images.py`). Augmentation is a left-right mirror only (`WIDIV_UPRIGHT`).

## Verification (`code/impc/check_impc_pipeline.py`)

**Plumbing.** Everything the trainers read is in place:
- the genotype matrix (8,835 × 817)
- the split: 6,575 train / 730 validation / 1,530 test mice
- 8,803 images at 512 px RGB with white corners
- the encoder forward pass: 8 tokens × 256
- augmentation: mirror only, never a quarter turn

**Two problems found in the raw data, both fixed:**

1. **The imaging scale changed.** Control mice measure about 1,290 px of trunk on images from 2013 to Nov 2015, and about 1,527 px afterwards. The IMPC's tibia lengths in mm stay flat over the same years, so this is a magnification change, not biology. Each imaging setup (identified by its JPEG width) is rescaled to the reference scale (`scale_groups.csv`).
2. **Imaging date is a strong batch effect.** Among wild-type controls alone, the imaging month explains 75% of trunk-length and 81% of X-ray-darkness variance. A knockout line is imaged within a few weeks (median 2), so without correction a model would learn each line's date as its gene effect.
   - The genotype matrix therefore includes the imaging month, and gene effects are read against controls of the same month, as in the IMPC's own analysis.
   - 69% of knockout mice were imaged in a week that also had at least 3 controls.

**Is there a genetic signal in the images?** It was measured on simple image features, each mouse against same-month, same-sex controls, with a null that shuffles genotypes within month × sex (so it keeps all batch effects):

| Feature | Variance explained by genotype | Chance level | Excess |
|---|---|---|---|
| Silhouette area | 0.32 | 0.20 | +0.12 (p ≤ 0.005) |
| Trunk length | 0.37 | 0.22 | +0.15 (p ≤ 0.005) |
| X-ray darkness | 0.39 | 0.21 | +0.18 (p ≤ 0.005) |

The genes the IMPC itself calls differ more from same-month controls in body area:

| Genes | Mean \|z\| in body area | Other genes |
|---|---|---|
| X-ray-called (33) | 1.04 | 0.54 |
| Body-composition (DEXA)-called (114) | 0.80 | 0.51 |

**Verdict:** the data works for the pipeline, provided imaging batch is conditioned on, which is done.

**Expect:**
- **Body size and shape:** clear effects.
- **Fine skeletal phenotypes:** most of the X-ray calls are tibia-length changes of a fraction of a mm. At 4 source px per training px these are probably below what a 512 px whole-body image can show.

## Model size (`code/impc/model_size_analysis.py`)

| | WiDiv roots | RIL v5 roots | IMPC X-rays |
|---|---|---|---|
| Training images | 2,930 | 1,780 | 6,574 |
| Genotype conditions | 733 | 1,780 | 1,543 (+ month) |
| Images per condition | 4.0 | 1.0 | 4.3 |
| PCs for 90% of image variance (128 px) | 26 | 40 | 69 |
| Fine detail (mean \|Laplacian\| on the subject) | 0.31 | 0.32 | 0.06 |
| Parameters per image, medium (8.8M) | 3,000 | 4,900 | 1,340 |
| Parameters per image, large (29.3M) | 10,000 | 16,500 | 4,460 |

**Recommendation: large, with medium alongside as a baseline.** Both are in the batch script.
- **Capacity per image:** medium would run at about a quarter of the capacity per image any earlier model had. Large brings it back to the ratio medium had on the RIL.
- **Memorisation risk is lower than the RIL's:** about 4 mice share each condition, so a genotype cannot become a name tag for a single image.
- **More global variety** (pose, limbs, tail): 69 PCs for 90% of variance, against 26–40 for the roots.
- **Less fine detail per pixel** than the rendered roots, so the LiteVAE keeps its size.

Choose between the two on validation loss, genotype gain and memorisation ratio.

## Running it

On the laptop, done once:

```
python code/impc/fetch_impc_xray.py --centre BCM
WIDIV_DATASET=impc_bcm python code/impc/organize_impc.py
WIDIV_DATASET=impc_bcm python code/impc/prepare_impc_images.py
python code/impc/check_impc_pipeline.py
```

Copy to Hellbender:
- `dataset/impc_bcm/genetic/`
- `dataset/impc_bcm/metadata/train_test_split.csv`
- `results/impc_bcm/scaled_512/`
- `results/impc_bcm/scaled_512_metadata.csv`

Then run:

```
sbatch code/impc/train_litevae_impc_hellbender.sbatch      # first: the X-ray autoencoder
sbatch code/impc/train_impc_hellbender.sbatch              # then: medium (0) and large (1)
```
