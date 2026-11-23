## Complete Project Structure

```
project_root/
├── code/                          # Source only - never written to
│   ├── paths.py                  # Every file location in the project
│   ├── crop_root_model.py        # YOLOv8-based root detection
│   ├── litevae/                  # Image autoencoder
│   │   ├── models/               # Encoder, decoder, blocks, wavelet
│   │   ├── train_litevae.py      # Training entry point
│   │   └── evaluation/           # Reconstruction tests
│   ├── latent_diffusion/         # Genotype-conditioned image generation
│   │   ├── models/               # UNet, SNP encoders, LDM wrapper
│   │   ├── data/                 # Dataset classes
│   │   ├── diffusion/            # Scheduler and sampling
│   │   ├── utils/                # Attention collection helpers
│   │   ├── training/             # train_onehot.py (current), train.py (legacy)
│   │   ├── generation/           # Image generation entry points
│   │   ├── analysis/             # SNP attention and contribution studies
│   │   └── validation/           # Encoding correctness checks
│   └── feature_segmentation/     # Trait measurement
│       ├── prepare_dataset.py    # Annotations -> YOLO dataset
│       ├── train_segmentation.py # Training entry point
│       ├── vessel_counting.py    # Vessel counting methods
│       └── evaluation/           # Fidelity and accuracy tests
├── dataset/                       # All data files (primary working data)
│   ├── images/                   # All LAT images (FLAT - no subfolders)
│   │   └── [image].JPG
│   └── metadata/                 # All metadata CSV files
│       ├── image_metadata.csv    # Image info + genotype mapping
│       ├── kinship_matrix.csv    # SNP-based genetic relatedness
│       ├── genotype_list.csv     # Unique genotype list
│       └── MEMA_gene_matrix.parquet  # SNP founder codes per genotype
├── models/                        # Trained weights, one folder per model
│   ├── root_detection.pt         # YOLOv8 model for root detection
│   ├── litevae/                  # litevae_epoch_N.pt, litevae_best.pt
│   ├── diffusion_onehot/         # diffusion_onehot_epoch_N.pt
│   ├── diffusion_numeric/        # diffusion_numeric_epoch_N.pt (legacy)
│   └── feature_segmentation/     # feature_segmentation_best.pt
├── results/                       # Everything any script produces
│   ├── cropped_images/           # Output cropped root images
│   ├── training/                 # Per-model previews, loss curves, logs
│   ├── figures/                  # Generated plots and visualizations
│   └── analysis/                 # Analysis results and statistics
├── README.md                      # How to use the pipeline
└── claude.md                      # This file - project structure spec
```

Checkpoints are named `<model>_epoch_<N>.pt` after the model that wrote them,
so a loose `.pt` file still identifies itself. Training writes weights to
`models/` and everything visual to `results/training/<model>/`.

`code/paths.py` is the single definition of every input and output location.
Scripts import their paths from it rather than hardcoding them, which is what
keeps outputs out of `code/` regardless of the working directory a script is
launched from.

## Data Organization Specifications

### Dataset Folder Structure

**`dataset/images/`** - All LAT Images (Flat Directory)
- All images in a single directory (NO subfolders)
- Strict naming convention: `{genotype}_{rootnode}_{replication}_{rootnumber}.JPG`
- Components:
  - `genotype`: Genotype identifier (e.g., MEMA001)
  - `rootnode`: Root type (primary, seminal, crown, etc.)
  - `replication`: Plant replicate number (1 or 2)
  - `rootnumber`: Root ID within plant (1 or 2)
- Example: `MEMA001_primary_1_1.JPG`

**`dataset/metadata/image_metadata.csv`** - Image Information & Genotype Mapping
- CSV with image metadata for all images in dataset/images/
- Required columns (11 total, in this order):
  - `image_id`: Unique numeric identifier (auto-generated)
  - `new_filename`: Standardized JPG filename
  - `original_filename`: Original filename from raw data
  - `genotype`: Genotype ID (MEMA001, etc.)
  - `rootnode`: Root type classification (primary, etc.)
  - `replication`: Plant replicate (1 or 2)
  - `rootnumber`: Root number (1 or 2)
  - `pixel_size_um`: Resolution (micrometers per pixel)
  - `image_width_px`: Image width in pixels
  - `image_height_px`: Image height in pixels
  - `quality_notes`: Quality assessment/notes
- One row per image

**`dataset/metadata/kinship_matrix.csv`** - Genetic Relatedness Data
- Square matrix with genotype IDs as row and column labels
- Values are SNP-based correlations (0-1 scale, higher = more related)
- Dimensions: N x N (where N = number of unique genotypes)
- Example: 158 x 158 for MEMA dataset (158 genotypes)

**`dataset/metadata/genotype_list.csv`** - Unique Genotypes
- Simple list of all unique genotypes in the dataset
- Single column: `genotype_id`
- Example: MEMA001, MEMA002, MEMA003, etc.
- One row per unique genotype

### Models Folder

**`models/`** - Pre-trained Models Directory
- `root_detection.pt`: YOLOv8 trained model for detecting root regions
  - Input: Full LAT image
  - Output: Bounding boxes around detected roots
  - Used by: `code/crop_root_model.py`
  - Format: PyTorch .pt file

### Results Folder

**`results/`** - All Output Files
- `figures/`: Generated plots, charts, visualizations
- `models/`: Trained or fine-tuned model outputs
- `analysis/`: Analysis results, statistics, reports
- `cropped_images/`: Root-cropped images from YOLOv8 detection
  - Naming: Same as source with `_crop_#` suffix
  - Example: `MEMA001_primary_1_1_crop_1.JPG`


## Command: `/organize-data`

**How to use:** Type this directly into the LLM chat:
/organize-data /path/to/your/data

text
The AI will guide you through organizing your raw LAT data into the standardized project structure. Just follow the conversation.

**What the AI does:**
1. Explores your data (finds images, metadata, detects naming patterns)
2. Asks clarifying questions only when needed
3. Copies images to `dataset/images/` with standardized names: `{genotype}_{rootnode}_{replication}_{rootnumber}.JPG`
4. Generates `dataset/metadata/image_metadata.csv` (image_id, new_filename, original_filename, genotype, rootnode, replication, rootnumber, pixel_size_um, image_width_px, image_height_px, quality_notes)
5. Generates `dataset/metadata/genotype_list.csv`
6. Handles kinship matrix if found
7. Provides summary and next steps


### Agent Task: Analyze and Organize User's Dataset

**Trigger:** User provides a new dataset or runs `/organize-data` command

**Your Job:**
1. **Discover and Analyze the Dataset**
   - Examine the user's raw image data folder structure
   - Find all image files (JPG, PNG, TIFF, etc.)
   - Identify metadata files (CSV, Excel, text files with genotype/image mappings)
   - Detect any pixel scale information (from filenames, scale images, documentation, metadata)
   - Analyze naming conventions and folder organization patterns

2. **Understand the Data Structure**
   - Parse image filenames to extract: genotype, root type, replication, root number, date, etc.
   - Map image files to genotypes using any available metadata
   - Determine pixel scales (um/pixel) from:
     - Embedded in filenames
     - Scale calibration files/folders
     - Metadata CSV/Excel files
     - Documentation in the dataset
     - If unavailable, ask user or estimate from scale reference images

3. **Organize Into Project Structure**
   - Create/use `dataset/images/` directory (flat, no subfolders)
   - Rename images to standard format: `{genotype}_{rootnode}_{replication}_{rootnumber}.JPG`
   - Create `dataset/metadata/` directory
   - Generate `image_metadata.csv` with columns:
     - `image_id`, `new_filename`, `original_filename`, `genotype`, `rootnode`, `replication`, `rootnumber`, `pixel_size_um`, `image_width_px`, `image_height_px`, `quality_notes`
   - Generate `genotype_list.csv` with unique genotypes
   - Handle kinship/genetic data if available, else note it's missing

4. **Handle Edge Cases**
   - Unknown/ambiguous naming conventions → ask user or make reasonable assumptions
   - Multiple possible pixel scales → ask user which applies or use best estimate
   - Missing metadata → extract from filenames or folder structure
   - Duplicate filenames → append counter suffix
   - Non-standard image formats → convert to JPG if needed

5. **Provide Summary and Confirmation**
   - Report number of images found and organized
   - List unique genotypes detected
   - Confirm pixel scales used
   - Point user to next step: `python code/crop_root_model.py`

### Key Principles

- **Flexible, Not Rigid**: Handle ANY dataset format, don't assume specific structure
- **Best-Effort Detection**: Use all available clues (filenames, folders, metadata, scale images) to understand the data
- **User Guidance**: Ask clarifying questions only when truly ambiguous
- **No Hardcoded Paths**: Work with whatever the user provides
- **Complete Automation**: End result should be a fully organized, ready-to-process dataset


## Live testing on this machine

For running and testing ideas against trained models in real time, such as
during a meeting. Never train here - training runs on Hellbender.

**Environment.** Run everything through uv, which uses the `.venv` built from
`pyproject.toml` / `uv.lock` (Python 3.11, torch 2.6 CUDA 12.4, RTX 3050 with
4 GB):

```
uv run --offline --locked --project D:/DNA_to_Display python path/to/script.py
```

`--offline --locked` means nothing is resolved or downloaded, so it works with
no network. `--project` makes it use this project's `.venv` from any working
directory, including for scripts in the scratchpad. If something seems wrong with the machine, `uv run --offline
--locked code/preflight.py` diagnoses it, and `code/check_model_toolkit.py`
exercises the whole toolkit on epochs 300 and 500.

**The toolkit.** [code/model_toolkit.py](code/model_toolkit.py) loads a root
checkpoint once and exposes what a test needs. For a one-off request, write a
short script in the scratchpad that imports it and saves output with
`output_dir(name)`, which lands in `results/playground/<name>/`:

```python
import sys; sys.path.insert(0, r'D:\DNA_to_Display\code')
from model_toolkit import load_session, measure, output_dir, save_grid
s = load_session(300)        # 300 / 500 / 25 = medium epochs, 'best', 'original', or a path
```

| Call | Returns |
|---|---|
| `s.genotypes`, `s.loci`, `s.founders`, `s.matrix` | the 200 genotypes, 43,788 loci, founders 1-8, codes |
| `s.group(g)`, `s.genotypes_in('held out')` | 'trained', 'held out' or 'never imaged' for this checkpoint |
| `s.codes(x)` | founder codes; every call below accepts a name, a list of names, or codes |
| `s.image_paths(g)`, `s.real_images(g, n)` | real cropped images, uint8 256x256 |
| `s.project(x)`, `s.embed(x)` | PCA scores (N, 168); encoder tokens (N, 8, 256) |
| `s.distance(a, b)`, `s.nearest(x, k, space)` | loci/PCA/token distances; nearest real genotypes |
| `s.founder_share(x)` | share of loci from each founder |
| `s.founder_line(f)`, `s.swap(x, loci, source)`, `s.recombinant(parents, n_blocks, seed)` | edited genotypes as codes |
| `s.generate(x, seeds, steps)` | images (genotypes, seeds, 256, 256, 3); same seed = same noise across genotypes |
| `s.denoise_loss(images, conditions, timesteps)` | image x condition denoising error, shared noise; lower = the genotype fits the image better |
| `measure(images)` | segmenter traits per image (root/stele diameter, vessel area/count, ratio) |
| `save_grid(images, path, row_labels, col_labels)` | labelled contact sheet |

**Timing, measured.** A typical request from a fresh process - load epoch 300,
edit a genotype, generate 8 images at 50 steps, measure them and save a grid -
takes about 14 s end to end: 8 s of imports and loading, 4 s generating, 1 s
measuring. Warm, 3 images take about 1 s and `denoise_loss` under 1 s. The
very first `measure` or `generate` after a reboot can take about 10 s longer
while the disk cache and CUDA warm up. A second checkpoint in the same script
loads in about 3.5 s and shares the SNP table, LiteVAE and segmenter, so put
comparisons in one script. Keep `generate` batches at 8 or fewer on 4 GB.

**Reading results honestly.**
- Held-out genotypes are held out of the diffusion model's training only; the
  PCA inside the SNP encoder was fitted on all 200.
- Locus order is the parquet's column order, sorted by gene ID, which is not
  strictly genomic position, so a "block" of loci is not a guaranteed
  haplotype block.
- `measure` uses the 512 px segmenter trained 2026-09-17. On images it never
  trained on, its errors were: vessel count 0.59, vessel area about 4%, root
  diameter 1.6 px, stele diameter 1.9 px (cross_validate_segmentation.py).
  Trait values from before that date used unfilled masks - root diameter about
  38 px too small - and are not comparable. Generated roots were not part of
  that validation.
- Locally, `'best'` is epoch 25 - the Hellbender run's best checkpoint was
  not copied here.
