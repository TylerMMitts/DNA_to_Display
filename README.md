# DNA to Display

Generates maize root cross-section images from genotype. You give it SNP data
for a plant, it produces the root image that genotype is predicted to grow.

The pipeline is four trained models chained together:

| Model | What it does |
|---|---|
| `root_detection` | Finds and crops the root out of a raw laser-ablation scan |
| `litevae` | Compresses a 256x256 root image to a 4x32x32 latent, and back |
| `diffusion_onehot` | Generates a latent from a genotype, using cross-attention on the SNPs |
| `feature_segmentation` | Labels root, stele and vessels so traits can be measured |

Generation runs `diffusion_onehot` to make a latent, then the `litevae` decoder
to turn that latent into a picture. `feature_segmentation` is how generated
images get scored against real ones.

## Setup

```bash
pip install -r requirements.txt
```

Then put the weights in place. If you were given a folder of weights, it should
end up looking exactly like this:

```
models/
├── root_detection.pt
├── litevae/
│   └── litevae_epoch_250.pt
├── diffusion_onehot/
│   └── diffusion_onehot_epoch_100.pt
├── diffusion_numeric/
│   └── diffusion_numeric_epoch_500.pt
└── feature_segmentation/
    └── feature_segmentation_best.pt
```

Every checkpoint is named after the model that produced it, so a loose `.pt`
file still tells you what it is. Files are `<model>_epoch_<N>.pt`, plus
`<model>_best.pt` for the lowest validation loss of a run.

Anything that picks "the latest checkpoint" also accepts the older
`checkpoint_epoch_<N>.pt` name, so weights copied straight off a training
cluster work without being renamed first. It compares epoch numbers
numerically, so epoch 100 wins over epoch 20.

To point at a specific checkpoint instead, edit the five `*_MODEL` lines in
[code/paths.py](code/paths.py). Nothing else needs changing: every script reads
its weights through that one file.

## Where everything lives

```
dataset/          inputs you provide - images and metadata
models/           trained weights, one folder per model
results/          everything any script produces
code/             source only, never written to
```

Nothing is ever written next to the script that produced it. Paths are resolved
against the project root, so you can run a script from any working directory
and it reads and writes the same places.

## Running the pipeline

Steps 1 and 2 are needed once, to get from raw scans to training data. If you
were handed weights and just want to generate images, skip to step 5.

**1. Organize your raw data.** Inside Claude Code:

```
/organize-data /path/to/your/raw/data
```

This fills `dataset/images/` with standardized filenames
(`{genotype}_{rootnode}_{replication}_{rootnumber}.JPG`) and writes the
metadata CSVs. See [claude.md](claude.md) for the exact schema.

**2. Crop the roots out of the scans.**

```bash
python code/crop_root_model.py
```

Reads `dataset/images/`, writes `results/cropped_images/`. Everything
downstream trains on the cropped images, not the raw scans.

**3. Train the autoencoder.**

```bash
python code/litevae/train_litevae.py
```

Weights to `models/litevae/`, reconstructions and loss curves to
`results/training/litevae/`. Resumes automatically from the newest checkpoint
in the weights folder.

**4. Train the diffusion model.**

```bash
python code/latent_diffusion/training/train_onehot.py
```

Weights to `models/diffusion_onehot/`, preview images and loss history to
`results/training/diffusion_onehot/`. This needs a trained LiteVAE, which it
loads frozen and never updates.

`training/train.py` is the older version that fed SNP founder codes to the
network as plain numbers 1-8. That treats founder 8 as "eight times founder 1",
which is meaningless for what are really just category labels.
`train_onehot.py` one-hot encodes them instead. Keep `train.py` only for
reproducing the old checkpoints.

**5. Generate images from genotypes.**

```bash
python code/latent_diffusion/generation/generate_from_dataset.py
```

Generates one image per row of the metadata and saves it beside the real
photograph for comparison, in `results/diffusion_results/`. Safe to interrupt
and rerun - it skips images that already exist.

To see what the eight founder parents are predicted to look like:

```bash
python code/latent_diffusion/generation/generate_parent_archetypes.py
```

**6. Train the trait segmenter** (only needed if you want to measure traits and
have no `feature_segmentation` weights).

```bash
python code/feature_segmentation/train_segmentation.py
```

It prepares the annotated dataset if that has not been done, trains, then
copies the best weights to `models/feature_segmentation/` and validates them.
Plots and logs go to `results/training/feature_segmentation/`.

## Changing settings

Every runnable script keeps its settings in one `class cfg` block at the top of
`main()`. Edit the values there and rerun - there are no command line flags.

```python
def main():
    # Edit these values, then run:
    #     python code/latent_diffusion/training/train_onehot.py
    class cfg:
        save_dir = DIFFUSION_ONEHOT_DIR
        num_epochs = 150
        batch_size = 16
```

Paths in those blocks come from [code/paths.py](code/paths.py), which is the
single place any file location is defined. Change a path there and every script
follows.

## Scripts that need another script run first

Most scripts only need the weights and the dataset. These five read a folder
that another script writes, so they fail on a fresh checkout until you run the
producer first:

| Run this | only after this |
|---|---|
| `evaluation/train_vs_test_accuracy.py` | `evaluation/genetic_fidelity_test.py` and `evaluation/reconstruction_fidelity_test.py` |
| `evaluation/latent_comparison.py` | `evaluation/genetic_fidelity_test.py` |
| `analysis/diagnose_founder_similarity.py` | `generation/founder_archetype_strategies.py` |
| `analysis/compare_conditioning_grids.py` | `analysis/test_conditioning_strength.py` |
| `analysis/select_diverse_snp_maps.py` | nothing, but it repeats the genome sweep from `analysis/rank_snp_contributions.py`, so run that first if you want both |

## What each script is for

**Building the models** - `code/`

| Script | Purpose |
|---|---|
| `crop_root_model.py` | Crops roots out of raw scans with the YOLOv8 detector |
| `litevae/train_litevae.py` | Trains the image autoencoder |
| `latent_diffusion/training/train_onehot.py` | Trains the genotype-conditioned diffusion model |
| `latent_diffusion/training/train.py` | The older numeric-SNP trainer, kept for reproducibility |
| `feature_segmentation/prepare_dataset.py` | Turns annotations into a 256x256 YOLO dataset |
| `feature_segmentation/train_segmentation.py` | Trains the root/stele/vessel segmenter |

**Generating images** - `code/latent_diffusion/generation/`

| Script | Purpose |
|---|---|
| `generate_from_dataset.py` | One generated image per real image, side by side |
| `generate_parent_archetypes.py` | What each of the eight founder parents should look like |
| `founder_archetype_strategies.py` | Compares ways of building a founder genotype to generate from |

**Understanding what the model learned** - `code/latent_diffusion/analysis/`

| Script | Question it answers |
|---|---|
| `analyze_snp_attention.py` | Which SNP tokens does the UNet attend to, and where? |
| `rank_snp_contributions.py` | Which individual SNPs change the image most, swept genome-wide |
| `select_diverse_snp_maps.py` | Which SNPs have the most *different* spatial effects, not just the strongest |
| `analyze_snp_spatial_contribution.py` | Where in the image does one named locus act? |
| `analyze_genotype_contribution.py` | How much does the genotype move the output at all? |
| `analyze_pca_sensitivity.py` | Which PCA components of the SNP vector matter? |
| `compare_genotype_contributions.py` | Do genotypes separate better under one-hot than numeric encoding? |
| `test_conditioning_strength.py` | Does changing the genotype change the image more than changing the seed? |
| `compare_conditioning_grids.py` | Side-by-side grids of the above, by genotype |
| `diagnose_founder_similarity.py` | Founder images differ numerically but look alike - why? |

**Checking the models are right** - `code/latent_diffusion/validation/` and
`code/feature_segmentation/evaluation/`

| Script | Question it answers |
|---|---|
| `validation/validate_founder_encoding.py` | Does numeric founder coding distort genetic similarity? |
| `validation/validate_onehot_encoding.py` | Does the one-hot encoding preserve it? |
| `evaluation/genetic_fidelity_test.py` | Do generated traits track the real traits for that genotype? |
| `evaluation/train_vs_test_accuracy.py` | Is it better on genotypes it trained on than unseen ones? |
| `evaluation/reconstruction_fidelity_test.py` | Does LiteVAE preserve traits through encode and decode? |
| `evaluation/validate_vessel_counting.py` | Which vessel-counting method matches hand counts? |
| `evaluation/vessel_interpolation_test.py` | Is vessel count a smooth axis in the latent space? |
| `evaluation/latent_average_test.py` | What does averaging two latents produce? |
| `evaluation/latent_comparison.py` | How do latents compare across genotypes? |

`genetic_fidelity_test.py` is the one that matters most: it is the end-to-end
check on whether genotype actually predicts phenotype here, rather than the
model producing plausible roots that ignore their conditioning.

## Notes on running this

Scripts pick CUDA automatically when it is available and fall back to CPU.
Diffusion sampling on CPU is slow enough that you should set `max_images` to
something small for a first pass.

`train_segmentation.py` defaults to CPU with `workers = 0` on purpose. On
Windows, Ultralytics' default of 8 dataloader workers spawns processes that each
re-import torch, which exhausts the system commit limit and fails with
"The paging file is too small". The comments in that config explain the rest.
