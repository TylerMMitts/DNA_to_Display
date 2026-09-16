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

The UNet and SNP encoder train together from scratch. Weights to
`models/diffusion_onehot_<size>/`, preview images and loss history to
`results/training/diffusion_onehot_<size>/`. This needs a trained LiteVAE,
which it loads frozen and never updates.

`model_size` in the config picks `small` (2.7M UNet), `medium` (8.8M, the
default) or `large` (29M). The original model in `models/diffusion_onehot/`
was 209M parameters and memorised its ~700 training images, so these are
sized for the data instead. To train on Hellbender, one size per GPU:

```bash
uv sync
mkdir -p results/training/slurm
sbatch code/latent_diffusion/training/train_onehot_hellbender.sbatch
```

Every epoch prints a *genotype gain*: how much worse the model denoises when
handed the wrong genotype. Held-out gain above zero means it learned something
about genotypes that carries over to ones it never saw; a trained gain far
above the held-out one means it is recalling training images instead. Pick the
size with the best held-out gain and validation loss, then point
`DIFFUSION_ONEHOT_MODEL` in `code/paths.py` at its `_best.pt`.

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

## The seed dataset

A second dataset of seed kernel plates, about 500 genotypes with one image each.
Same eight founders and the same one-hot treatment as the roots, but a different
locus set, so it gets its own models throughout and never shares weights with
the root pipeline.

Each dataset has its own training scripts rather than one script with a switch,
so which dataset you are training is decided by which file you run.

**1. Crop the plates.**

```bash
python code/crop_seed_scans.py
```

Reads `dataset/seed_scans/`, writes `results/seeds/cropped_images/` and
`seed_image_metadata.csv`. The plates carry a genotype name above the kernel and
a scale bar below it; both are cropped away, but the bar is measured first. It
has to be: every plate was rendered to a fixed kernel height, so the bar is the
only record of how large the kernel really is.

**2. Put every kernel at one scale.**

```bash
python code/rescale_seed_crops.py
```

Resizes each crop by its own bar length and pads it onto a 256x256 canvas, so
kernels differ in pixels exactly as much as they differ in life - a 2x range
that is invisible in the plates as rendered. Padding, not resizing to fit:
resizing each image to the canvas would make every kernel the same size again.

**3. Train the seed autoencoder.**

```bash
python code/litevae/train_litevae_seeds.py
```

Weights to `models/litevae_seeds/`, reconstructions and loss curve to
`results/training/litevae_seeds/`. The root LiteVAE is not a substitute: it was
fitted to photographed cross-sections, and the diffusion model trains inside
whatever latent space the autoencoder provides.

**4. Train the seed diffusion model.**

```bash
python code/latent_diffusion/training/train_seeds.py
```

Weights to `models/diffusion_seeds_<size>/`, previews and loss history to
`results/training/diffusion_seeds_<size>/`. Needs step 3 finished first. Sizes
and the genotype-gain readout work exactly as they do for the root trainer.

On Hellbender, autoencoder first and then the diffusion model:

```bash
sbatch code/litevae/train_litevae_seeds_hellbender.sbatch
sbatch code/latent_diffusion/training/train_seeds_hellbender.sbatch
```

Augmentation differs from the root trainers on purpose. Kernels have a real top
and bottom and every plate is drawn the same way up, so vertical flips and
rotations are off. Colour jitter is off too: the plates encode kernel colour as
horizontal bands, making colour a trait being predicted rather than a nuisance
to be robust to. Horizontal flips are all that remain.

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
| `generate_held_out_seeds.py` | Seven generated images beside every root whose genotype the model never trained on |

**Understanding what the model learned** - `code/latent_diffusion/analysis/`

| Script | Question it answers |
|---|---|
| `snp_output_contribution.py` | Where in the final 256x256 image does one SNP change the picture, and which tissue does it act on? |
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

## Analysing a trained model

Every analysis above was first run on the original one-hot model. To run all of
them on another root diffusion model, set `checkpoint` in the config of
[code/evaluate_diffusion_model.py](code/evaluate_diffusion_model.py) and run:

```bash
python code/evaluate_diffusion_model.py
```

Everything lands in `results/model_analysis/<checkpoint name>/`, one subfolder
per analysis, with `model_info.json` recording what was analysed and
`pipeline_log.csv` recording how each step went. A `_best.pt` checkpoint gets
its epoch added to the folder name, because that file is overwritten as
training improves.

| Step | Script | What it answers |
|---|---|---|
| `memorization` | `analysis/analyze_genotype_memorization.py` | Does the right genotype help on held-out plants, or only on trained ones? |
| `snp_encoder` | `validation/test_snp_encoder.py` | Does this model's SNP encoder keep genotypes apart? |
| `genetic_fidelity` | `evaluation/genetic_fidelity_test.py` | Do generated traits track real ones, measured by the segmenter? |
| `train_vs_test` | `evaluation/train_vs_test_accuracy.py` | The same, split by trained and held-out genotypes |
| `latent_comparison` | `evaluation/latent_comparison.py` | How far generated latents sit from real ones |
| `attention` | `analysis/analyze_snp_attention.py` | Which SNP tokens are attended to, and where |
| `genotype_contribution` | `analysis/analyze_genotype_contribution.py` | How much the genotype moves the output |
| `snp_spatial_contribution` | `analysis/analyze_snp_spatial_contribution.py` | Where named loci act |
| `snp_ranking` | `analysis/rank_snp_contributions.py` | Strongest SNPs, genome-wide |
| `snp_diverse_maps` | `analysis/select_diverse_snp_maps.py` | SNPs with the most different spatial effects |
| `snp_output_contribution` | `analysis/snp_output_contribution.py` | Those SNPs' effects in the final image, by tissue |
| `parent_archetypes` | `generation/generate_parent_archetypes.py` | The eight founders as this model draws them |
| `founder_strategies` | `generation/founder_archetype_strategies.py` | Ways of building a founder genotype, compared |
| `held_out_seeds` | `generation/generate_held_out_seeds.py` | Every held-out root beside seven generated images of its genotype |
| `gallery` (off) | `generation/generate_from_dataset.py` | Real and generated side by side, for browsing |

Rerunning for the same model skips every step whose folder already exists, so
after a new step is added only that step runs. A step the last run logged as
failed, or left unfinished because the job was killed, is run again, and so is
any step that reads the output of one run this time. To redo a finished step,
for example after changing its settings, name it in `rerun`.

Comment a step out of `steps` to skip it, and use `step_settings` to change any
one script's settings for a quicker pass, for example
`{'genetic_fidelity': {'max_images': 40}}`. Every setting is checked against its
script before anything runs, so a misspelt name fails in seconds rather than
hours into a job. Each step runs in its own process, so a crash is logged and
the remaining steps carry on; a step that reads a failed step's output is
skipped.

`train_vs_test` also reports the LiteVAE ceiling from
`results/reconstruction_fidelity/` (`evaluation/reconstruction_fidelity_test.py`).
That measures the autoencoder alone, so it is run once and shared by every model
rather than repeated per model. Without it the step still runs and skips that
section.

On Hellbender, after editing `checkpoint`:

```bash
mkdir -p results/model_analysis/slurm
sbatch code/evaluate_diffusion_model_hellbender.sbatch
```

This is for root models only. The seed models use a different locus set and
autoencoder, which these scripts do not load. Two analyses are left out on
purpose: `test_conditioning_strength.py` and `compare_genotype_contributions.py`
compare one-hot against numeric encoding, so they are about two models at once
rather than one.

Each script also still runs on its own exactly as before; the pipeline only
redirects its checkpoint, output folder and caches.

## Notes on running this

Scripts pick CUDA automatically when it is available and fall back to CPU.
Diffusion sampling on CPU is slow enough that you should set `max_images` to
something small for a first pass.

`train_segmentation.py` defaults to CPU with `workers = 0` on purpose. On
Windows, Ultralytics' default of 8 dataloader workers spawns processes that each
re-import torch, which exhausts the system commit limit and fails with
"The paging file is too small". The comments in that config explain the rest.
