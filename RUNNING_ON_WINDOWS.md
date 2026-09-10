# Running this on Windows

Written for the coding agent doing the setup. Work top to bottom. Every step
has a check; if a check fails, fix it before moving on rather than continuing
and hoping. The whole thing is four commands if nothing is wrong.

Nothing here needs converting. Checkpoints are ordinary PyTorch files and every
loader in the repo passes `map_location`, so a checkpoint written on a Mac or a
Linux box loads on Windows untouched. If you were planning a conversion step,
skip it.

---

## 1. Install uv

uv is the only thing you install by hand. It creates the environment, picks the
Python version, and installs PyTorch. There is no venv to activate and no conda.

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Check:** `uv --version` prints a version. If the shell cannot find it, open a
new terminal so the updated PATH is picked up.

---

## 2. Bring the environment up

From the repo root:

```powershell
uv run code/preflight.py
```

The first run takes a few minutes: uv downloads Python, PyTorch and the rest
into `.venv\`. Every later run starts instantly.

You do **not** need to hunt for the right PyTorch wheel. `pyproject.toml` points
Windows at PyTorch's CUDA index, so an NVIDIA box gets a CUDA build without any
`--index-url` incantation. This is deliberate: the ordinary PyPI wheel on
Windows is CPU-only, and installing it is the reason a machine with a good GPU
reports `torch.cuda.is_available() == False` and trains twenty times too slowly.

**Check:** preflight's `gpu` line. On an NVIDIA machine it must name the card:

```
  [ok  ] gpu               CUDA 12.4 - NVIDIA GeForce RTX 4090
```

If it says `CPU-only torch wheel` on a machine that has an NVIDIA card,
something overrode the index. Force it:

```powershell
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

If the machine genuinely has no NVIDIA GPU, `CPU-only` is correct and expected.
Everything still runs, roughly 6-13x slower. Carry on.

---

## 3. Put the weights and data in place

Preflight will tell you which of these are missing. There are two ways forward
and you must pick deliberately.

### If you were handed the real dataset and weights

They travel as folders, not through git - the repository deliberately contains
no `.pt` files beyond the small root detector. Copy them in so the tree looks
like this:

```
dataset\images\            the LAT scans, flat, no subfolders
dataset\metadata\          image_metadata.csv, MEMA_gene_matrix.parquet, kinship_matrix.csv
models\litevae\            litevae_epoch_N.pt
models\diffusion_onehot\   diffusion_onehot_epoch_N.pt
models\feature_segmentation\feature_segmentation_best.pt
```

<!-- WEIGHTS SOURCE: fill in where the handover folder actually lives - a
     shared drive path, a release URL, whoever to ask. Until this line is
     replaced, an agent cannot fetch the real weights unaided and should use
     the synthetic path below to verify the machine, then ask a human. -->

Then make sure `models\.synthetic` does **not** exist. That file marks the
weights beside it as fake, and while it is present every script prints a banner
saying so.

### If you have no weights yet

Mint a complete stand-in project - drawn images, random-weight checkpoints -
so the machine can be verified today:

```powershell
uv run code/make_synthetic_project.py
```

This is a real end-to-end rehearsal, not a plumbing check: the synthetic
genotypes drive vessel count and stele size through designated causal loci, so
training on it converges and the fidelity test has a right answer to find.

What it is **not** is a source of results. The weights are freshly initialised.
Anything generated is noise and any analysis run against it is measuring an
untrained network. It stamps `models\.synthetic`, which makes every script in
the repo print a banner while it exists. Delete that file only when real
weights replace these, and never commit it.

**Check:** `uv run code/preflight.py` ends with `READY.`

---

## 4. Run something

```powershell
uv run code/latent_diffusion/generation/generate_from_dataset.py
```

**Check:** images appear under `results\diffusion_results\`, and the run's
final line reports `0 failed`.

For a first pass on a CPU-only machine, open the script and set `max_images` in
its `Config` block to something small - full-dataset DDIM sampling on CPU is
about a second per image per 50 steps and the dataset is not small.

---

## Windows-specific things worth knowing

**Forcing a device.** Every script asks `paths.pick_device()`, which takes CUDA,
then Apple Metal, then CPU. Override it for a run without editing anything:

```powershell
$env:DNA_DEVICE = "cpu"; uv run code/preflight.py
```

Set it in the environment before the process starts. Several scripts read it at
import time, so setting `os.environ` from inside Python is too late.

**Dataloader workers.** Windows spawns worker processes rather than forking, so
each one re-imports the training module. The training scripts are guarded for
this and `train_segmentation.py` deliberately defaults to `workers = 0`, because
Ultralytics' default of 8 spawns processes that each re-import and can wedge.
If a training run hangs at zero progress before the first epoch, set
`num_workers = 0` in that script's `Config` and try again.

**`.gitignore` eats `*.txt`.** If you write notes, logs or a second requirements
file with a `.txt` extension, git will silently ignore it. Use `.md`, or
`git add -f`.

**Long paths.** Results nest a few levels deep. If you hit a path length error,
enable long paths once, as administrator:

```powershell
git config --system core.longpaths true
```

---

## When preflight says NOT READY

It names the reason on the line above. The three common ones:

| What it says | What to do |
|---|---|
| packages missing | `uv run code/preflight.py` again - uv installs them |
| `dataset/images  empty` | copy the handed-over data in, or run the synthetic minter |
| a checkpoint `will not load` | the file is truncated; recopy it |

If preflight is green and a script still fails, that is a real bug rather than a
setup problem. Report it with the failing command and the traceback.
