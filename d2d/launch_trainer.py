# Runs the project's own trainers on a d2d dataset without changing a file of them.
#
#     python -m d2d.launch_trainer <dataset> litevae
#     python -m d2d.launch_trainer <dataset> diffusion        (DIFFUSION_SIZE picks the size)
#     python -m d2d.launch_trainer <dataset> litevae|diffusion|both --dry-run
#
# --dry-run is a smoke test light enough for the laptop: the trainer runs its whole
# setup (paths, patches, config, data, split, models, LiteVAE loading) with 0 data
# workers, then, in place of the training loop, loads one batch and passes one
# image through the models without gradients. No optimiser step, nothing written
# to the workspace: every output goes to a temporary folder that is deleted after.
# The diffusion dry run encodes images on the fly instead of building the latent
# cache, and uses a random-weight stand-in LiteVAE when none is trained yet.
#
# The trainers (code/widiv/train_litevae_widiv.py, train_widiv.py) take every
# input and output location from two modules, paths.py and widiv_paths.py. This
# launcher, in memory only:
#   1. imports the real paths.py and points its DATASET_DIR, RESULTS_DIR,
#      MODELS_DIR and TRAINING_RESULTS_DIR at the d2d workspace;
#   2. registers a stand-in widiv_paths module built from the dataset's d2d.json,
#      with every name the widiv modules import;
#   3. reads the trainer's source, swaps the one dictionary lookup that knows only
#      the built-in dataset names for one that falls back to d2d's defaults, and
#      runs it as __main__.
# The trainer files on disk are never written. If a trainer changes so that a
# patch no longer applies, the launcher stops and says which one.
import os
import re
import sys
import types
from pathlib import Path

from .project import PROJECT_ROOT, ROOT, Dataset

CODE = PROJECT_ROOT / 'code'
TRAINERS = {
    'litevae': CODE / 'widiv' / 'train_litevae_widiv.py',
    'diffusion': CODE / 'widiv' / 'train_widiv.py',
}
# (pattern, replacement) applied to the trainer source in memory.
PATCHES = {
    'diffusion': [(r"\n\}\[WIDIV_DATASET\]\n",
                   "\n}.get(WIDIV_DATASET, __import__('widiv_paths').D2D_TRAIN_DEFAULTS)\n")],
    'litevae': [(r"(num_epochs = \{[^}]*\})\[WIDIV_DATASET\]",
                 r"\1.get(WIDIV_DATASET, __import__('widiv_paths').D2D_LITEVAE_EPOCHS)"),
                # Batch 8 at 512 px fills a 32 GB GPU (29 GB allocated); D2D_LITEVAE_BATCH
                # (set by litevae_batch below) lets a smaller GPU run.
                (r"batch_size = 8\n",
                 "batch_size = int(__import__('os').environ.get('D2D_LITEVAE_BATCH', 8))\n")],
}


# Further patches for --dry-run only: 0 workers, and the training loop replaced by
# _d2d_dry(locals()), which the launcher puts in the trainer's globals.
_LOOP = {'litevae': 'for epoch in range(start_epoch, cfg.num_epochs + 1):',
         'diffusion': 'for epoch in range(start_epoch, cfg.num_epochs):'}
DRY_PATCHES = {
    'litevae': [(r"\n        num_workers = 8\n", "\n        num_workers = 0\n")],
    'diffusion': [(r"\nif __name__ == '__main__':\n    main\(\)\n",
                   "\nif __name__ == '__main__':\n"
                   "    main({'cache_latents': False, 'num_workers': 0, 'batch_size': 2})\n")],
}
for _w, _loop in _LOOP.items():
    DRY_PATCHES[_w].append((r"\n    " + re.escape(_loop) + r"\n",
                            "\n    _d2d_dry(locals()); return\n    " + _loop + "\n"))


def _finite(*xs):
    import torch
    return all(bool(torch.isfinite(x).all()) for x in xs)


def _gpu_peak():
    import torch
    return f', peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB' if torch.cuda.is_available() else ''


def dry_litevae(ns):
    """One batch through the LiteVAE without gradients; saves the random-weight models as a stand-in."""
    import torch
    cfg, enc, dec, dev = ns['cfg'], ns['encoder'], ns['decoder'], ns['device']
    images = next(iter(ns['train_loader']))
    print(f'dry run: a training batch {tuple(images.shape)}, values {images.min():.2f} to {images.max():.2f}; '
          f'on this GPU a real run would use batch {cfg.batch_size}, {len(ns["train_loader"])} batches per epoch')
    enc.eval(); dec.eval()
    with torch.no_grad():
        x = images[:1].to(dev)
        z, mean, logvar = enc(x)
        recon = dec(z, save_steps=False)
        loss, recon_loss, kl, mse = ns['criterion'](recon, x, mean, logvar)
    ok = recon.shape == x.shape and _finite(z, recon, loss)
    print(f'dry run: image {tuple(x.shape)} -> latent {tuple(z.shape)} -> reconstruction {tuple(recon.shape)}, '
          f'loss {loss.item():.4f}{_gpu_peak()}')
    stand_in = sys.modules['widiv_paths'].D2D_DRY_LITEVAE
    stand_in.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'encoder_state_dict': enc.state_dict(), 'decoder_state_dict': dec.state_dict(),
                'config': ns['model_config'], 'image_size': cfg.image_size, 'epoch': 0}, stand_in)
    if not ok:
        raise SystemExit('dry run FAILED: shapes do not match or values are not finite')
    print('dry run: LiteVAE OK')


def dry_diffusion(ns):
    """One training batch through the SNP encoder and UNet without gradients."""
    import torch
    cfg = ns['cfg']
    z, snp = next(ns['training_batches']())
    with torch.no_grad():
        loss = ns['denoising_loss'](z, snp)
    print(f'dry run: latents {tuple(z.shape)}, genotype input {tuple(snp.shape)}, denoising loss {loss.item():.4f}'
          f'{_gpu_peak()}; {len(ns["train_samples"])} training images, run {ns["run_name"]}')
    if not _finite(z, loss):
        raise SystemExit('dry run FAILED: values are not finite')
    print(f'dry run: diffusion OK (the real run uses batch 16 and the latent cache; '
          f'{cfg.num_epochs} epochs)')


DRY_HOOKS = {'litevae': dry_litevae, 'diffusion': dry_diffusion}


def litevae_batch():
    """Batch 8 (the trainer's own) on a 32 GB GPU or larger, else the largest that fits.
    On 32 GB it ran 22 epochs, then hit a fragmentation OOM (2.4 GB reserved but unusable);
    expandable_segments, set in run(), is what fixes that."""
    if os.environ.get('D2D_LITEVAE_BATCH'):
        return
    import torch
    if not torch.cuda.is_available():
        return
    gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    batch = 8 if gb >= 30 else 4 if gb >= 16 else 2
    if batch != 8:
        os.environ['D2D_LITEVAE_BATCH'] = str(batch)
        print(f'd2d: {torch.cuda.get_device_name(0)} has {gb:.0f} GB; LiteVAE batch {batch} instead of 8 '
              '(batch 8 at 512 px needs a 32 GB GPU). Set D2D_LITEVAE_BATCH to override.', flush=True)


def widiv_paths_module(ds):
    cfg = ds.config()
    pipe = cfg.get('pipeline', {})
    m = types.ModuleType('widiv_paths')
    d = ds.dataset_dir
    r = ds.results_dir
    size = pipe.get('image_size', 512)
    qtl = (cfg.get('genetics') or {}).get('env', {}).get('WIDIV_QTL_TABLE')
    values = dict(
        WIDIV_DATASET=ds.name, RIL=False, IMPC=False,
        WIDIV_UPRIGHT=bool(pipe.get('upright', False)),
        WIDIV_DIR=d, WIDIV_IMAGES_DIR=d / 'images', WIDIV_MASKS_DIR=d / 'masks',
        WIDIV_PIXEL_SIZE_UM=pipe.get('pixel_size_um') or 0.0,
        WIDIV_GENETIC_DIR=ds.genetic_dir, WIDIV_GENOTYPES=ds.genotypes_path, WIDIV_SNP_MAP=ds.snp_map_path,
        WIDIV_METADATA_DIR=ds.metadata_dir, WIDIV_SPLIT=ds.split_path,
        WIDIV_SPLITS=tuple(pipe.get('splits', ['split_random'])),
        WIDIV_GENETIC_VALUES=None,
        WIDIV_QTL=(ROOT / qtl) if qtl else None,
        WIDIV_PLOT_PHENOTYPES=ds.traits_path if ds.traits_path.exists() else None,
        WIDIV_GENOTYPE_MEANS=ds.traits_path if ds.traits_path.exists() else None,
        WIDIV_IMAGE_METADATA=ds.image_index_path, WIDIV_GENOTYPE_LIST=ds.metadata_dir / 'genotype_list.csv',
        WIDIV_KINSHIP_MATRIX=ds.metadata_dir / 'kinship_matrix.csv', WIDIV_PLOT_TABLE=ds.image_index_path,
        WIDIV_RESULTS_DIR=r, WIDIV_CANVAS_PX=(cfg.get('images') or {}).get('canvas_px', size),
        WIDIV_IMAGE_SIZE=size, WIDIV_SCALED_DIR=r / f'scaled_{size}',
        WIDIV_SCALED_MASKS_DIR=r / f'scaled_{size}_masks', WIDIV_SCALED_METADATA=r / f'scaled_{size}_metadata.csv',
        WIDIV_GENOTYPE_CACHE=ROOT / 'results' / 'cache' / f'{ds.name}_genotypes.npz',
        WIDIV_LATENT_CACHE_DIR=ROOT / 'results' / 'cache' / f'{ds.name}_latents',
        WIDIV_LITEVAE_DIR=ROOT / 'models' / f'litevae_{ds.name}_v3',
        WIDIV_LITEVAE_MODEL=ROOT / 'models' / f'litevae_{ds.name}_v3' / f'litevae_{ds.name}_v3_best.pt',
        WIDIV_LITEVAE_V2_MODEL=ROOT / 'models' / 'litevae_none_v2' / 'missing.pt',
        D2D_TRAIN_DEFAULTS={'epochs': int(pipe.get('epochs', 400)),
                            'window_warmup': int(pipe.get('window_warmup', 80)),
                            'window_ramp': int(pipe.get('window_ramp', 80)),
                            'pca_variance': float(pipe.get('pca_variance', 0.95))},
        D2D_LITEVAE_EPOCHS=int(pipe.get('litevae_epochs', 250)),
    )
    for k, v in values.items():
        setattr(m, k, v)
    return m


def prepare(ds, dry=None):
    """Point the project's path modules at the workspace (or, for a dry run, every
    output at the folder dry). Returns nothing; sys.modules is set."""
    for p in (CODE, CODE / 'widiv'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import paths  # the real module, then redirected
    paths.DATASET_DIR = ROOT / 'dataset'
    paths.RESULTS_DIR = ROOT / 'results'
    paths.MODELS_DIR = ROOT / 'models'
    paths.TRAINING_RESULTS_DIR = ROOT / 'results' / 'training'
    W = widiv_paths_module(ds)
    if dry is not None:
        paths.MODELS_DIR = dry / 'models'
        paths.TRAINING_RESULTS_DIR = dry / 'training'
        W.D2D_DRY_LITEVAE = dry / 'models' / W.WIDIV_LITEVAE_DIR.name / W.WIDIV_LITEVAE_MODEL.name
        if not W.WIDIV_LITEVAE_MODEL.exists():
            W.WIDIV_LITEVAE_MODEL = W.D2D_DRY_LITEVAE
        W.WIDIV_LITEVAE_DIR = W.D2D_DRY_LITEVAE.parent
        W.WIDIV_GENOTYPE_CACHE = dry / 'cache' / W.WIDIV_GENOTYPE_CACHE.name
        W.WIDIV_LATENT_CACHE_DIR = dry / 'cache' / W.WIDIV_LATENT_CACHE_DIR.name
    sys.modules['widiv_paths'] = W


def run(name, which, dry=None):
    # Before torch is imported: lets freed blocks be reused, avoiding fragmentation OOMs.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    ds = Dataset(name)
    if not ds.config_path.exists():
        raise SystemExit(f'no d2d dataset {name!r} in {ROOT}')
    path = TRAINERS[which]
    source = path.read_text(encoding='utf-8')
    for pattern, repl in PATCHES[which] + (DRY_PATCHES[which] if dry is not None else []):
        source, n = re.subn(pattern, repl, source)
        if n != 1:
            raise SystemExit(f'{path.name} has changed: the launcher patch {pattern!r} matched {n} times. '
                             'Update PATCHES in d2d/launch_trainer.py.')
    os.environ['WIDIV_DATASET'] = name
    for k, v in (ds.config().get('training') or {}).get('env', {}).items():
        os.environ.setdefault(k, str(v))
    # Workspace-relative table paths become absolute on whatever machine this runs.
    q = os.environ.get('WIDIV_QTL_TABLE')
    if q and not Path(q).is_absolute():
        os.environ['WIDIV_QTL_TABLE'] = str(ROOT / q)
    prepare(ds, dry)
    if which == 'litevae':
        litevae_batch()
    sys.argv = [str(path)]
    print(f"d2d: {'DRY RUN of' if dry is not None else 'training'} {which} for {name} in {ROOT} "
          f"(trainer {path.relative_to(PROJECT_ROOT)}, unmodified on disk)", flush=True)
    code = compile(source, str(path), 'exec')
    g = {'__name__': '__main__', '__file__': str(path)}
    if dry is not None:
        g['_d2d_dry'] = DRY_HOOKS[which]
    exec(code, g)


def dry_run(name, which):
    """litevae, diffusion or both, against a temporary output folder deleted afterwards.
    A diffusion dry run with no trained LiteVAE first runs the LiteVAE one for its stand-in."""
    import shutil
    import tempfile
    dry = Path(tempfile.mkdtemp(prefix='d2d_dryrun_'))
    try:
        steps = ['litevae', 'diffusion'] if which == 'both' else [which]
        if steps == ['diffusion'] and not widiv_paths_module(Dataset(name)).WIDIV_LITEVAE_MODEL.exists():
            print('d2d: no trained LiteVAE yet; the LiteVAE dry run makes a random-weight stand-in first')
            steps = ['litevae', 'diffusion']
        for w in steps:
            run(name, w, dry)
        print(f"\ndry run passed: {', '.join(steps)} (temporary outputs in {dry} removed)")
    finally:
        shutil.rmtree(dry, ignore_errors=True)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--dry-run']
    dry_flag = len(args) < len(sys.argv) - 1
    if len(args) != 2 or args[1] not in (*TRAINERS, *(('both',) if dry_flag else ())):
        raise SystemExit('usage: python -m d2d.launch_trainer <dataset> litevae|diffusion\n'
                         '       python -m d2d.launch_trainer <dataset> litevae|diffusion|both --dry-run')
    if dry_flag:
        dry_run(*args)
    else:
        run(*args)
