# Training setup: how big the models should be, how long to train, and the
# job scripts that train them.
#
# Sizing follows what the earlier datasets taught (code/impc/model_size_analysis.py):
#   - capacity per training image: medium (8.8M UNet) at about 3,000-5,000
#     parameters per image trained the WiDiv and RIL roots well; at 1,300 (mice)
#     it was too small. More image variety (principal components needed for 90%
#     of image variance: 26-40 for the roots, 69 for the X-rays) needs more.
#   - epochs are step-matched: the RIL's 800 epochs of 1,780 images (about 1.4
#     million images seen) for diffusion, and about 570,000 for the LiteVAE.
#   - window gates warm up and ramp over about 20% of training each.
# A second, larger size is recommended alongside when the choice is close, the
# way medium and large ran side by side for the mice.
#
# Training itself never runs here unless the dataset's configuration allows it:
# the scripts are written for a SLURM cluster (or a local GPU), and the dashboard
# follows the runs from their history files.
import math

import numpy as np
import pandas as pd
from PIL import Image

from ..project import write_json
from .base import NOTES, Stage, array, boolean, integer, obj, string

UNET = {'small': 2.7e6, 'medium': 8.8e6, 'large': 29.3e6}
DIFFUSION_IMAGES_SEEN = 1.42e6
LITEVAE_IMAGES_SEEN = 5.7e5
BASE_PPI = 3000            # parameters per training image that worked for low-variety images
REFERENCE = [  # dataset, training images, PCs for 90%, size used, outcome
    {'dataset': 'WiDiv roots', 'train_images': 2930, 'pcs90': 26, 'size': 'medium', 'outcome': 'good'},
    {'dataset': 'RIL v5 roots', 'train_images': 1780, 'pcs90': 40, 'size': 'medium', 'outcome': 'good (r 0.81)'},
    {'dataset': 'IMPC X-rays', 'train_images': 6574, 'pcs90': 69, 'size': 'large', 'outcome': 'medium too small'},
]


def image_complexity(paths, n=300):
    paths = list(paths)[:n]
    if not paths:
        return {'pcs90': None, 'detail': None}
    from scipy import ndimage
    grey = np.stack([np.asarray(Image.open(p).convert('L').resize((128, 128)), dtype=np.float32) / 255
                     for p in paths])
    flat = grey.reshape(len(grey), -1)
    flat = flat - flat.mean(0)
    s = np.linalg.svd(flat, compute_uv=False) ** 2
    cum = np.cumsum(s) / s.sum()
    detail = []
    for p in paths[:60]:
        a = np.asarray(Image.open(p).convert('L'), dtype=np.float32) / 255
        m = a < 0.96
        if m.any():
            detail.append(float(np.abs(ndimage.laplace(a))[m].mean()))
    return {'pcs90': int(np.searchsorted(cum, 0.9) + 1), 'detail': round(float(np.mean(detail)), 3) if detail else None}


class Training(Stage):
    name = 'training'
    needs = ('images', 'genetics')
    effort = 'low'

    def analyze(self, ds):
        meta = pd.read_csv(ds.scaled_metadata_path())
        split = pd.read_csv(ds.split_path)
        train_lines = set(split.line[split.split_random == 'train'].astype(str))
        meta['genotype'] = meta.genotype.astype(str)
        train = meta[meta.genotype.isin(train_lines)]
        n_train = int(round(len(train) * 0.9))          # 10% of training lines are validation
        conditions = int(round(train.genotype.nunique() * 0.9))
        cx = image_complexity(ds.scaled_dir() / f for f in train.scaled_filename.sample(
            min(300, len(train)), random_state=0))
        gpu = _local_gpu()
        return {
            'train_images': n_train, 'train_conditions': conditions,
            'images_per_condition': round(n_train / max(conditions, 1), 2),
            'image_complexity': cx,
            'params_per_image': {k: int(v / max(n_train, 1)) for k, v in UNET.items()},
            'reference_datasets': REFERENCE,
            'local_gpu': gpu,
            'encoding': (ds.config().get('genetics') or {}).get('encoding'),
        }

    def needs_agent(self, ds, report, default):
        return False, 'sizing follows the step-matching and capacity rules'

    def digest(self, ds, report):
        return report

    def schema(self, ds, report):
        return obj({
            'sizes': array(string(enum=['small', 'medium', 'large']), 'One or two sizes to train side by side.'),
            'litevae_epochs': integer(), 'diffusion_epochs': integer(),
            'window_warmup': integer(), 'window_ramp': integer(),
            'executor': string(enum=['slurm', 'local']),
            'notes': NOTES,
        })

    def default_decision(self, ds, report):
        n = max(report['train_images'], 1)
        pcs = report['image_complexity'].get('pcs90') or 40
        target = BASE_PPI * float(np.clip(pcs / 30, 0.75, 2.5))
        ppi = {k: v / n for k, v in UNET.items()}
        best = min(ppi, key=lambda k: abs(math.log(ppi[k] / target)))
        order = ['small', 'medium', 'large']
        sizes = [best]
        if ppi[best] < target and best != 'large':
            sizes.append(order[order.index(best) + 1])
        epochs = int(np.clip(round(DIFFUSION_IMAGES_SEEN / n / 10) * 10, 100, 2000))
        vae = int(np.clip(round(LITEVAE_IMAGES_SEEN / (n / 0.9) / 10) * 10, 40, 400))
        cfg = ds.config()
        executor = 'local' if cfg.get('training', {}).get('allow_local_training') else 'slurm'
        return {'sizes': sizes, 'litevae_epochs': vae, 'diffusion_epochs': epochs,
                'window_warmup': int(epochs * 0.2), 'window_ramp': int(epochs * 0.2), 'executor': executor,
                'notes': (f"{best} gives {int(ppi[best])} parameters per image against a target of "
                          f"{int(target)} for this image variety ({pcs} PCs for 90%); "
                          f"{epochs} diffusion and {vae} LiteVAE epochs match the earlier runs' steps.")}

    def prompt(self, ds, report):
        return ('Choose model size(s) and training length from the dataset size and image '
                'variety, using the reference datasets as calibration.')

    def apply(self, ds, d):
        cfg = ds.config()
        cl = cfg.get('cluster') or {}
        env = {'WIDIV_DATASET': ds.name, 'WIDIV_SPLIT': 'random',
               'WIDIV_NUM_EPOCHS': str(d['diffusion_epochs']),
               'WIDIV_LITEVAE_EPOCHS': str(d['litevae_epochs']),
               'WIDIV_WINDOW_WARMUP': str(d['window_warmup']), 'WIDIV_WINDOW_RAMP': str(d['window_ramp'])}
        env.update((cfg.get('genetics') or {}).get('env', {}))
        out = ds.stage_dir('training')
        out.mkdir(parents=True, exist_ok=True)
        exports = '\n'.join(f'export {k}={v}' for k, v in env.items())
        header = lambda job, extra='': '\n'.join([  # noqa: E731
            '#!/bin/bash',
            f'#SBATCH --job-name={job}',
            f"#SBATCH --partition={cl.get('partition', 'gpu')}",
            '#SBATCH --gres=gpu:1', '#SBATCH --cpus-per-task=8', '#SBATCH --mem=64G',
            f"#SBATCH --time={cl.get('time', '2-00:00:00')}",
            '#SBATCH --output=d2d/workspace/results/training/slurm/%x_%A_%a.out',
            (f"#SBATCH --account={cl['account']}" if cl.get('account') else '##SBATCH --account=your_account'),
            extra, 'set -euo pipefail', 'cd "$SLURM_SUBMIT_DIR"',
            f"VENV=\"${{VENV:-$SLURM_SUBMIT_DIR/{cl.get('venv', 'dna-env')}}}\"",
            'if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi',
            exports,
            'echo "d2d job $SLURM_JOB_NAME  node $(hostname)  job ${SLURM_JOB_ID:-none}"; nvidia-smi -L || true',
            # Stops before any training if an input is missing or the trainers changed.
            f'python -u -m d2d.cluster_check {ds.name}', ''])
        # The project's trainers, run unmodified through d2d's launcher (it points them
        # at the d2d workspace in memory), from the repository root.
        (out / 'train_litevae.sbatch').write_text(
            header(f'litevae_{ds.name}') + f'python -u -m d2d.launch_trainer {ds.name} litevae\n', newline='\n')
        sizes = ' '.join(d['sizes'])
        (out / 'train_diffusion.sbatch').write_text(
            header(f'diffusion_{ds.name}', f'#SBATCH --array=0-{len(d["sizes"]) - 1}')
            + f'SIZES=({sizes})\nexport DIFFUSION_SIZE=${{DIFFUSION_SIZE:-${{SIZES[${{SLURM_ARRAY_TASK_ID:-0}}]}}}}\n'
            + f'python -u -m d2d.launch_trainer {ds.name} diffusion\n', newline='\n')
        ps = ['$env:' + f'{k} = "{v}"' for k, v in env.items()]
        (out / 'run_local.ps1').write_text(
            '# Local training: only where this machine is meant to train. Run from the repository root.\n'
            + '\n'.join(ps) + f'\nuv run python -u -m d2d.launch_trainer {ds.name} litevae\n'
            + ''.join(f'$env:DIFFUSION_SIZE = "{s}"; uv run python -u -m d2d.launch_trainer {ds.name} diffusion\n'
                      for s in d['sizes']), newline='\n')
        ws = 'd2d/workspace'
        copy = [f'{ws}/dataset/{ds.name}/', f'{ws}/results/{ds.name}/scaled_512/',
                f'{ws}/results/{ds.name}/scaled_512_metadata.csv', f'{ws}/results/{ds.name}/d2d/training/']
        remote = cl.get('remote_root', '<cluster project root>')
        host = cl.get('host', '<you>@<cluster>')
        rel = f'{ws}/results/{ds.name}/d2d/training'
        (out / 'copy_to_cluster.txt').write_text(
            'On the cluster, pull the repository (for the d2d/ code), copy these into the same\n'
            'places under its root, then from that root:\n'
            '  mkdir -p d2d/workspace/results/training/slurm\n'
            f'  sbatch {rel}/train_litevae.sbatch\n'
            f'  sbatch {rel}/train_diffusion.sbatch   (after the LiteVAE)\n'
            'Training writes only inside d2d/workspace/ (models/ and results/training/).\n\n'
            + '\n'.join(f'scp -r {p} {host}:{remote}/{p}' for p in copy) + '\n', newline='\n')
        ds.update_config(training={'sizes': d['sizes'], 'litevae_epochs': d['litevae_epochs'],
                                   'diffusion_epochs': d['diffusion_epochs'], 'executor': d['executor'],
                                   'env': env},
                         pipeline={'epochs': d['diffusion_epochs'], 'litevae_epochs': d['litevae_epochs'],
                                   'window_warmup': d['window_warmup'], 'window_ramp': d['window_ramp']})
        return {'summary': f"scripts for {', '.join(d['sizes'])} written ({d['executor']})",
                'scripts': [str(p.relative_to(ds.root)) for p in out.iterdir()], 'env': env}


def _local_gpu():
    """Name and memory of this machine's GPU, from nvidia-smi: importing torch only to
    ask would load gigabytes of libraries into the dashboard server."""
    import subprocess
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        if out:
            name, mib = [s.strip() for s in out[0].rsplit(',', 1)]
            return {'name': name, 'memory_gb': round(float(mib) * 1.048576 / 1000, 1)}
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None
