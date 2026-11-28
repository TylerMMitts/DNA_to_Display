# Training elsewhere: a self-contained bundle to train on another machine, and the
# import of its results.
#
# The bundle (results/<dataset>/d2d/bundle/d2d_train_<dataset>.zip) unpacks into one
# folder that holds everything training needs and nothing else:
#   code/                 the project's trainers and models (read only, as here)
#   d2d/                  the app's code (launcher, checks, training runner)
#   d2d/workspace/        this dataset's genotypes, split, prepared images and settings
#   pyproject.toml, uv.lock   the Python environment
#   RUN_TRAINING.md       directions: requirements, setup, check, train, bring back
# On the other machine `python -m d2d.train <dataset>` trains and finally writes
# d2d_results_<dataset>.zip, which Import results (dashboard) or import_results() unpacks
# here. Only this dataset's run folders are accepted from it.
import shutil
import threading
import time
import zipfile
from pathlib import Path, PurePosixPath

from . import system
from .project import PROJECT_ROOT, ROOT, read_json, write_json
from .train import run_dirs

_threads = {}
STORED = {'.png', '.jpg', '.jpeg', '.webp', '.gz', '.zip', '.pt', '.npz', '.parquet'}


def bundle_dir(ds):
    return ds.stage_dir('training').parent / 'bundle'


def status(ds):
    s = read_json(bundle_dir(ds) / 'status.json', {'state': 'none'})
    t = _threads.get(ds.name)
    if s.get('state') == 'building' and not (t and t.is_alive()):
        s = {'state': 'failed', 'message': 'the build stopped (the dashboard restarted?); build it again'}
    return s


def _files(ds):
    """(source path, path inside the bundle) for everything training needs."""
    out = []

    def tree(src, arc, skip=()):
        for p in sorted(Path(src).rglob('*')):
            rel = p.relative_to(src)
            if p.is_file() and '__pycache__' not in rel.parts and not any(rel.parts[0] == s for s in skip) \
                    and p.suffix != '.pyc':
                out.append((p, f'{arc}/{rel.as_posix()}'))

    tree(PROJECT_ROOT / 'code', 'code', skip=('session_scripts',))
    tree(PROJECT_ROOT / 'd2d', 'd2d', skip=('workspace', '.claude'))
    for f in ('pyproject.toml', 'uv.lock', '.python-version'):
        if (PROJECT_ROOT / f).exists():
            out.append((PROJECT_ROOT / f, f))
    ws = 'd2d/workspace'
    tree(ds.dataset_dir, f'{ws}/dataset/{ds.name}')
    size = (ds.config().get('pipeline') or {}).get('image_size', 512)
    r = ds.results_dir
    for d in (f'scaled_{size}', f'scaled_{size}_masks', 'd2d/training'):
        if (r / d).exists():
            tree(r / d, f'{ws}/results/{ds.name}/{d}')
    if (r / f'scaled_{size}_metadata.csv').exists():
        out.append((r / f'scaled_{size}_metadata.csv', f'{ws}/results/{ds.name}/scaled_{size}_metadata.csv'))
    return out


def directions(ds, data_gb=None):
    cfg = ds.config()
    t = cfg.get('training') or {}
    env = t.get('env') or {}
    sizes = t.get('sizes') or ['medium']
    n = ds.name
    def amount(v, unit):
        return v if isinstance(v, str) else f'{v:g} {unit}'
    labels = {'gpu_memory_gb': 'GPU memory', 'ram_gb': 'System memory', 'cpu_cores': 'CPU cores',
              'disk_free_gb': 'Free disk', 'power': 'Power and cooling'}
    req = '\n'.join(f"| {labels.get(r['key'], r['key'])} | {amount(r['minimum'], r['unit'])} | "
                    f"{amount(r['recommended'], r['unit'])} | {r['why']} |"
                    for r in system.check(int((data_gb or 0) * 1e9))['rows'])
    return f"""# Training {n} on another machine

This folder is everything needed to train the DNA to Display models for **{n}**:
the trainers (`code/`), the app's runner (`d2d/`), this dataset's prepared images,
genotypes and settings (`d2d/workspace/`), and the Python environment
(`pyproject.toml`, `uv.lock`). Nothing outside this folder is read or written.

Settings chosen by the pipeline: diffusion size {', '.join(sizes)}, {env.get('WIDIV_NUM_EPOCHS', '?')} diffusion
epochs, {env.get('WIDIV_LITEVAE_EPOCHS', '?')} LiteVAE epochs, encoding `{env.get('WIDIV_SNP_ENCODING', 'default')}`.

## 1. What the machine needs

| | Minimum | Recommended | Why |
|---|---|---|---|
{req}

Linux or Windows with an NVIDIA GPU and its driver (CUDA 12). Run
`python -m d2d.train {n} check` to compare a machine with this table before training.

## 2. Set up Python (once)

With [uv](https://docs.astral.sh/uv/) (fastest; installs Python 3.11 and the exact versions tested):

    uv sync
    source .venv/bin/activate          # Windows: .venv\\Scripts\\activate

Or an existing environment with Python 3.11+ and PyTorch with CUDA: install the
packages listed in `pyproject.toml` (`pip install -e .` from this folder).

## 3. Check, then train

From this folder:

    python -m d2d.train {n} check       # the machine and the data (a minute)
    python -m d2d.train {n} dry-run     # every setup step plus one batch per model (a few minutes, trains nothing)
    python -m d2d.train {n}             # trains: LiteVAE, then diffusion ({', '.join(sizes)})

Training takes a day or more. Keep it running when you log out:

    nohup python -m d2d.train {n} > train.log 2>&1 &      # then: tail -f train.log

If it stops for any reason, run the same command again: finished steps are skipped and
an interrupted one resumes from its newest checkpoint (saved every 5 epochs).

### On a SLURM cluster

Ask for one GPU of 32 GB or more, 8 cores and 64 GB, then run the commands above on
the node, or submit the prepared jobs from this folder:

    mkdir -p d2d/workspace/results/training/slurm
    jid=$(sbatch --parsable d2d/workspace/results/{n}/d2d/training/train_litevae.sbatch)
    sbatch --dependency=afterok:$jid d2d/workspace/results/{n}/d2d/training/train_diffusion.sbatch

Add `--gres=gpu:<type>:1` to `sbatch` to pick a GPU type (`sinfo -o "%P %G"` lists them).
After the jobs finish, run `python -m d2d.train {n} pack`.

## 4. Bring the results back

When training finishes, `d2d_results_{n}.zip` is written in this folder (or run
`python -m d2d.train {n} pack` at any time for the runs so far). Copy it to the computer
with the dashboard and use **Pipeline → Import results**. The pipeline then checks the
runs, generates and scores images, and fills in the Results page by itself.
"""


def start(ds):
    if (t := _threads.get(ds.name)) and t.is_alive():
        return status(ds)
    if not ((ds.config().get('training') or {}).get('env')):
        raise ValueError('the training stage has to finish first: it chooses the settings the bundle carries')
    out = bundle_dir(ds)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'status.json', {'state': 'building', 'done': 0, 'total': 0, 'started': time.time()})
    t = threading.Thread(target=_build, args=(ds,), daemon=True, name=f'bundle-{ds.name}')
    _threads[ds.name] = t
    t.start()
    return status(ds)


def _build(ds):
    out = bundle_dir(ds)
    final = out / f'd2d_train_{ds.name}.zip'
    tmp = final.with_suffix('.part')
    try:
        files = _files(ds)
        data_gb = sum(p.stat().st_size for p, a in files if a.startswith('d2d/workspace/')) / 1e9
        top = f'd2d_train_{ds.name}'
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            z.writestr(f'{top}/RUN_TRAINING.md', directions(ds, data_gb))
            for i, (p, arc) in enumerate(files):
                z.write(p, f'{top}/{arc}',
                        compress_type=zipfile.ZIP_STORED if p.suffix.lower() in STORED else zipfile.ZIP_DEFLATED)
                if i % 200 == 0:
                    write_json(out / 'status.json', {'state': 'building', 'done': i, 'total': len(files)})
        tmp.replace(final)
        (out / 'RUN_TRAINING.md').write_text(directions(ds, data_gb), encoding='utf-8')
        write_json(out / 'status.json', {'state': 'ready', 'file': final.name, 'files': len(files),
                                         'size_gb': round(final.stat().st_size / 1e9, 2)})
        ds.log('training', f'training bundle ready: {final.name} ({final.stat().st_size / 1e9:.2f} GB)')
    except Exception as e:  # noqa: BLE001 - shown in the dashboard
        tmp.unlink(missing_ok=True)
        write_json(out / 'status.json', {'state': 'failed', 'message': f'{type(e).__name__}: {e}'})


def import_results(ds, zip_path):
    """Unpack d2d_results_<dataset>.zip into the workspace. Only this dataset's run
    folders under results/training/ and models/ are taken; anything else is refused."""
    taken, refused = 0, []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            parts = PurePosixPath(info.filename.replace('\\', '/')).parts
            if parts[:1] == ('workspace',):
                parts = parts[1:]
            if parts[:2] == ('results', 'training') and len(parts) >= 4:
                run = parts[2]
            elif parts[:1] == ('models',) and len(parts) >= 3:
                run = parts[1]
            else:
                run = None
            if run is None or not _is_run(ds.name, run) or any(p in ('..', '') or ':' in p for p in parts):
                refused.append(info.filename)
                continue
            dst = ROOT.joinpath(*parts)
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dst, 'wb') as f:
                shutil.copyfileobj(src, f, 1 << 20)
            taken += 1
    if not taken:
        raise ValueError(f'no {ds.name} training runs in that zip' + (f' ({len(refused)} other files ignored)' if refused else ''))
    ds.log('training', f'imported {taken} files of training results' + (f'; ignored {len(refused)} others' if refused else ''))
    return {'imported': taken, 'ignored': len(refused)}


def _is_run(name, folder):
    prefixes = [f'litevae_{name}_v3'] + [f'diffusion_{name}_{s}_' for s in ('small', 'medium', 'large')]
    return any(folder == f or folder.startswith(f) for f in prefixes)


__all__ = ['start', 'status', 'directions', 'import_results', 'bundle_dir', 'run_dirs']
