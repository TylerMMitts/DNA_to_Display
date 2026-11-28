# Trains a d2d dataset from start to finish on whatever machine this runs on: this
# computer, a workstation, or a cluster node. From the folder that holds code/ and d2d/:
#
#     python -m d2d.train <dataset>              check the machine, then LiteVAE, then every diffusion size
#     python -m d2d.train <dataset> check        only the machine and data checks
#     python -m d2d.train <dataset> dry-run      setup plus one no-grad batch per model (minutes, trains nothing)
#     python -m d2d.train <dataset> litevae      one step
#     python -m d2d.train <dataset> diffusion
#     python -m d2d.train <dataset> pack         zip the finished runs for the dashboard's Import results
#
# Steps already finished are skipped, and an interrupted step resumes from its newest
# checkpoint, so rerunning the same command is always safe. When everything is done it
# writes d2d_results_<dataset>.zip next to code/ for importing back.
# --force trains even on a machine below the minimum requirements.
import os
import subprocess
import sys
import zipfile

from . import system
from .project import PROJECT_ROOT, ROOT, Dataset


def settings(ds):
    t = ds.config().get('training') or {}
    if not t.get('env'):
        raise SystemExit(f'{ds.name}: the training stage has not run yet, so there are no settings to train with')
    return t['env'], t.get('sizes') or ['medium']


def litevae_done(ds, env):
    d = ROOT / 'models' / f'litevae_{ds.name}_v3'
    return (d / f"litevae_{ds.name}_v3_epoch_{env['WIDIV_LITEVAE_EPOCHS']}.pt").exists()


def diffusion_done(ds, size):
    return any((p / 'run_summary.json').exists()
               for p in (ROOT / 'results' / 'training').glob(f'diffusion_{ds.name}_{size}_*'))


def machine_check(force):
    c = system.check()
    print('\nThis machine against what training needs:')
    for r in c['rows']:
        have = r.get('have_text') or (f"{r['have']:g} {r['unit']}" if r['have'] is not None else 'unknown')
        need = (f"min {r['minimum']:g}, recommended {r['recommended']:g} {r['unit']}"
                if isinstance(r['minimum'], (int, float)) else r['recommended'])
        mark = {'ok': 'ok  ', 'min': 'low ', 'fail': 'FAIL', 'unknown': '?   '}[r['level']]
        print(f"  {mark} {r['key']:<14} {have:<34} ({need})")
    g = c['specs']['gpu']
    if g and g['used_gb'] > 1:
        print(f"  note: {g['used_gb']:g} GB of the GPU is already in use by another program")
    print(f"  {c['time_note']}")
    if c['verdict'] == 'not_recommended':
        msg = 'this machine is below the minimum for training'
        if not force:
            raise SystemExit(f'\nstopped: {msg}. Train elsewhere (Export in the dashboard), or add --force.')
        print(f'\nwarning: {msg}; going on because of --force')
    elif c['verdict'] == 'possible':
        print('\nThis machine meets the minimum but not the recommendation: training works, more slowly.')


def step(env, *args):
    print(f"\n=== python -m {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, '-u', '-m', *args], cwd=PROJECT_ROOT,
                       env={**os.environ, **env, 'PYTHONUNBUFFERED': '1'})
    if r.returncode:
        raise SystemExit(f'stopped: {args[0]} exited with code {r.returncode}. Rerun the same command to resume.')


def run_dirs(base, name):
    """This dataset's run folders only (not those of a dataset whose name starts the same)."""
    prefixes = [f'litevae_{name}_v3'] + [f'diffusion_{name}_{s}_' for s in ('small', 'medium', 'large')]
    return [p for p in base.glob('*') if p.is_dir() and any(p.name == f or p.name.startswith(f) for f in prefixes)]


def pack(ds):
    """The run folders and the best and newest checkpoint of each model, as one zip."""
    out = PROJECT_ROOT / f'd2d_results_{ds.name}.zip'
    files = []
    for run in run_dirs(ROOT / 'results' / 'training', ds.name):
        files += [p for p in run.rglob('*') if p.is_file()]
    for mdir in run_dirs(ROOT / 'models', ds.name):
        pts = sorted(mdir.glob('*_epoch_*.pt'), key=lambda p: int(p.stem.rsplit('_', 1)[1]))
        files += list(mdir.glob('*_best.pt')) + pts[-1:]
    if not files:
        raise SystemExit('nothing to pack yet: no training runs for this dataset')
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for p in files:
            z.write(p, 'workspace/' + p.relative_to(ROOT).as_posix())
    print(f'\nwrote {out} ({out.stat().st_size / 1e9:.2f} GB): in the dashboard, Pipeline > Import results')
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    force = '--force' in argv
    argv = [a for a in argv if a != '--force']
    usage = 'usage: python -m d2d.train <dataset> [all|check|dry-run|litevae|diffusion|pack] [--force]'
    if not argv or len(argv) > 2:
        raise SystemExit(usage)
    name, what = argv[0], (argv[1] if len(argv) > 1 else 'all')
    if what not in ('all', 'check', 'dry-run', 'litevae', 'diffusion', 'pack'):
        raise SystemExit(usage)
    ds = Dataset(name)
    if not ds.config_path.exists():
        raise SystemExit(f'no d2d dataset {name!r} in {ROOT}')
    if what == 'pack':
        pack(ds)
        return
    env, sizes = settings(ds)
    print(f'd2d training for {name}: project {PROJECT_ROOT}, sizes {", ".join(sizes)}, '
          f"LiteVAE {env['WIDIV_LITEVAE_EPOCHS']} epochs, diffusion {env['WIDIV_NUM_EPOCHS']} epochs")
    machine_check(force or what in ('check', 'dry-run'))
    step(env, 'd2d.cluster_check', name)
    if what == 'check':
        return
    if what == 'dry-run':
        step({**env, 'DIFFUSION_SIZE': sizes[0]}, 'd2d.launch_trainer', name, 'both', '--dry-run')
        return
    if what in ('all', 'litevae'):
        if litevae_done(ds, env):
            print('\nLiteVAE already finished; skipping it')
        else:
            step(env, 'd2d.launch_trainer', name, 'litevae')
    if what in ('all', 'diffusion'):
        for size in sizes:
            if diffusion_done(ds, size):
                print(f'\ndiffusion {size} already finished; skipping it')
                continue
            step({**env, 'DIFFUSION_SIZE': size}, 'd2d.launch_trainer', name, 'diffusion')
    if what == 'all':
        pack(ds)
    print('\ndone.')


if __name__ == '__main__':
    main()
