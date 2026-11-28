# Generated datasets on demand: pick lines and how many images per line, and get a
# zip of generated images with one metadata table describing every image.
#
# A job runs in a background thread of the dashboard. It writes into
# results/<dataset>/exports/<job id>/:
#   images/<line>_seed<s>.png    the generated images (same seed = same starting noise
#                                for every line)
#   metadata.csv                 one row per image: file, line, seed, which group the line
#                                belongs to for this model (trained / validation / test /
#                                never imaged), its covariates, its real trait values when
#                                known, and the traits measured on the generated image
#   genotypes.csv                the genotype rows of the chosen lines (optional)
#   README.txt                   what every column means, the checkpoint and settings
#   <job id>.zip                 all of the above
#
# Test lines are sealed: they are only generated when the request says so explicitly.
import json
import shutil
import threading
import time
import uuid
import zipfile

import numpy as np
import pandas as pd

from .project import read_json, write_json

MAX_IMAGES = 4000
_jobs = {}
_lock = threading.Lock()


def line_groups(ds, ckpt_path):
    """Every genotyped line and its group for this checkpoint."""
    import torch
    from .stages.genetics import load_genotypes
    lines, _, _ = load_genotypes(ds)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    val, test = set(map(str, ck.get('val_genotypes', []))), set(map(str, ck.get('test_genotypes', [])))
    split = pd.read_csv(ds.split_path) if ds.split_path.exists() else pd.DataFrame(columns=['line', 'split_random'])
    test |= set(split.line[split.split_random == 'test'].astype(str))
    imaged = set(pd.read_csv(ds.scaled_metadata_path()).genotype.astype(str)) if ds.scaled_metadata_path().exists() else set()
    out = {}
    for l in map(str, lines):
        out[l] = ('test' if l in test else 'validation' if l in val else 'trained' if l in imaged else 'never imaged')
    del ck
    return out


def options(ds):
    from .generate import find_checkpoint
    ckpt = find_checkpoint(ds, (ds.config().get('qc') or {}).get('recommended_checkpoint', ''))
    if ckpt is None:
        return {'checkpoint': None, 'groups': {}}
    cache = ds.results_dir / 'exports' / 'line_groups.json'
    cached = read_json(cache, {})
    if cached.get('checkpoint') != str(ckpt):
        cached = {'checkpoint': str(ckpt), 'groups': line_groups(ds, ckpt)}
        write_json(cache, cached)
    g = cached['groups']
    counts = pd.Series(g).value_counts().to_dict() if g else {}
    return {'checkpoint': ckpt.name, 'counts': counts, 'max_images': MAX_IMAGES,
            'examples': {k: [l for l, v in g.items() if v == k][:5] for k in counts}}


def pick_lines(ds, spec):
    groups = read_json(ds.results_dir / 'exports' / 'line_groups.json', {}).get('groups', {})
    if spec.get('custom_lines'):
        want = [l.strip() for l in str(spec['custom_lines']).replace(',', '\n').splitlines() if l.strip()]
        unknown = [l for l in want if l not in groups]
        if unknown:
            raise ValueError(f'{len(unknown)} lines are not in the genotype table, e.g. {unknown[:3]}')
        sealed = [l for l in want if groups[l] == 'test']
        if sealed and not spec.get('include_test'):
            raise ValueError(f'{len(sealed)} of these are sealed test lines; tick "include sealed test lines" '
                             'only for a final check')
        return want
    allowed = set(spec.get('groups') or ['validation'])
    if 'test' in allowed and not spec.get('include_test'):
        allowed.discard('test')
        if not allowed:
            raise ValueError('test lines are sealed; tick "include sealed test lines" only for a final check')
    pool = sorted(l for l, g in groups.items() if g in allowed)
    n = int(spec.get('n_lines') or len(pool))
    if not pool:
        raise ValueError('no lines in the chosen groups')
    rng = np.random.default_rng(int(spec.get('pick_seed', 0)))
    return sorted(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist())


def start(ds, spec):
    lines = pick_lines(ds, spec)
    n_seeds = max(1, int(spec.get('images_per_line', 4)))
    total = len(lines) * n_seeds
    if total > MAX_IMAGES:
        raise ValueError(f'{total} images requested; the limit per export is {MAX_IMAGES}')
    job = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:4]
    state = {'id': job, 'dataset': ds.name, 'status': 'queued', 'done': 0, 'total': total, 'message': 'queued',
             'lines': len(lines), 'images_per_line': n_seeds, 'spec': spec, 'started': time.time()}
    d = ds.results_dir / 'exports' / job
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / 'job.json', state)
    with _lock:
        _jobs[(ds.name, job)] = state
    threading.Thread(target=_run, args=(ds, job, lines, n_seeds, spec), daemon=True).start()
    return state


def _set(ds, job, **kw):
    with _lock:
        st = _jobs.setdefault((ds.name, job), {})
        st.update(kw)
        write_json(ds.results_dir / 'exports' / job / 'job.json', st)


def _run(ds, job, lines, n_seeds, spec):
    from .generate import find_checkpoint, generate
    from .measure import measure_paths
    d = ds.results_dir / 'exports' / job
    try:
        ckpt = find_checkpoint(ds, (ds.config().get('qc') or {}).get('recommended_checkpoint', ''))
        seed0 = int(spec.get('seed_start', 0))
        steps = int(spec.get('steps', 50))
        # Reuse images the analysis stage already made with the same checkpoint and seeds.
        cache = ds.results_dir / 'analysis' / 'generated' / 'validation'
        img_dir = d / 'images'
        img_dir.mkdir(exist_ok=True)
        if steps == 50 and cache.exists():
            for l in lines:
                for s in range(seed0, seed0 + n_seeds):
                    src = cache / f'{l}_seed{s}.png'
                    if src.exists() and not (img_dir / src.name).exists():
                        shutil.copy2(src, img_dir / src.name)
        _set(ds, job, status='generating', message='generating images',
             done=len(list(img_dir.glob('*.png'))))
        generate(ds, ckpt, 'validation', n_seeds, img_dir, steps=steps, lines=lines, seed_start=seed0,
                 progress=lambda k, n: _set(ds, job, done=min(k, n) + (len(lines) * n_seeds - n)))
        files = sorted(img_dir.glob('*_seed*.png'))
        meta = pd.DataFrame({'file': ['images/' + f.name for f in files],
                             'line': [f.stem.rsplit('_seed', 1)[0] for f in files],
                             'seed': [int(f.stem.rsplit('_seed', 1)[1]) for f in files]})
        meta = meta[meta.line.isin(lines) & meta.seed.between(seed0, seed0 + n_seeds - 1)]
        groups = read_json(ds.results_dir / 'exports' / 'line_groups.json', {}).get('groups', {})
        meta['line_group'] = meta.line.map(groups)
        meta['checkpoint'] = ckpt.name
        meta['sampling_steps'] = steps
        c = covariates(ds, lines)
        if c is not None and len(c.columns):
            meta = meta.join(c.add_prefix('covariate: '), on='line')
        if spec.get('include_real_traits', True):
            real = ds.results_dir / 'traits' / 'real_line_means.csv'
            if real.exists():
                r = pd.read_csv(real, index_col=0)
                r.index = r.index.astype(str)
                meta = meta.join(r.add_prefix('real line mean: '), on='line')
            if ds.traits_path.exists():
                t = pd.read_csv(ds.traits_path)
                t['line'] = t['line'].astype(str)
                cols = [c for c in t.columns if c not in ('line', 'image_id')]
                meta = meta.join(t.groupby('line')[cols].mean().add_prefix('imported trait: '), on='line')
        if spec.get('measure', True):
            _set(ds, job, status='measuring', message='measuring traits on the generated images')
            recipe = (ds.config().get('traits') or {}).get('recipe')
            m = measure_paths([d / f for f in meta.file], recipe) if recipe else measure_paths([d / f for f in meta.file])
            m['file'] = 'images/' + m['image'].astype(str).map(lambda s: s.split('/')[-1].split('\\')[-1])
            keep = [c for c in m.columns if c not in ('image', 'file', 'ok', 'error')]
            meta = meta.merge(m[['file'] + keep].add_prefix('generated: ').rename(columns={'generated: file': 'file'}),
                              on='file', how='left')
        meta.to_csv(d / 'metadata.csv', index=False)
        if spec.get('include_genotypes', False):
            from .stages.genetics import load_genotypes
            names, cols, x = load_genotypes(ds)
            idx = {l: i for i, l in enumerate(names)}
            pd.DataFrame(x[[idx[l] for l in lines]], index=pd.Index(lines, name='line'), columns=cols
                         ).to_csv(d / 'genotypes.csv')
        (d / 'README.txt').write_text(_readme(ds, ckpt, spec, lines, n_seeds, meta), encoding='utf-8')
        _set(ds, job, status='zipping', message='writing the zip')
        zp = d / f'{ds.name}_generated_{job}.zip'
        with zipfile.ZipFile(zp, 'w', zipfile.ZIP_STORED) as z:     # PNGs are already compressed
            for f in ['README.txt', 'metadata.csv', 'genotypes.csv']:
                if (d / f).exists():
                    z.write(d / f, f'{ds.name}_generated/{f}')
            for f in meta.file:
                z.write(d / f, f'{ds.name}_generated/{f}')
        _set(ds, job, status='done', message=f'{len(meta)} images', done=len(meta), zip=zp.name,
             zip_mb=round(zp.stat().st_size / 1e6, 1), finished=time.time())
    except Exception as e:  # noqa: BLE001 - shown in the dashboard
        import traceback
        ds.log('exports', traceback.format_exc(), level='error')
        _set(ds, job, status='failed', message=f'{type(e).__name__}: {e}')


def covariates(ds, lines):
    """Per-line covariate values: intake's line_covariates.csv, or else decoded from the
    one-hot covariate columns of the genotype table (SEX_male -> sex = male / not male,
    BATCH_2016-03 -> batch = 2016-03)."""
    p = ds.metadata_dir / 'line_covariates.csv'
    if p.exists():
        return pd.read_csv(p, dtype=str).set_index('line')
    from .analysis_extra import covariate_columns
    from .stages.genetics import load_genotypes
    names, cols, x = load_genotypes(ds)
    cov = covariate_columns(ds, cols)
    if not cov:
        return None
    idx = {l: i for i, l in enumerate(names)}
    rows = x[[idx[l] for l in lines if l in idx]][:, cov]
    out = pd.DataFrame(index=pd.Index([l for l in lines if l in idx], name='line'))
    groups = {}
    for j, c in zip(range(len(cov)), [cols[i] for i in cov]):
        prefix, _, level = c.partition('_')
        groups.setdefault(prefix.lower(), []).append((j, level or c))
    for name, members in groups.items():
        if len(members) == 1:
            j, level = members[0]
            out[name] = np.where(rows[:, j] > 0, level, f'not {level}')
        else:
            vals = rows[:, [j for j, _ in members]]
            levels = np.array([lv for _, lv in members], dtype=object)
            out[name] = np.where(vals.max(axis=1) > 0, levels[vals.argmax(axis=1)], 'other')
    return out


def _readme(ds, ckpt, spec, lines, n_seeds, meta):
    cols = '\n'.join(f'  {c}' for c in meta.columns)
    return f"""Generated images: {ds.name}
==========================================

{len(meta)} images: {len(lines)} lines x {n_seeds} images each, made by d2d from the
genotype alone. Checkpoint {ckpt.name}, {spec.get('steps', 50)} sampling steps,
seeds {spec.get('seed_start', 0)}-{int(spec.get('seed_start', 0)) + n_seeds - 1}. The same seed starts from the same noise
for every line, so images with equal seeds differ only by genotype.

metadata.csv, one row per image:
  file                     path of the image inside this zip
  line                     the line (genotype) the image was generated for
  seed                     noise seed
  line_group               the line's role for this model: trained (its real images were
                           used in training), validation or test (held out), never imaged
  checkpoint, sampling_steps
  covariate: *             how/when the line's real images were taken (sex, imaging month...);
                           part of the conditioning when the genotype table carries them
  real line mean: *        traits measured the same way on the line's real images
  imported trait: *        trait values from the uploaded trait table (line means)
  generated: *             traits measured on this generated image

All columns:
{cols}

genotypes.csv (when included): the genotype rows of these lines, 0/1/2 per marker.
"""


def list_jobs(ds):
    base = ds.results_dir / 'exports'
    out = []
    for p in sorted(base.glob('*/job.json'), reverse=True) if base.exists() else []:
        st = read_json(p, {})
        live = _jobs.get((ds.name, st.get('id')))
        st = dict(live or st)
        if st.get('status') in ('queued', 'generating', 'measuring', 'zipping') and not live:
            st['status'], st['message'] = 'failed', 'interrupted (the dashboard restarted)'
        out.append(st)
    return out


def zip_path(ds, job):
    st = read_json(ds.results_dir / 'exports' / job / 'job.json', {})
    if st.get('status') != 'done' or not st.get('zip'):
        return None
    return ds.results_dir / 'exports' / job / st['zip']


def delete(ds, job):
    d = ds.results_dir / 'exports' / job
    if d.exists() and d.parent.name == 'exports':
        shutil.rmtree(d)
