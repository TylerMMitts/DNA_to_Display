# The d2d dashboard: a local web app over the workspace.
#
#     python -m d2d dashboard            -> http://127.0.0.1:8765
#
# Read-mostly. It shows every dataset's stages, reports, figures and decisions,
# training curves and previews with live QC alerts, and the statistical results.
# It can start a stage, approve or edit a decision, and pick up a Claude Code
# decision; each stage runs in a background thread, one at a time per dataset.
# Files are served only from inside the workspace.
import json
import threading
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from .. import autopilot, llm, qc_checks as Q, settings
from ..project import ROOT, STAGE_TITLES, STAGES, Dataset, _clean, list_datasets, read_json, write_json
from ..stages import base

STATIC = Path(__file__).parent / 'static'
app = FastAPI(title='d2d dashboard')
_jobs = {}
_lock = threading.Lock()


def _ds(name):
    ds = Dataset(name)
    if not ds.config_path.exists():
        raise HTTPException(404, f'no dataset {name!r}')
    return ds


def _url(path):
    p = Path(path).resolve()
    try:
        return '/files/' + str(p.relative_to(ROOT)).replace('\\', '/')
    except ValueError:
        return None


def _figures(ds, stage):
    d = ds.stage_dir(stage)
    return [{'name': p.name, 'url': _url(p)} for p in sorted(d.glob('*.png'))] if d.exists() else []


def _background(ds, key, fn, *args, **kw):
    with _lock:
        t = _jobs.get((ds.name, key))
        if t and t.is_alive():
            raise HTTPException(409, f'{key} is already running')
        th = threading.Thread(target=fn, args=args, kwargs=kw, daemon=True)
        _jobs[(ds.name, key)] = th
        th.start()


@app.get('/', response_class=HTMLResponse)
def index():
    # Never cached: the page changes with the app, and a stale copy calls old endpoints.
    return HTMLResponse((STATIC / 'index.html').read_text(encoding='utf-8'),
                        headers={'Cache-Control': 'no-store'})


@app.get('/static/{name}')
def static(name: str):
    p = (STATIC / name).resolve()
    if p.parent != STATIC.resolve() or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get('/files/{path:path}')
def files(path: str):
    p = (ROOT / path).resolve()
    if not str(p).startswith(str(ROOT)) or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)


@app.get('/api/datasets')
def datasets():
    out = []
    for n in list_datasets():
        st = Dataset(n).state()
        out.append({'name': n, 'stages': {s: st[s]['status'] for s in STAGES},
                    'autopilot': st.get('autopilot', {})})
    return {'workspace': str(ROOT), 'datasets': out, 'stages': STAGES, 'titles': STAGE_TITLES,
            'backends': llm.BACKENDS, 'backend': settings.get()['backend'], 'pending_tasks': len(llm.pending_tasks())}


UPLOADS = ROOT / 'uploads'


def _safe_name(name):
    name = (name or '').strip()
    if not name or not name.replace('_', '').replace('-', '').isalnum():
        raise HTTPException(400, 'name: letters, digits, - and _ only')
    return name


@app.put('/api/upload/{name}')
async def upload_file(name: str, path: str, request: Request):
    """One file of a dataset upload, streamed to workspace/uploads/<name>/<path>."""
    name = _safe_name(name)
    rel = Path(path.replace(chr(92), '/'))
    if rel.is_absolute() or '..' in rel.parts:
        raise HTTPException(400, 'bad path')
    dst = (UPLOADS / name / rel).resolve()
    if not str(dst).startswith(str((UPLOADS / name).resolve())):
        raise HTTPException(400, 'bad path')
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, 'wb') as f:
        async for chunk in request.stream():
            f.write(chunk)
    return {'ok': True}


@app.post('/api/datasets')
def new_dataset(body: dict = Body(...)):
    """Register a dataset and start the autopilot. The source is either files
    uploaded to workspace/uploads/<name>/ (a zip is unpacked) or a folder path."""
    import zipfile
    name = _safe_name(body.get('name'))
    ds = Dataset(name)
    if ds.config_path.exists():
        raise HTTPException(409, f'{name} exists')
    if body.get('uploaded'):
        src = UPLOADS / name
        zips = list(src.rglob('*.zip'))
        if zips and len([p for p in src.rglob('*') if p.is_file()]) == len(zips):
            for z in zips:
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(src / z.stem)
                z.unlink()
    else:
        src = Path((body.get('source') or '').strip())
    if not src.exists():
        raise HTTPException(400, f'source folder not found: {src}')
    # Where training runs, asked when the data is imported: 'local' (this computer),
    # 'export' (a bundle for another machine) or 'later' (ask again at training).
    where = body.get('where') or ('local' if body.get('allow_local_training') else 'later')
    if where not in ('local', 'export', 'later'):
        raise HTTPException(400, "where must be 'local', 'export' or 'later'")
    # No agent or cluster settings per dataset: the agent is the one in Settings.
    write_json(ds.config_path, {'name': name, 'source': str(src.resolve()),
                                'training': {'allow_local_training': where == 'local', 'where': where}})
    autopilot.start(name)
    return {'ok': True}


@app.post('/api/ds/{name}/pause')
def pause(name: str):
    _ds(name)
    autopilot.pause(name)
    return autopilot.status(Dataset(name))


@app.post('/api/ds/{name}/resume')
def resume(name: str):
    _ds(name)
    autopilot.start(name)
    return autopilot.status(Dataset(name))


@app.post('/api/ds/{name}/stage/{stage}/rerun')
def rerun_from(name: str, stage: str, body: dict = Body(default={})):
    """While paused: optionally replace a stage's decision, then redo that stage and
    everything after it. The autopilot continues from there when resumed."""
    ds = _ds(name)
    if stage not in STAGES:
        raise HTTPException(404)
    if not autopilot.status(ds).get('paused'):
        raise HTTPException(409, 'pause the autopilot first')
    later = STAGES[STAGES.index(stage):]
    if body.get('decision') is not None:
        st = base.approve(ds, stage, body['decision'])
        if st['status'] != 'done':
            raise HTTPException(400, st.get('message', 'decision rejected'))
        later = later[1:]
    for s in later:
        ds.set_stage(s, status='pending', message='to be redone')
    if stage in ('training', 'images', 'intake', 'genetics'):
        from ..autopilot import _set
        _set(ds, training_started=False)
    return {'ok': True}


@app.get('/api/ds/{name}')
def dataset(name: str):
    ds = _ds(name)
    usage = {}
    path = ds.state_dir / 'usage.jsonl'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            u = usage.setdefault(r['stage'], {'calls': 0, 'input_tokens': 0, 'output_tokens': 0,
                                              'cache_read_input_tokens': 0})
            u['calls'] += 1
            for k in ('input_tokens', 'output_tokens', 'cache_read_input_tokens'):
                u[k] += r.get(k, 0)
    running = [k for (n, k), t in _jobs.items() if n == name and t.is_alive()]
    return {'name': name, 'config': ds.config(), 'state': ds.state(), 'events': ds.events(150),
            'usage': usage, 'running': running, 'autopilot': autopilot.status(ds)}


@app.get('/api/ds/{name}/stage/{stage}')
def stage(name: str, stage: str):
    ds = _ds(name)
    if stage not in STAGES:
        raise HTTPException(404)
    report = ds.report(stage)
    return {'state': ds.state()[stage], 'report': report, 'decision': ds.decision(stage),
            'result': read_json(ds.stage_dir(stage) / 'result.json'),
            'task': read_json(llm.task_path(ds, stage)), 'figures': _figures(ds, stage),
            'schema': base.get_stage(stage).schema(ds, report) if report else None}


@app.post('/api/ds/{name}/stage/{stage}/run')
def run_stage(name: str, stage: str, body: dict = Body(default={})):
    ds = _ds(name)
    _background(ds, stage, base.run, ds, stage, None, auto_approve=bool(body.get('auto')),
                force_agent=bool(body.get('force_agent')))
    return {'started': stage}


@app.post('/api/ds/{name}/stage/{stage}/approve')
def approve_stage(name: str, stage: str, body: dict = Body(default={})):
    ds = _ds(name)
    _background(ds, stage, base.approve, ds, stage, body.get('decision'))
    return {'started': f'apply {stage}'}


@app.post('/api/ds/{name}/stage/{stage}/resume')
def resume_stage(name: str, stage: str):
    ds = _ds(name)
    return base.resume(ds, stage)


@app.get('/api/ds/{name}/training')
def training(name: str):
    ds = _ds(name)
    meta = ds.scaled_metadata_path()
    real = []
    if meta.exists():
        m = pd.read_csv(meta)
        real = [ds.scaled_dir() / f for f in m.scaled_filename.sample(min(40, len(m)), random_state=1)]
    runs = []
    for run in Q.find_runs(ds):
        h = Q.load_history(run)
        issues = Q.check_litevae(h) if run['kind'] == 'litevae' else Q.check_diffusion(h)
        previews = Q.latest_previews(run, 8)
        if previews and real:
            issues += Q.check_images(previews, real, train_real=real)
        keep = [c for c in h.columns if c in (
            'epoch', 'train_loss', 'val_loss', 'recon_loss', 'val_plain_mse', 'all_white_mse',
            'val_loss_wrong_genotype', 'train_eval_loss', 'genotype_gain_heldout', 'genotype_gain_trained', 'lr')]
        runs.append({'name': run['name'], 'kind': run['kind'], 'summary': Q.summary(h, run['kind']),
                     'history': h[keep].to_dict('list') if len(h) else {}, 'issues': issues,
                     'previews': [{'name': p.name, 'url': _url(p)} for p in previews]})
    return _clean({'runs': runs})


@app.get('/api/ds/{name}/results')
def results(name: str):
    ds = _ds(name)
    out = ds.results_dir / 'analysis'
    res = {}
    for role in ('validation', 'test'):
        acc = out / f'trait_accuracy_{role}.csv'
        pts = out / f'scatter_points_{role}.csv'
        if acc.exists():
            ex = read_json(out / f'extras_{role}.json', {})
            pairs = out / f'memorisation_pairs_{role}.png'
            res[role] = {'accuracy': pd.read_csv(acc).to_dict('records'),
                         'points': pd.read_csv(pts).to_dict('records') if pts.exists() else [],
                         'figure': _url(out / f'scatter_{role}.png') if (out / f'scatter_{role}.png').exists() else None,
                         'extras': ex, 'pairs_figure': _url(pairs) if pairs.exists() else None}
    gen = sorted((out / 'generated' / 'validation').glob('*_seed0.png'))[:24] if out.exists() else []
    res['examples'] = [{'name': p.stem, 'url': _url(p)} for p in gen]
    # Matching real images for the examples, when the scaled images are in the workspace.
    meta = ds.scaled_metadata_path()
    if meta.exists() and gen:
        m = pd.read_csv(meta)
        first = m.groupby(m.genotype.astype(str)).scaled_filename.first()
        for e in res['examples']:
            line = e['name'].rsplit('_seed', 1)[0]
            if line in first.index:
                e['real_url'] = _url(ds.scaled_dir() / first[line])
            e['line'] = line
    qc = ds.report('qc') or {}
    res['qc'] = {'verdict': (ds.config().get('qc') or {}).get('verdict'),
                 'recommended_checkpoint': (ds.config().get('qc') or {}).get('recommended_checkpoint'),
                 'runs': qc.get('runs', []), 'generated_issues': qc.get('generated_issues', [])}
    res['genetics'] = {'encoding': (ds.config().get('genetics') or {}).get('encoding')}
    gm = read_json(out / 'gene_maps.json')
    if gm:
        for g in gm.get('genes', []):
            g['map_url'] = _url(out / g['map']) if g.get('map') else None
        gm['gwas_figure'] = _url(out / 'gwas_validation.png') if (out / 'gwas_validation.png').exists() else None
        res['genes'] = gm
    return _clean(res)


@app.get('/api/ds/{name}/exports')
def exports_list(name: str):
    from .. import exports as X
    ds = _ds(name)
    return _clean({'options': X.options(ds), 'jobs': X.list_jobs(ds)})


@app.post('/api/ds/{name}/exports')
def exports_start(name: str, body: dict = Body(...)):
    from .. import exports as X
    ds = _ds(name)
    X.options(ds)                     # makes sure the line groups are known
    try:
        return X.start(ds, body)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get('/api/ds/{name}/exports/{job}/download')
def exports_download(name: str, job: str):
    from .. import exports as X
    ds = _ds(name)
    p = X.zip_path(ds, job)
    if p is None or not p.exists():
        raise HTTPException(404, 'not ready')
    return FileResponse(p, media_type='application/zip', filename=p.name)


@app.delete('/api/ds/{name}/exports/{job}')
def exports_delete(name: str, job: str):
    from .. import exports as X
    ds = _ds(name)
    if not job.replace('_', '').isalnum():
        raise HTTPException(400)
    X.delete(ds, job)
    return {'ok': True}


@app.get('/api/ds/{name}/export.zip')
def export(name: str, images: int = 0):
    """Every results table, JSON and figure of the analysis (and, with images=1, the
    generated images), plus the dataset's decisions, as one zip."""
    import io
    import zipfile
    from fastapi.responses import Response
    ds = _ds(name)
    out = ds.results_dir / 'analysis'
    if not out.exists():
        raise HTTPException(404, 'no results yet')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and (images or 'generated' not in p.relative_to(out).parts):
                z.write(p, f'{name}_results/{p.relative_to(out).as_posix()}')
        tr = ds.results_dir / 'traits'
        for p in sorted(tr.glob('*.csv')) if tr.exists() else []:
            z.write(p, f'{name}_results/real_traits/{p.name}')
        for s in STAGES:
            for f in ('decision.json', 'result.json'):
                p = ds.stage_dir(s) / f
                if p.exists():
                    z.write(p, f'{name}_results/pipeline/{s}_{f}')
        z.writestr(f'{name}_results/config.json', json.dumps(ds.config(), indent=1))
    return Response(buf.getvalue(), media_type='application/zip',
                    headers={'Content-Disposition': f'attachment; filename="{name}_results{"_with_images" if images else ""}.zip"'})


# --- settings ------------------------------------------------------------------------------

@app.get('/api/settings')
def settings_get():
    from .. import settings
    return settings.public()


@app.put('/api/settings')
def settings_put(body: dict = Body(...)):
    from .. import settings
    body.pop('api_key', None)                      # the key has its own endpoint
    try:
        settings.save({k: v for k, v in body.items() if k in settings.DEFAULTS})
    except ValueError as e:
        raise HTTPException(400, str(e))
    return settings.public()


@app.put('/api/settings/api-key')
def settings_api_key(body: dict = Body(...)):
    from .. import settings
    try:
        settings.set_api_key(body.get('key'))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return settings.public()


@app.post('/api/settings/test-api')
def settings_test_api():
    from .. import settings
    ok, message = settings.test_api()
    return {'ok': ok, 'message': message}


# --- where training runs: this computer, or a bundle for another machine ----------------

@app.get('/api/system')
def system_check(bytes: int = 0, source: str = ''):
    """This computer against what training needs. The dataset's size (bytes, or a
    folder path to measure) sets the disk requirement."""
    from .. import system
    if not bytes and source.strip() and Path(source.strip()).is_dir():
        bytes = system.folder_bytes(source.strip())
    return system.check(bytes or None)


@app.post('/api/ds/{name}/training/where')
def training_where(name: str, body: dict = Body(...)):
    where = body.get('where')
    if where not in ('local', 'export'):
        raise HTTPException(400, "where must be 'local' or 'export'")
    ds = _ds(name)
    ds.update_config(training={'where': where, 'allow_local_training': where == 'local'})
    ds.log('training', 'training will run on this computer' if where == 'local'
           else 'training will run elsewhere: a bundle is built when the settings are ready')
    return {'ok': True, 'where': where}


@app.get('/api/ds/{name}/bundle')
def bundle_status(name: str):
    from .. import bundle
    ds = _ds(name)
    return {**bundle.status(ds), 'ready_to_build': bool((ds.config().get('training') or {}).get('env'))}


@app.post('/api/ds/{name}/bundle')
def bundle_build(name: str):
    from .. import bundle
    try:
        return bundle.start(_ds(name))
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.get('/api/ds/{name}/bundle/download')
def bundle_download(name: str):
    from .. import bundle
    ds = _ds(name)
    s = bundle.status(ds)
    p = bundle.bundle_dir(ds) / (s.get('file') or 'missing')
    if s.get('state') != 'ready' or not p.exists():
        raise HTTPException(404, 'no bundle built yet')
    return FileResponse(p, media_type='application/zip', filename=p.name)


@app.get('/api/ds/{name}/bundle/directions')
def bundle_directions(name: str):
    from fastapi.responses import PlainTextResponse
    from .. import bundle
    return PlainTextResponse(bundle.directions(_ds(name)), media_type='text/markdown')


@app.put('/api/ds/{name}/import-results')
async def import_results(name: str, request: Request):
    """d2d_results_<name>.zip from another machine, streamed, then unpacked into the workspace."""
    from .. import bundle
    ds = _ds(name)
    tmp = UPLOADS / '_results' / f'{name}.zip'
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, 'wb') as f:
        async for chunk in request.stream():
            f.write(chunk)
    try:
        return bundle.import_results(ds, tmp)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001 - a bad zip
        raise HTTPException(400, f'could not read that zip: {e}')
    finally:
        tmp.unlink(missing_ok=True)


def serve(host='127.0.0.1', port=8765):
    print(f'd2d dashboard: http://{host}:{port}   (workspace {ROOT})')
    autopilot.resume_all()
    uvicorn.run(app, host=host, port=port, log_level='warning')
