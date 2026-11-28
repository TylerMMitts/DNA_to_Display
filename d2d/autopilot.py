# Autopilot: carries a dataset from submission to results with no approvals.
#
# After a dataset is submitted, one background worker per dataset runs every
# stage in order, applies each decision as soon as it is made, and keeps going
# until the analysis is done. The user's only control is pause / resume.
#
#   stages      intake -> images -> traits -> genetics -> training -> qc -> analysis
#               Each decision is applied automatically. A failed stage is retried
#               once; a second failure pauses the autopilot with the reason.
#   Claude Code with the claude-code backend, a stage waits until /d2d has answered
#               its task in Claude Code, then continues by itself.
#   training    after the training stage writes the jobs, the autopilot starts them:
#                 local    when the dataset allows local training (this machine's GPU)
#                 cluster  when a cluster host is set and auto_submit is on: copies the
#                          data up and submits the LiteVAE and diffusion jobs as a
#                          chain over ssh (your ssh key, never a password prompt), then
#                          pulls the training curves and best checkpoints back every
#                          few minutes
#                 waiting  otherwise: it waits for run folders to appear in the
#                          workspace (copied back by hand)
#   watching    while training runs, the QC checks run on every new history. A
#               failing check (collapse, divergence, no genotype signal) pauses the
#               autopilot and the QC agent writes what to do.
#   finishing   when a diffusion run has reached its planned epochs, QC and the
#               analysis run, and the autopilot stops.
#
# State lives in the dataset's state.json under 'autopilot', so the dashboard
# shows it and a restarted server picks up where it left off.
import io
import shlex
import subprocess
import sys
import tarfile
import threading
import time
import traceback

from . import qc_checks as Q, settings
from .project import APP_DIR, PROJECT_ROOT, ROOT, STAGES, Dataset, read_json, write_json
from .stages import base

POLL_S = 10                 # how often a waiting worker checks again
SYNC_EVERY_S = 300          # how often training results are pulled from the cluster
_workers = {}
_lock = threading.Lock()


def status(ds):
    return ds.state().get('autopilot', {'phase': 'idle', 'paused': False, 'message': ''})


def _set(ds, **fields):
    st = ds.state()
    ap = st.get('autopilot', {})
    ap.update(fields, updated=time.time())
    st['autopilot'] = ap
    write_json(ds.state_dir / 'state.json', st)
    if 'message' in fields and fields['message']:
        ds.log('autopilot', fields['message'])


def start(name):
    """Start (or resume) the dataset's worker. Safe to call repeatedly."""
    ds = Dataset(name)
    was = status(ds)
    _set(ds, paused=False, enabled=True,
         message='resumed' if was.get('paused') else ('starting' if not was.get('enabled') else was.get('message', '')))
    with _lock:
        w = _workers.get(name)
        if w and w.is_alive():
            return
        w = threading.Thread(target=_run, args=(name,), daemon=True, name=f'autopilot-{name}')
        _workers[name] = w
        w.start()


def pause(name, reason='paused by you'):
    _set(Dataset(name), paused=True, message=reason)


def resume_all():
    """At server start: restart every worker that was running and not paused."""
    base_dir = ROOT / 'dataset'
    if not base_dir.exists():
        return
    for p in base_dir.iterdir():
        ds = Dataset(p.name)
        if ds.config_path.exists():
            ap = status(ds)
            if ap.get('enabled') and not ap.get('paused') and ap.get('phase') != 'finished':
                start(p.name)


def _wait_while_paused(ds):
    while status(ds).get('paused'):
        time.sleep(2)


def _run(name):
    ds = Dataset(name)
    try:
        backend = settings.get()['backend']       # one agent for every dataset (Settings)
        for stage in STAGES:
            _wait_while_paused(ds)
            if stage == 'qc' or stage == 'analysis':
                if not _training_finished(ds):
                    _train_and_watch(ds)
                    _wait_while_paused(ds)
            if ds.state()[stage]['status'] == 'done':
                continue
            # Read per stage, so an agent changed in Settings applies from the next stage.
            if not _run_stage(ds, stage, settings.get()['backend']):
                return
            if stage == 'qc':
                verdict = (ds.decision('qc') or {}).get('verdict')
                if verdict == 'fail':
                    pause(name, 'QC failed after training: see the QC stage for what to do')
                    _wait_while_paused(ds)
        _set(ds, phase='finished', message='all stages done: results are on the Results page')
    except Exception as e:  # noqa: BLE001 - shown in the dashboard, worker stops
        ds.log('autopilot', traceback.format_exc(), level='error')
        _set(ds, phase='error', paused=True, message=f'autopilot stopped: {type(e).__name__}: {e}')


def _run_stage(ds, stage, backend):
    """Run one stage to done. Returns False if the autopilot had to pause."""
    for attempt in (1, 2):
        _set(ds, phase=stage, message=f'running {stage}' + (' (retry)' if attempt == 2 else ''))
        st = base.run(ds, stage, backend, auto_approve=True)
        if st['status'] == 'deciding':
            _claude_code_auto(ds, stage)
        while st['status'] == 'deciding':           # a Claude Code task in flight
            _set(ds, phase=stage, message=f'{stage}: waiting for /d2d in Claude Code'
                 + (' (running it automatically)' if settings.get()['claude_code_auto'] else ''))
            time.sleep(POLL_S)
            _wait_while_paused(ds)
            st = base.resume(ds, stage)
        if st['status'] == 'review':                # decided without auto-approve (resume path)
            st = base.approve(ds, stage)
        if st['status'] == 'done':
            return True
        ds.log('autopilot', f"{stage} failed (attempt {attempt}): {st.get('message')}", level='warning')
    pause(ds.name, f"{stage} failed twice: {st.get('message')}")
    _wait_while_paused(ds)
    return ds.state()[stage]['status'] == 'done' or _run_stage(ds, stage, backend)


def _claude_code_auto(ds, stage):
    """With 'answer automatically' on (Settings), run /d2d in Claude Code headless, signed
    in with the person's own subscription, allowed only the tools the skill uses."""
    if not settings.get()['claude_code_auto']:
        return
    cmd = settings.claude_command()
    if not cmd:
        ds.log('autopilot', 'answer automatically is on, but Claude Code was not found: set its path in Settings',
               level='warning')
        return
    log = ds.state_dir / 'claude_code.log'
    # The fixed model, so a decision does not depend on the session's default model.
    args = [cmd, '-p', '/d2d', '--model', settings.MODEL, '--allowedTools', 'Read', 'Write', 'Edit', 'Agent',
            'Bash(uv run python run.py:*)']

    def go():
        with open(log, 'a', encoding='utf-8') as f:
            f.write(f'\n=== {time.strftime("%Y-%m-%d %H:%M")} {ds.name} {stage}\n')
            f.flush()
            code = subprocess.call(args, cwd=APP_DIR, stdout=f, stderr=subprocess.STDOUT)
        if code:
            ds.log('autopilot', f'Claude Code exited with code {code}; see {log.name}, or run /d2d yourself',
                   level='warning')
    threading.Thread(target=go, daemon=True).start()
    ds.log('autopilot', f'{stage}: asked Claude Code to answer (/d2d, headless)')


# --- training ----------------------------------------------------------------------

def _training_finished(ds):
    """A diffusion run is finished when it reached d2d's planned epochs, or when the
    trainer wrote run_summary.json (it does so only at the end of a run, so a run
    trained elsewhere with a different epoch count also counts)."""
    target = int((ds.config().get('pipeline') or {}).get('epochs', 0) or 0)
    for run in Q.find_runs(ds):
        if run['kind'] != 'diffusion':
            continue
        h = Q.load_history(run)
        if not len(h) or not _has_checkpoint(ds):
            continue
        last = int(h.epoch.max())
        summary = read_json(run['dir'] / 'run_summary.json', {})
        planned = int(summary.get('epochs') or 0)
        if (target and last >= target) or (planned and last >= planned):
            return True
    return False


def _has_checkpoint(ds):
    return any((ROOT / 'models').glob(f'diffusion_{ds.name}*/*.pt'))


def _train_and_watch(ds):
    cfg = ds.config()
    cl = cfg.get('cluster') or {}
    ap = status(ds)
    if not ap.get('training_started'):
        if (cfg.get('training') or {}).get('allow_local_training'):
            _start_local(ds)
            mode = 'local'
        elif cl.get('host') and cl.get('remote_root') and cl.get('auto_submit'):
            _submit_cluster(ds)
            mode = 'cluster'
        else:
            mode = 'manual'
            if (cfg.get('training') or {}).get('where') == 'export':
                _build_bundle(ds)
        _set(ds, training_started=True, training_mode=mode)
    mode = status(ds).get('training_mode')
    last_sync, seen = 0.0, {}
    while not _training_finished(ds):
        _wait_while_paused(ds)
        # Chosen in the dashboard while waiting: train here after all.
        if mode == 'manual' and (ds.config().get('training') or {}).get('allow_local_training'):
            _start_local(ds)
            mode = 'local'
            _set(ds, training_mode=mode)
        if mode == 'cluster' and time.time() - last_sync > SYNC_EVERY_S:
            try:
                _pull_cluster(ds)
            except Exception as e:  # noqa: BLE001 - network hiccups are retried
                ds.log('autopilot', f'pull from cluster failed, will retry: {e}', level='warning')
            last_sync = time.time()
        fails = []
        for run in Q.find_runs(ds):
            h = Q.load_history(run)
            if len(h) == seen.get(run['name']):
                continue
            seen[run['name']] = len(h)
            issues = Q.check_litevae(h) if run['kind'] == 'litevae' else Q.check_diffusion(h)
            fails += [f"{run['name']}: {i['check']}" for i in issues if i['severity'] == 'fail']
        if fails:
            base.run(ds, 'qc', settings.get()['backend'], auto_approve=True)
            pause(ds.name, 'training problem found: ' + '; '.join(fails[:3]) + '. The QC stage says what to do.')
            continue
        epochs = {r['name']: seen.get(r['name'], 0) for r in Q.find_runs(ds)}
        where = (ds.config().get('training') or {}).get('where')
        msg = {'local': 'training on this computer', 'cluster': 'training on the cluster',
               'manual': ('training elsewhere: download the bundle below, then import its results'
                          if where == 'export' else 'ready to train: choose where below')}[mode]
        _set(ds, phase='training', message=msg + (f" ({', '.join(f'{k} epoch {v}' for k, v in epochs.items())})"
                                                 if epochs else ''))
        time.sleep(POLL_S if mode != 'cluster' else 60)


def _build_bundle(ds):
    from . import bundle
    try:
        bundle.start(ds)
        ds.log('autopilot', 'building the training bundle for another machine')
    except ValueError as e:
        ds.log('autopilot', f'training bundle not built: {e}', level='warning')


def _start_local(ds):
    """d2d.train on this machine: LiteVAE, then each diffusion size; finished steps are
    skipped and interrupted ones resume. --force because the person chose this machine
    after seeing how it compares with the requirements."""
    log = ds.state_dir / 'training_local.log'
    cmds = [[sys.executable, '-u', '-m', 'd2d.train', ds.name, step, '--force'] for step in ('litevae', 'diffusion')]

    def chain():
        with open(log, 'a', encoding='utf-8') as f:
            for i, cmd in enumerate(cmds):
                if subprocess.call(cmd, cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT) != 0:
                    ds.log('autopilot', f'local training ({cmd[5]}) exited with an error; see {log.name}', level='error')
                    pause(ds.name, 'local training failed: see training_local.log')
                    return
    threading.Thread(target=chain, daemon=True).start()
    ds.log('autopilot', 'local training started')


def _ssh(cl, command, **kw):
    """Run a command on the cluster with the user's ssh key; never prompts."""
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', cl['host'], command],
                          check=True, timeout=kw.pop('timeout', 600), **kw)


def _submit_cluster(ds):
    cl = ds.config()['cluster']
    root = cl['remote_root'].rstrip('/')
    ws = f"{root}/d2d/workspace"
    _set(ds, phase='training', message='copying data to the cluster')
    # Stream the dataset and its training images up as one tar over ssh.
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as tar:
        for rel in (f'dataset/{ds.name}', f'results/{ds.name}/scaled_512', f'results/{ds.name}/scaled_512_metadata.csv',
                    f'results/{ds.name}/d2d/training'):
            p = ROOT / rel
            if p.exists():
                tar.add(p, arcname=rel)
    _ssh(cl, f'mkdir -p {shlex.quote(ws)} && tar xf - -C {shlex.quote(ws)}', input=buf.getvalue(), timeout=3600)
    t = f'd2d/workspace/results/{ds.name}/d2d/training'
    _ssh(cl, f'cd {shlex.quote(root)} && git pull --ff-only -q || true; '
             f'mkdir -p d2d/workspace/results/training/slurm && '
             f'jid=$(sbatch --parsable {t}/train_litevae.sbatch) && '
             f'sbatch --dependency=afterok:$jid {t}/train_diffusion.sbatch')
    ds.log('autopilot', 'LiteVAE and diffusion jobs submitted on the cluster (diffusion waits for the LiteVAE)')


def _pull_cluster(ds):
    """Bring back training curves, previews and best checkpoints (not every periodic one)."""
    cl = ds.config()['cluster']
    ws = f"{cl['remote_root'].rstrip('/')}/d2d/workspace"
    cmd = (f'cd {shlex.quote(ws)} && tar cf - --exclude="*.pt" results/training/*{ds.name}* 2>/dev/null; '
           f'true')
    out = _ssh(cl, cmd, capture_output=True).stdout
    if out:
        with tarfile.open(fileobj=io.BytesIO(out), mode='r:') as tar:
            tar.extractall(ROOT, filter='data')
    out = _ssh(cl, f'cd {shlex.quote(ws)} && tar cf - models/*{ds.name}*/*_best.pt 2>/dev/null; true',
               capture_output=True, timeout=1800).stdout
    if out:
        with tarfile.open(fileobj=io.BytesIO(out), mode='r:') as tar:
            tar.extractall(ROOT, filter='data')
