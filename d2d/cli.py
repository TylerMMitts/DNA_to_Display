# d2d command line. From the repository root:
#
#   python -m d2d new <name> --source <folder>     register a dataset
#   python -m d2d run <name> <stage|all> [--backend rules|api|claude-code] [--auto] [--force-agent]
#   python -m d2d approve <name> <stage>           apply a reviewed decision
#   python -m d2d status <name>
#   python -m d2d tasks                            Claude Code tasks waiting for /d2d
#   python -m d2d task-done <name> <stage>         check a decision written by Claude Code, continue
#   python -m d2d query <name> <stage> report <pointer> | artifacts | table <file> [n] | figure <file>
#   python -m d2d dashboard [--port 8765]
#   python -m d2d usage <name>                     tokens spent per stage
#   python -m d2d demo                             make a small synthetic dataset and register it
#
# From inside d2d/ (where Claude Code runs the agents), use `python run.py ...`.
import argparse
import json
import sys

from . import llm, retrieval
from .project import ROOT, STAGES, Dataset, list_datasets, read_json, write_json


def cmd_new(a):
    ds = Dataset(a.name)
    if ds.config_path.exists() and not a.force:
        raise SystemExit(f'{a.name} exists; use --force to start over')
    from pathlib import Path
    src = Path(a.source).resolve()
    if not src.exists():
        raise SystemExit(f'source folder not found: {src}')
    cfg = {'name': a.name, 'source': str(src),
           'training': {'allow_local_training': a.allow_local_training},
           'cluster': {k: v for k, v in (('host', a.cluster_host), ('remote_root', a.cluster_root),
                                         ('partition', a.partition), ('account', a.account),
                                         ('venv', a.venv)) if v}}
    write_json(ds.config_path, cfg)
    print(f'registered {a.name}: source {src}\nworkspace {ROOT}\nnext: python -m d2d run {a.name} intake')


def cmd_run(a):
    from .stages.base import run
    ds = Dataset(a.name)
    stages = STAGES if a.stage == 'all' else [a.stage]
    for s in stages:
        st = run(ds, s, a.backend, auto_approve=a.auto, force_agent=a.force_agent)
        print(f"{s:9s} {st['status']:9s} {st.get('message', '')}")
        if st['status'] != 'done' and a.stage == 'all':
            break


def cmd_approve(a):
    from .stages.base import approve
    decision = json.load(open(a.decision, encoding='utf-8')) if a.decision else None
    st = approve(Dataset(a.name), a.stage, decision)
    print(st['status'], st.get('message', ''))


def cmd_status(a):
    names = [a.name] if a.name else list_datasets()
    for n in names:
        ds = Dataset(n)
        state = ds.state()
        ap = state.get('autopilot') or {}
        flag = ' (paused)' if ap.get('paused') else ''
        print(f"== {n}   autopilot: {ap.get('phase', 'off')}{flag}  {ap.get('message', '')}")
        for s in STAGES:
            st = state.get(s, {'status': 'pending'})
            print(f"  {s:9s} {st['status']:9s} {st.get('message', '')}")


def cmd_tasks(a):
    tasks = llm.pending_tasks()
    if not tasks:
        print('no pending tasks')
    for p, t in tasks:
        print(f"{t['dataset']} {t['stage']} -> agent {t['agent']}  ({p})")


def cmd_task_done(a):
    from .stages.base import get_stage, resume
    ds = Dataset(a.name)
    task_file = llm.task_path(ds, a.stage)
    task = read_json(task_file)
    decision = read_json(ds.stage_dir(a.stage) / 'decision.json')
    if task is None or decision is None:
        raise SystemExit('no task or no decision.json for that stage')
    errors = llm.validate(decision, task['schema'])
    if errors:
        print('decision.json does not match the schema:\n  ' + '\n  '.join(errors[:20]))
        sys.exit(1)
    task['status'] = 'done'
    write_json(task_file, task)
    st = resume(ds, a.stage)
    print(f"accepted; {a.stage} is now {st['status']}")


def cmd_query(a):
    ds = Dataset(a.name)
    if a.what == 'report':
        print(retrieval.get(ds, a.stage, a.arg or ''))
    elif a.what == 'artifacts':
        print(retrieval.artifacts(ds, a.stage))
    elif a.what == 'table':
        print(retrieval.table(ds, a.stage, a.arg, int(a.n or 10)))
    elif a.what == 'figure':
        p = retrieval.figure_path(ds, a.stage, a.arg or '')
        print(p if p else 'no such figure; see artifacts')


def cmd_usage(a):
    ds = Dataset(a.name)
    path = ds.state_dir / 'usage.jsonl'
    rows = [json.loads(l) for l in open(path, encoding='utf-8')] if path.exists() else []
    tot = {}
    for r in rows:
        t = tot.setdefault(r['stage'], {'calls': 0, 'input': 0, 'output': 0, 'cache_read': 0})
        t['calls'] += 1
        t['input'] += r.get('input_tokens', 0)
        t['output'] += r.get('output_tokens', 0)
        t['cache_read'] += r.get('cache_read_input_tokens', 0)
    for s, t in tot.items():
        print(f"{s:9s} calls {t['calls']:3d}  input {t['input']:8d}  cached {t['cache_read']:8d}  output {t['output']:7d}")
    if not tot:
        print('no Claude API calls recorded (rules, cache and Claude Code decisions cost no API tokens)')


def cmd_dashboard(a):
    from .dashboard.server import serve
    serve(a.host, a.port)


def cmd_demo(a):
    from .demo import make_demo
    folder = make_demo()
    ns = argparse.Namespace(name=a.name, source=str(folder), force=True, allow_local_training=False,
                            cluster_host=None, cluster_root=None, partition=None, account=None, venv=None)
    cmd_new(ns)


def main(argv=None):
    p = argparse.ArgumentParser(prog='d2d', description='DNA to Display: agent-guided dataset preparation')
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('new')
    s.add_argument('name'); s.add_argument('--source', required=True); s.add_argument('--force', action='store_true')
    s.add_argument('--allow-local-training', action='store_true')
    for opt in ('cluster-host', 'cluster-root', 'partition', 'account', 'venv'):
        s.add_argument(f'--{opt}')
    s.set_defaults(fn=cmd_new)
    s = sub.add_parser('run')
    s.add_argument('name'); s.add_argument('stage', choices=list(STAGES) + ['all'])
    s.add_argument('--backend', choices=llm.BACKENDS); s.add_argument('--auto', action='store_true')
    s.add_argument('--force-agent', action='store_true')
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser('approve')
    s.add_argument('name'); s.add_argument('stage', choices=STAGES); s.add_argument('--decision')
    s.set_defaults(fn=cmd_approve)
    s = sub.add_parser('status'); s.add_argument('name', nargs='?'); s.set_defaults(fn=cmd_status)
    s = sub.add_parser('tasks'); s.set_defaults(fn=cmd_tasks)
    s = sub.add_parser('task-done'); s.add_argument('name'); s.add_argument('stage', choices=STAGES)
    s.set_defaults(fn=cmd_task_done)
    s = sub.add_parser('query')
    s.add_argument('name'); s.add_argument('stage', choices=STAGES)
    s.add_argument('what', choices=['report', 'artifacts', 'table', 'figure'])
    s.add_argument('arg', nargs='?'); s.add_argument('n', nargs='?')
    s.set_defaults(fn=cmd_query)
    s = sub.add_parser('usage'); s.add_argument('name'); s.set_defaults(fn=cmd_usage)
    s = sub.add_parser('dashboard'); s.add_argument('--port', type=int, default=8765)
    s.add_argument('--host', default='127.0.0.1'); s.set_defaults(fn=cmd_dashboard)
    s = sub.add_parser('demo'); s.add_argument('--name', default='demo'); s.set_defaults(fn=cmd_demo)
    a = p.parse_args(argv)
    a.fn(a)
