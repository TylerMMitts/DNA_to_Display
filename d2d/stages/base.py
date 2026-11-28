# The life of one pipeline stage, the same for every stage:
#
#   analyze   deterministic: measure the data, write report.json and figures
#   decide    the agent (rules / Claude API / Claude Code) reads the report and
#             answers the stage's question as JSON matching the stage's schema;
#             anything that fails the schema falls back to the rule-based answer
#   review    the decision waits for the user's approval (dashboard or CLI),
#             unless the run was started with auto-approve
#   apply     deterministic: carry out the approved decision
#
# The AI never touches the data directly. It reads reports and contact sheets and
# returns settings; code applies them. That keeps every run reproducible from its
# decision.json, and lets a person edit a decision before it is applied.
import time
import traceback

from .. import llm
from ..project import Dataset, STAGES, read_json, write_json


class Stage:
    name = ''
    needs = ()           # stages that must be done first
    effort = 'medium'    # Claude API effort for this stage's decision (low/medium/high)

    def analyze(self, ds: Dataset) -> dict:
        raise NotImplementedError

    def needs_agent(self, ds: Dataset, report: dict, default: dict):
        """(True, why) when the rule-based answer is uncertain enough to ask the
        agent; (False, why) to skip the call and save its tokens. Stages override
        this with their own confidence checks; the base asks every time."""
        return True, 'no confidence check for this stage'

    def digest(self, ds: Dataset, report: dict) -> dict:
        """The short version of the report the agent sees up front; the rest is
        one retrieval call away. Stages override this to pick what matters."""
        return _trim(report, max_list=8, max_str=300)

    def schema(self, ds: Dataset, report: dict) -> dict:
        raise NotImplementedError

    def default_decision(self, ds: Dataset, report: dict) -> dict:
        raise NotImplementedError

    def prompt(self, ds: Dataset, report: dict) -> str:
        """The question put to the agent; the report is appended as JSON."""
        return 'Decide the settings for this stage.'

    def attachments(self, ds: Dataset, report: dict) -> list:
        """Images the agent should look at (contact sheets, overlays, plots)."""
        return []

    def apply(self, ds: Dataset, decision: dict) -> dict:
        raise NotImplementedError


# --- schema helpers (strict JSON schema: every key required, nothing extra) -------

def obj(props, description=None):
    s = {'type': 'object', 'properties': props, 'required': list(props),
         'additionalProperties': False}
    if description:
        s['description'] = description
    return s


def string(description=None, enum=None):
    s = {'type': 'string'}
    if enum:
        s['enum'] = list(enum)
    if description:
        s['description'] = description
    return s


def number(description=None):
    return {'type': 'number', **({'description': description} if description else {})}


def integer(description=None):
    return {'type': 'integer', **({'description': description} if description else {})}


def boolean(description=None):
    return {'type': 'boolean', **({'description': description} if description else {})}


def array(items, description=None):
    return {'type': 'array', 'items': items, **({'description': description} if description else {})}


NOTES = string('Two to five sentences for the user: what you decided and why, and anything '
               'they should check.')


# --- running stages -------------------------------------------------------------

def get_stage(name):
    from . import REGISTRY
    if name not in REGISTRY:
        raise SystemExit(f'unknown stage {name!r}; stages are {", ".join(STAGES)}')
    return REGISTRY[name]


def run(ds, name, backend=None, auto_approve=False, force_agent=False):
    """Analyze, then decide. Returns the stage state."""
    stage = get_stage(name)
    state = ds.state()
    for need in stage.needs:
        if state[need]['status'] != 'done':
            return ds.set_stage(name, status='failed',
                                message=f'run the {need} stage first (it is {state[need]["status"]})')
    ds.set_stage(name, status='running', message='analysing', backend=llm.resolve_backend(backend))
    t0 = time.time()
    failed = _analyze_guarded(ds, name)
    if failed:
        return ds.set_stage(name, status='failed', message=failed)
    ds.log(name, f'analysis took {time.time() - t0:.0f} s')
    return decide(ds, name, backend, auto_approve, force_agent)


def _analyze_guarded(ds, name):
    """The stage's analysis in a watched child process (resources.run_guarded): a memory
    crash or a stall is retried at a lower memory level by itself. Returns None on
    success, else the message to show."""
    import sys
    from .. import resources
    from ..project import PROJECT_ROOT
    report_path = ds.stage_dir(name) / 'report.json'
    report_path.unlink(missing_ok=True)
    log = ds.state_dir / f'{name}_analysis.log'
    ds.state_dir.mkdir(parents=True, exist_ok=True)

    def retry(why, lvl):
        ds.log(name, f'analysis {why}; retrying by itself with less memory (level {lvl} of 2)', level='warning')
        ds.set_stage(name, status='running', message=f'analysing (retry with less memory: {why})')

    code, tail = resources.run_guarded([sys.executable, '-u', '-m', 'd2d.stage_worker', ds.name, name], log,
                                       on_retry=retry, cwd=PROJECT_ROOT)
    if code == 0 and report_path.exists():
        return None
    stop = [ln.split('D2D_STAGE_STOP: ', 1)[1] for ln in tail.splitlines() if 'D2D_STAGE_STOP: ' in ln]
    if stop:
        return stop[-1]
    ds.log(name, tail[-4000:], level='error')
    if resources.is_memory_problem(tail):
        return ('ran out of memory even with the least memory settings: close other programs, free space on '
                'the system drive (Windows grows its page file there), then redo this stage')
    last = [ln for ln in tail.strip().splitlines() if ln.strip()][-1:] or ['see the log']
    return f'analysis failed: {last[0][:300]} (full log: {log.name})'


def decide(ds, name, backend=None, auto_approve=False, force_agent=False):
    stage = get_stage(name)
    report = ds.report(name)
    if report is None:
        return ds.set_stage(name, status='failed', message='no report: run the stage first')
    schema = stage.schema(ds, report)
    default = stage.default_decision(ds, report)
    bad = llm.validate(default, schema)
    if bad:        # a bug in the stage, not in the agent
        raise RuntimeError(f'{name}: rule-based decision fails its own schema: {bad}')
    backend = llm.resolve_backend(backend)
    if backend != 'rules' and not force_agent:
        ask, why = stage.needs_agent(ds, report, default)
        if not ask:
            ds.log(name, f'agent skipped, rule-based answer is confident: {why}')
            backend = 'rules'
            # A task left from an earlier attempt at this stage no longer needs an answer.
            old = read_json(llm.task_path(ds, name))
            if old and old.get('status') == 'pending':
                write_json(llm.task_path(ds, name), dict(old, status='superseded', superseded_by='rules'))
    import json
    prompt = (stage.prompt(ds, report).strip()
              + '\n\nDigest of the analysis (more with the retrieval tools; ask only for what '
                'you need):\n' + json.dumps(stage.digest(ds, report), separators=(',', ':'), default=str)
              + '\n\nThe rule-based answer, which you may keep, change or replace:\n'
              + json.dumps(default, separators=(',', ':')))
    ds.set_stage(name, status='deciding', message='waiting for the agent')
    try:
        decision, source = llm.decide(ds, name, prompt, stage.attachments(ds, report), schema,
                                      default, backend, effort=stage.effort)
    except llm.Pending as p:
        return ds.set_stage(name, status='deciding', message=str(p))
    except SystemExit as e:
        return ds.set_stage(name, status='failed', message=str(e))
    errors = llm.validate(decision, schema)
    if errors:
        ds.log(name, 'agent decision rejected: ' + '; '.join(errors[:8]), level='warning')
        decision, source = dict(default, notes=default.get('notes', '') +
                                ' (The agent answer failed validation; this is the rule-based one.)'), 'rules'
    write_json(ds.stage_dir(name) / 'decision.json', decision)
    ds.set_stage(name, status='review', message=f'decision from {source}', decision_source=source)
    if auto_approve:
        return approve(ds, name)
    return ds.state()[name]


def resume(ds, name):
    """Pick up a decision written by a Claude Code session."""
    st = ds.state()[name]
    task = read_json(llm.task_path(ds, name))
    if st['status'] == 'deciding' and task and task.get('status') == 'done':
        return decide(ds, name, backend='claude-code')
    return st


def approve(ds, name, decision=None):
    stage = get_stage(name)
    if decision is not None:
        errors = llm.validate(decision, stage.schema(ds, ds.report(name)))
        if errors:
            return ds.set_stage(name, status='review', message='edited decision rejected: ' + '; '.join(errors[:5]))
        write_json(ds.stage_dir(name) / 'decision.json', decision)
    decision = ds.decision(name)
    if decision is None:
        return ds.set_stage(name, status='failed', message='no decision to apply')
    try:
        ds.set_stage(name, status='applying', message='applying the decision')
        t0 = time.time()
        result = stage.apply(ds, decision)
        write_json(ds.stage_dir(name) / 'result.json', result)
        ds.log(name, f'apply took {time.time() - t0:.0f} s')
    except SystemExit as e:
        return ds.set_stage(name, status='failed', message=str(e))
    except Exception as e:  # noqa: BLE001
        ds.log(name, traceback.format_exc(), level='error')
        return ds.set_stage(name, status='failed', message=f'{type(e).__name__}: {e}')
    return ds.set_stage(name, status='done', message=result.get('summary', 'done'))


def _trim(report, max_list=40, max_str=2000):
    """Keep the prompt short: long lists and strings are cut; keys starting
    with '_' (bulky detail meant for retrieval and the dashboard) are dropped."""
    if isinstance(report, dict):
        return {k: _trim(v, max_list, max_str) for k, v in report.items() if not str(k).startswith('_')}
    if isinstance(report, list):
        return [_trim(v, max_list, max_str) for v in report[:max_list]] + (
            [f'... {len(report) - max_list} more'] if len(report) > max_list else [])
    if isinstance(report, float):
        return round(report, 4)
    if isinstance(report, str) and len(report) > max_str:
        return report[:max_str] + '...'
    return report
