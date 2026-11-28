# The AI behind each stage's decision, with three interchangeable backends.
#
#   rules        no AI: every stage has a rule-based default decision. Works
#                offline and is what the other two fall back to.
#   api          the Claude API (Anthropic SDK), billed to an API key. Runs
#                unattended, so the dashboard can drive a whole stage.
#   claude-code  your own Claude Code, signed in with your Claude subscription
#                (Pro or Max). Anthropic does not allow third-party apps to offer
#                claude.ai sign-in, so d2d never signs in for you: it writes the
#                stage's question to a task file, and you run `/d2d` in Claude Code
#                in this folder. The skill hands each task to its agent
#                (.claude/agents/d2d-<stage>.md), which writes the decision back.
#
# Keeping token use low (the order things are tried in):
#   1. no call at all: a stage whose rule-based answer is unambiguous skips the
#      agent (Stage.needs_agent), and an identical question asked before is
#      answered from the decision cache (results/.d2d_cache/);
#   2. a short digest instead of the full report, with retrieval tools
#      (retrieval.py) to pull a report section, table rows or one figure on demand;
#   3. a stable prefix (agent instructions, then tools) marked for prompt
#      caching, so the turns of a tool loop and repeat runs read it from cache;
#   4. images capped at 1024 px (about 1,400 tokens each), only the ones the
#      stage lists up front, the rest on request;
#   5. effort set per stage: low for sizing and statistics, higher only where the
#      judgement is genuinely hard (Stage.effort).
# Every call's token use is appended to results/<dataset>/d2d/usage.jsonl.
import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path

from . import retrieval, settings
from .project import APP_DIR, ROOT, read_json, write_json

# The model and API key come from the Settings page (d2d/settings.py); D2D_MODEL and
# ANTHROPIC_API_KEY in the environment override them.
AGENTS_DIR = APP_DIR / '.claude' / 'agents'
CACHE_DIR = ROOT / '.d2d_cache'
BACKENDS = ('rules', 'api', 'claude-code')
MAX_TOOL_TURNS = 8
IMAGE_MAX_SIDE = 1024


def agent_instructions(stage):
    """The body of .claude/agents/d2d-<stage>.md, without its front matter."""
    path = AGENTS_DIR / f'd2d-{stage}.md'
    text = path.read_text(encoding='utf-8') if path.exists() else ''
    if text.startswith('---'):
        text = text.split('---', 2)[2]
    return text.strip()


def resolve_backend(name=None):
    name = name or os.environ.get('D2D_BACKEND') or settings.get()['backend']
    if name not in BACKENDS:
        raise SystemExit(f'backend must be one of {BACKENDS}, not {name!r}')
    return name


class Pending(Exception):
    """The decision has been handed to a Claude Code session and is not back yet."""


def cache_key(stage, prompt, schema, images, effort):
    h = hashlib.sha256()
    for part in (stage, settings.model(), effort, prompt, json.dumps(schema, sort_keys=True)):
        h.update(part.encode())
    for p in images:
        h.update(Path(p).read_bytes() if Path(p).exists() else str(p).encode())
    return h.hexdigest()[:24]


def decide(ds, stage, prompt, images, schema, default, backend=None, effort='medium', use_cache=True):
    """Ask for the stage's decision. Returns (decision, source).

    source says where the decision came from: 'rules', 'cache', 'api:<model>' or
    'claude-code'. Raises Pending for the claude-code backend until the decision
    file exists.
    """
    backend = resolve_backend(backend)
    if backend == 'rules':
        return default, 'rules'
    key = cache_key(stage, prompt, schema, images, effort)
    cached = read_json(CACHE_DIR / f'{key}.json') if use_cache else None
    if cached is not None:
        return cached, 'cache'
    if backend == 'api':
        decision, source = _decide_api(ds, stage, prompt, images, schema, default, effort)
    else:
        decision, source = _decide_claude_code(ds, stage, prompt, images, schema, default)
    if source != 'rules':
        write_json(CACHE_DIR / f'{key}.json', decision)
    return decision, source


def log_usage(ds, stage, source, usage):
    ds.state_dir.mkdir(parents=True, exist_ok=True)
    with open(ds.state_dir / 'usage.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps({'t': time.time(), 'stage': stage, 'source': source, **usage}) + '\n')


# --- Claude API ------------------------------------------------------------------

def _image_block(path, max_side=IMAGE_MAX_SIDE):
    from PIL import Image
    im = Image.open(path).convert('RGB')
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format='PNG')
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                                        'data': base64.standard_b64encode(buf.getvalue()).decode()}}


def _run_tool(ds, stage, name, args):
    """Execute one retrieval tool. Returns tool_result content."""
    if name == 'get_report':
        return retrieval.get(ds, stage, args.get('pointer', ''))
    if name == 'list_artifacts':
        return retrieval.artifacts(ds, stage)
    if name == 'get_table':
        return retrieval.table(ds, stage, args.get('name', ''), min(int(args.get('n', 10)), 40))
    if name == 'view_figure':
        p = retrieval.figure_path(ds, stage, args.get('name', ''))
        if p is None:
            return f'no figure {args.get("name")!r}; call list_artifacts'
        return [_image_block(p)]
    return f'unknown tool {name!r}'


def _decide_api(ds, stage, prompt, images, schema, default, effort):
    import anthropic
    key, _ = settings.api_key()
    # The saved or environment key; without one, the SDK's own lookup (an `ant auth login` profile).
    client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
    model = settings.model()
    content = []
    for p in images:
        content.append({'type': 'text', 'text': f'Figure: {Path(p).name}'})
        content.append(_image_block(p))
    content.append({'type': 'text', 'text': prompt})
    messages = [{'role': 'user', 'content': content}]
    # Stable prefix first (tools, then the agent's instructions), cached.
    system = [{'type': 'text', 'text': agent_instructions(stage) or f'You are the d2d {stage} agent.',
               'cache_control': {'type': 'ephemeral'}}]
    totals = {'input_tokens': 0, 'output_tokens': 0, 'cache_read_input_tokens': 0,
              'cache_creation_input_tokens': 0, 'turns': 0}
    response = None
    for _ in range(MAX_TOOL_TURNS):
        try:
            response = client.beta.messages.create(
                model=model,
                max_tokens=16000,
                betas=['server-side-fallback-2026-07-01'],
                fallbacks='default',               # a refused request is re-run on a fallback model
                thinking={'type': 'adaptive'},
                output_config={'effort': effort,
                               'format': {'type': 'json_schema', 'schema': schema}},
                cache_control={'type': 'ephemeral'},   # also cache the growing conversation
                system=system,
                tools=retrieval.API_TOOLS,
                messages=messages,
            )
        except anthropic.AuthenticationError:
            raise SystemExit('The Claude API rejected the credentials. Enter a key on the Settings page, '
                             'or switch the agent to your Claude subscription (Claude Code).')
        except anthropic.RateLimitError as e:
            raise SystemExit(f'Rate limited by the Claude API: {e.message}')
        except anthropic.APIStatusError as e:
            raise SystemExit(f'Claude API error {e.status_code}: {e.message}')
        except anthropic.APIConnectionError:
            raise SystemExit('Could not reach the Claude API (network).')
        u = response.usage
        for k in totals:
            if k != 'turns':
                totals[k] += getattr(u, k, 0) or 0
        totals['turns'] += 1
        if response.stop_reason != 'tool_use':
            break
        messages.append({'role': 'assistant', 'content': response.content})
        results = []
        for block in response.content:
            if block.type == 'tool_use':
                out = _run_tool(ds, stage, block.name, block.input if isinstance(block.input, dict) else {})
                results.append({'type': 'tool_result', 'tool_use_id': block.id, 'content': out})
        messages.append({'role': 'user', 'content': results})
    log_usage(ds, stage, f'api:{model}', totals)
    if response is None or response.stop_reason in ('refusal', 'max_tokens', 'tool_use'):
        why = {'refusal': 'The model declined', 'max_tokens': 'The answer was cut off',
               'tool_use': 'The agent used up its retrieval turns'}.get(getattr(response, 'stop_reason', ''), 'No answer')
        return dict(default, notes=f'{why}; the rule-based decision is used.'), 'rules'
    text = next((b.text for b in response.content if b.type == 'text'), None)
    if text is None:
        return dict(default, notes='No answer came back; the rule-based decision is used.'), 'rules'
    return json.loads(text), f'api:{response.model}'


# --- Claude Code (your subscription) -----------------------------------------------

def task_path(ds, stage):
    return ds.stage_dir(stage) / 'task.json'


def _decide_claude_code(ds, stage, prompt, images, schema, default):
    decision_file = ds.stage_dir(stage) / 'decision.json'
    task = read_json(task_path(ds, stage))
    # An answer counts only for the question it answered: after a redo the analysis, and
    # with it the prompt, changes, and the stage has to be asked again.
    if task and task.get('status') == 'done' and decision_file.exists() and task.get('prompt') == prompt:
        return read_json(decision_file), 'claude-code'
    write_json(task_path(ds, stage), {
        'dataset': ds.name,
        'stage': stage,
        'agent': f'd2d-{stage}',
        'status': 'pending',
        'created': time.time(),
        'workspace': str(ds.root),
        'images': [str(Path(p).resolve()) for p in images],
        'decision_path': str(decision_file.resolve()),
        'retrieval': (f'python run.py query {ds.name} {stage} report <json-pointer> | '
                      f'artifacts | table <name> [n] | figure <name>'),
        'schema': schema,
        'default_decision': default,
        'prompt': prompt,
    })
    raise Pending(f'Waiting for Claude Code: open Claude Code in {APP_DIR} and run /d2d')


def pending_tasks(root=ROOT):
    out = []
    for p in sorted((Path(root) / 'results').glob('*/d2d/*/task.json')):
        t = read_json(p)
        if t and t.get('status') == 'pending':
            out.append((p, t))
    return out


def validate(decision, schema, path='decision'):
    """Minimal JSON-schema check (types, required, enums, no extra keys)."""
    errors = []
    t = schema.get('type')
    types = {'object': dict, 'array': list, 'string': str, 'boolean': bool,
             'integer': int, 'number': (int, float), 'null': type(None)}
    allowed = t if isinstance(t, list) else [t] if t else []
    if allowed and not any(isinstance(decision, types[a]) and not (a in ('integer', 'number') and
                                                                  isinstance(decision, bool))
                           for a in allowed):
        return [f'{path}: expected {t}, got {type(decision).__name__}']
    if 'enum' in schema and decision not in schema['enum']:
        errors.append(f'{path}: {decision!r} is not one of {schema["enum"]}')
    if isinstance(decision, dict):
        props = schema.get('properties', {})
        for k in schema.get('required', []):
            if k not in decision:
                errors.append(f'{path}: missing {k!r}')
        if schema.get('additionalProperties') is False:
            for k in decision:
                if k not in props:
                    errors.append(f'{path}: unexpected key {k!r}')
        for k, v in decision.items():
            if k in props:
                errors += validate(v, props[k], f'{path}.{k}')
    if isinstance(decision, list) and 'items' in schema:
        for i, v in enumerate(decision):
            errors += validate(v, schema['items'], f'{path}[{i}]')
    return errors
