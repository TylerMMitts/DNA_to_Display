# Cheap retrieval for the agents: instead of pasting a whole report, tables and
# every figure into the prompt, an agent gets a short digest and pulls only what
# it needs, a few hundred tokens at a time:
#
#   get(pointer)          one value or subtree of the stage report, by JSON pointer
#   table(name, n)        the first n rows of a dataset table (or its columns only)
#   artifacts()           what figures and tables exist for this stage, by name
#   image(name)           one named figure, downscaled (API backend: an image block)
#
# The same functions back the Claude API tools (llm.py) and the
# `python -m d2d query` command the Claude Code agents call, so both backends
# retrieve the same way. Every answer is capped, so one call can never flood
# the context.
import json
from pathlib import Path

import pandas as pd

MAX_CHARS = 6000
ALLOWED_TABLE_DIRS = ('dataset', 'results')


def _cap(text, limit=MAX_CHARS):
    return text if len(text) <= limit else text[:limit] + f'\n... [{len(text) - limit} more characters cut]'


def get(ds, stage, pointer=''):
    """A JSON-pointer lookup in a stage report: '', 'tables/0/columns', ..."""
    node = ds.report(stage)
    if node is None:
        return f'no report for stage {stage!r}'
    # Git Bash on Windows turns a leading "/" into a path ("C:/Program Files/Git/x");
    # undo that, and accept pointers without the leading slash.
    pointer = pointer.replace(chr(92), '/')
    if ':/' in pointer and '/Git/' in pointer:
        pointer = pointer.split('/Git/', 1)[1]
    for part in [p for p in pointer.split('/') if p]:
        part = part.replace('~1', '/').replace('~0', '~')
        if isinstance(node, list):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                return f'no element {part!r} in a list of {len(node)}'
        elif isinstance(node, dict):
            if part not in node:
                return f'no key {part!r}; keys are {sorted(node)[:40]}'
            node = node[part]
        else:
            return f'{pointer!r} goes below a leaf value'
    if isinstance(node, dict):
        # Large subtrees come back as an outline the agent can drill into.
        text = json.dumps(node, indent=1, default=str)
        if len(text) > MAX_CHARS:
            outline = {k: (f'<{type(v).__name__} of {len(v)}>' if isinstance(v, (list, dict)) else v)
                       for k, v in node.items()}
            return json.dumps(outline, indent=1, default=str)
        return text
    return _cap(json.dumps(node, indent=1, default=str))


def artifacts(ds, stage):
    d = ds.stage_dir(stage)
    out = {'figures': sorted(p.name for p in d.glob('*.png')),
           'tables': sorted(p.name for p in d.glob('*.csv'))}
    return json.dumps(out, indent=1)


def figure_path(ds, stage, name):
    p = (ds.stage_dir(stage) / Path(name).name)
    return p if p.exists() and p.suffix.lower() in ('.png', '.jpg', '.jpeg') else None


def table(ds, stage, name, n=10, columns_only=False):
    """First rows of a stage table, or of a dataset table given as a project path."""
    p = ds.stage_dir(stage) / Path(name).name
    if not p.exists():
        cand = (ds.root / name).resolve()
        if cand.exists() and any(str(cand).startswith(str((ds.root / a).resolve())) for a in ALLOWED_TABLE_DIRS):
            p = cand
        else:
            return f'no table {name!r}'
    df = pd.read_csv(p, nrows=max(n, 1))
    if columns_only:
        return json.dumps({'columns': list(map(str, df.columns))})
    return _cap(df.head(n).to_csv(index=False))


# Tool definitions for the Claude API backend (client tools, strict schemas).
API_TOOLS = [
    {'name': 'get_report', 'strict': True,
     'description': 'Read one part of the full stage report by JSON pointer, e.g. "tables/0" or "". '
                    'Large objects come back as an outline; drill in with a longer pointer.',
     'input_schema': {'type': 'object', 'properties': {'pointer': {'type': 'string'}},
                      'required': ['pointer'], 'additionalProperties': False}},
    {'name': 'list_artifacts', 'strict': True,
     'description': 'List the figures and tables this stage produced.',
     'input_schema': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}},
    {'name': 'get_table', 'strict': True,
     'description': 'First n rows (max 40) of a stage table or a dataset/results CSV.',
     'input_schema': {'type': 'object', 'properties': {'name': {'type': 'string'},
                                                       'n': {'type': 'integer'}},
                      'required': ['name', 'n'], 'additionalProperties': False}},
    {'name': 'view_figure', 'strict': True,
     'description': 'Look at one figure produced by this stage (a contact sheet, overlay or plot).',
     'input_schema': {'type': 'object', 'properties': {'name': {'type': 'string'}},
                      'required': ['name'], 'additionalProperties': False}},
]
