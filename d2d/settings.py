# App-wide settings, edited on the dashboard's Settings page.
#
#   workspace/settings.json      everything except secrets
#   ~/.d2d/credentials.json      the Claude API key, outside the workspace so a training
#                                bundle, an export or a copied workspace never carries it
# ANTHROPIC_API_KEY in the environment still works and takes precedence over a saved key.
import json
import os
import shutil
import subprocess
from pathlib import Path

from .project import ROOT, read_json, write_json

SETTINGS = ROOT / 'settings.json'
CREDENTIALS = Path.home() / '.d2d' / 'credentials.json'
# One fixed model for every decision, so the pipeline's results never depend on a model
# choice. Changing it is a code change (and a new version of the pipeline), not a setting.
MODEL = 'claude-opus-5-5'
MODEL_LABEL = 'Claude Opus 5.5'
BACKENDS = ('rules', 'api', 'claude-code')
WHERE = ('ask', 'local', 'export')
DEFAULTS = {
    'backend': 'claude-code',          # the agent for every dataset: your Claude subscription
    'claude_code_auto': False,         # run `claude -p /d2d` by itself when a decision waits
    'claude_path': '',                 # the claude command, if it is not on PATH
    'training_where': 'ask',           # at import: ask, or always this computer / always export
}


def get():
    s = json.loads(json.dumps(DEFAULTS))
    saved = read_json(SETTINGS, {}) or {}
    for k, v in saved.items():
        if k in s:                  # keys from older versions (model, cluster) are ignored
            s[k] = v
    return s


def save(updates):
    """Merge updates into the settings after checking each value. Returns the settings."""
    s = get()
    for k, v in (updates or {}).items():
        if k == 'backend' and v not in BACKENDS:
            raise ValueError(f'backend must be one of {BACKENDS}')
        if k == 'training_where' and v not in WHERE:
            raise ValueError(f'training_where must be one of {WHERE}')
        if k in ('claude_code_auto',):
            v = bool(v)
        if k == 'claude_path':
            v = str(v or '').strip()
        if k not in DEFAULTS:
            raise ValueError(f'unknown setting {k!r}')
        s[k] = v
    write_json(SETTINGS, {k: s[k] for k in DEFAULTS})
    return s


def model():
    return MODEL


def api_key():
    """(key, source): the environment first, then the saved key."""
    if os.environ.get('ANTHROPIC_API_KEY'):
        return os.environ['ANTHROPIC_API_KEY'], 'environment'
    key = (read_json(CREDENTIALS, {}) or {}).get('anthropic_api_key')
    return (key, 'saved') if key else (None, None)


def set_api_key(key):
    key = (key or '').strip()
    if not key:
        CREDENTIALS.unlink(missing_ok=True)
        return
    if not key.startswith('sk-') or len(key) < 20 or any(c.isspace() for c in key):
        raise ValueError('that does not look like a Claude API key (they start with sk-)')
    CREDENTIALS.parent.mkdir(parents=True, exist_ok=True)
    write_json(CREDENTIALS, {'anthropic_api_key': key})
    try:
        os.chmod(CREDENTIALS, 0o600)       # owner only, where the OS supports it
    except OSError:
        pass


def claude_command():
    """The Claude Code command, or None."""
    p = get()['claude_path']
    if p and Path(p).exists():
        return p
    for name in ('claude', 'claude.cmd'):
        if found := shutil.which(name):
            return found
    npm = Path(os.environ.get('APPDATA', '')) / 'npm' / 'claude.cmd'
    return str(npm) if npm.exists() else None


def claude_code_status():
    cmd = claude_command()
    if not cmd:
        return {'found': False}
    try:
        v = subprocess.run([cmd, '--version'], capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        v = ''
    return {'found': True, 'path': cmd, 'version': v}


def public():
    """Settings for the dashboard: the key itself never leaves the server."""
    key, source = api_key()
    return {**get(), 'model': MODEL, 'model_label': MODEL_LABEL, 'backends': BACKENDS,
            'api_key': {'set': bool(key), 'source': source, 'hint': f'…{key[-4:]}' if key else '',
                        'stored_at': str(CREDENTIALS)},
            'claude_code': claude_code_status(), 'workspace': str(ROOT), 'settings_file': str(SETTINGS)}


def test_api():
    """A free call (list one model) to check the key. Returns (ok, message)."""
    import anthropic
    key, _ = api_key()
    if not key:
        return False, 'no API key set'
    try:
        anthropic.Anthropic(api_key=key).models.list(limit=1)
        return True, 'the key works'
    except anthropic.AuthenticationError:
        return False, 'the Claude API rejected this key'
    except anthropic.APIConnectionError:
        return False, 'could not reach the Claude API (network)'
    except anthropic.APIStatusError as e:
        return False, f'Claude API error {e.status_code}: {e.message}'
