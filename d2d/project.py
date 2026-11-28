# Where a d2d dataset lives on disk, and the pipeline's state for it.
#
# Everything d2d writes stays in its own workspace (d2d/workspace/ by default,
# or $D2D_WORKSPACE). It never writes to the project's dataset/, results/ or
# models/ folders and never edits the project's code: training runs the
# project's trainers through d2d/launch_trainer.py, which points them at the
# workspace in memory. Inside the workspace a dataset <name> uses the layout
# those trainers read:
#
#   dataset/<name>/d2d.json                     the dataset's configuration
#   dataset/<name>/metadata/image_index.csv     one row per source image
#   dataset/<name>/metadata/train_test_split.csv line, split_random
#   dataset/<name>/metadata/traits.csv          imported traits (optional)
#   dataset/<name>/genetic/genotypes_012.csv.gz line x marker, 0/1/2, -1 missing
#   dataset/<name>/genetic/snp_map.csv          snp, chr, pos
#   results/<name>/scaled_<size>/               training images
#   results/<name>/scaled_<size>_metadata.csv   scaled_filename, genotype, ...
#   results/<name>/d2d/                         pipeline state, reports, figures
#   models/                                     weights trained for d2d datasets
#
# State is plain JSON written atomically, so the dashboard, the CLI and a Claude
# Code session can all read it while another one writes it.
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent                 # d2d/
PROJECT_ROOT = APP_DIR.parent                             # the repository: code/ and models/ are read only
ROOT = Path(os.environ.get('D2D_WORKSPACE', APP_DIR / 'workspace')).resolve()

STAGES = ('intake', 'images', 'traits', 'genetics', 'training', 'qc', 'analysis')
STAGE_TITLES = {
    'intake': 'Organize the dataset',
    'images': 'Image preprocessing',
    'traits': 'Segmentation and traits',
    'genetics': 'Genetic encoding',
    'training': 'Training setup',
    'qc': 'Quality control',
    'analysis': 'Statistical analysis',
}
# Stage statuses, in the order a stage passes through them.
#   pending     not run yet
#   running     its deterministic analysis is running
#   deciding    waiting for the AI decision (API call in flight, or a task file
#               waiting for a Claude Code session)
#   review      decision ready, waiting for the user to approve it
#   applying    the approved decision is being carried out
#   done        finished
#   failed      stopped with an error (message in the stage state)


def _clean(x):
    """NaN and infinity become null, so browsers can parse every file."""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return _clean(x.item())
    except ImportError:
        pass
    return x


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Per thread: the autopilot, a stage and the dashboard can write the same file at once.
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(_clean(data), f, indent=2, default=_jsonable)
    # On Windows the replace fails while another thread or process has the file open for
    # reading (the dashboard polls state.json every few seconds); that lasts milliseconds.
    for attempt in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 39:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.05)


def read_json(path, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _jsonable(x):
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except ImportError:
        pass
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f'{type(x).__name__} is not JSON serialisable')


@dataclass
class Dataset:
    name: str
    root: Path = field(default_factory=lambda: ROOT)

    @property
    def dataset_dir(self):
        return self.root / 'dataset' / self.name

    @property
    def results_dir(self):
        return self.root / 'results' / self.name

    @property
    def state_dir(self):
        return self.results_dir / 'd2d'

    @property
    def config_path(self):
        return self.dataset_dir / 'd2d.json'

    @property
    def metadata_dir(self):
        return self.dataset_dir / 'metadata'

    @property
    def genetic_dir(self):
        return self.dataset_dir / 'genetic'

    @property
    def genotypes_path(self):
        return self.genetic_dir / 'genotypes_012.csv.gz'

    @property
    def snp_map_path(self):
        return self.genetic_dir / 'snp_map.csv'

    @property
    def image_index_path(self):
        return self.metadata_dir / 'image_index.csv'

    @property
    def split_path(self):
        return self.metadata_dir / 'train_test_split.csv'

    @property
    def traits_path(self):
        return self.metadata_dir / 'traits.csv'

    def scaled_dir(self, size=None):
        size = size or self.config().get('images', {}).get('target_size', 512)
        return self.results_dir / f'scaled_{size}'

    def scaled_metadata_path(self, size=None):
        size = size or self.config().get('images', {}).get('target_size', 512)
        return self.results_dir / f'scaled_{size}_metadata.csv'

    def stage_dir(self, stage):
        return self.state_dir / stage

    # Configuration --------------------------------------------------------------

    def config(self):
        return read_json(self.config_path, {})

    def update_config(self, **sections):
        cfg = self.config()
        for key, value in sections.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
        write_json(self.config_path, cfg)
        return cfg

    # Stage state ----------------------------------------------------------------

    def state(self):
        st = read_json(self.state_dir / 'state.json', {})
        for s in STAGES:
            st.setdefault(s, {'status': 'pending'})
        return st

    def set_stage(self, stage, **fields):
        st = self.state()
        st[stage].update(fields, updated=time.time())
        write_json(self.state_dir / 'state.json', st)
        if 'status' in fields:
            self.log(stage, f"status -> {fields['status']}"
                     + (f": {fields['message']}" if fields.get('message') else ''))
        return st[stage]

    def log(self, stage, message, level='info'):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_dir / 'events.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps({'t': time.time(), 'stage': stage, 'level': level,
                                'message': message}) + '\n')

    def events(self, limit=200):
        path = self.state_dir / 'events.jsonl'
        if not path.exists():
            return []
        lines = path.read_text(encoding='utf-8').splitlines()[-limit:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out

    def report(self, stage):
        return read_json(self.stage_dir(stage) / 'report.json')

    def decision(self, stage):
        return read_json(self.stage_dir(stage) / 'decision.json')


def list_datasets(root=ROOT):
    base = Path(root) / 'dataset'
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if (p / 'd2d.json').exists())
