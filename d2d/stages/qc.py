# Quality control across the whole run: inputs, training, and outputs.
#
#   inputs    what the image stage excluded and why
#   training  every LiteVAE and diffusion run of this dataset, from its history
#             file (qc_checks.py): spikes, collapse, divergence, memorisation,
#             overfitting, convergence
#   outputs   the latest training previews and any generated set, against real
#             training images: flat, off-brightness, mode collapse, copies
#
# The agent is only consulted when a check fires: it looks at the previews next
# to real images and says what went wrong and what to do (stop and restart from
# an epoch, drop images, pick another checkpoint).
import pandas as pd

from .. import qc_checks as Q
from ..imaging import contact_sheet
from .base import NOTES, Stage, array, obj, string


class QC(Stage):
    name = 'qc'
    needs = ('images',)
    effort = 'medium'

    def analyze(self, ds):
        out = ds.stage_dir('qc')
        out.mkdir(parents=True, exist_ok=True)
        excluded = ds.stage_dir('images') / 'excluded.csv'
        try:
            inputs = pd.read_csv(excluded).to_dict('records')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            inputs = []
        runs = []
        meta = pd.read_csv(ds.scaled_metadata_path()) if ds.scaled_metadata_path().exists() else None
        real = [ds.scaled_dir() / f for f in meta.scaled_filename.sample(min(60, len(meta)), random_state=1)] if meta is not None else []
        figures = {}
        for run in Q.find_runs(ds):
            h = Q.load_history(run)
            issues = Q.check_litevae(h) if run['kind'] == 'litevae' else Q.check_diffusion(h)
            previews = Q.latest_previews(run)
            if previews:
                issues += Q.check_images(previews, real, train_real=real)
                figures[run['name']] = contact_sheet(previews + real[:6], out / f"{run['name']}_previews.png",
                                                     labels=[p.stem for p in previews] + ['real'] * len(real[:6]),
                                                     title=f"{run['name']}: latest previews, then real images")
            runs.append({'run': run['name'], 'kind': run['kind'], **Q.summary(h, run['kind']), 'issues': issues})
        gen = sorted((ds.results_dir / 'analysis' / 'generated').glob('*/*.png'))[:200]
        gen_issues = Q.check_images(gen, real, train_real=real) if gen else []
        return {'excluded_inputs': len(inputs), 'excluded_examples': inputs[:10], 'runs': runs,
                'generated_issues': gen_issues, '_figures': figures}

    def needs_agent(self, ds, report, default):
        bad = [i for r in report['runs'] for i in r['issues'] if i['severity'] in ('warning', 'fail')]
        bad += [i for i in report['generated_issues'] if i['severity'] in ('warning', 'fail')]
        return (True, f'{len(bad)} checks fired') if bad else (False, 'every check passed')

    def digest(self, ds, report):
        return report

    def attachments(self, ds, report):
        bad = {r['run'] for r in report['runs'] if any(i['severity'] != 'info' for i in r['issues'])}
        return [p for k, p in report.get('_figures', {}).items() if k in bad][:2]

    def schema(self, ds, report):
        return obj({
            'verdict': string(enum=['ok', 'warning', 'fail']),
            'issues': array(obj({'check': string(), 'severity': string(enum=['info', 'warning', 'fail']),
                                 'detail': string(), 'action': string()})),
            'recommended_checkpoint': string('Run and epoch to use, e.g. "diffusion_x_medium epoch 210"; '
                                             '"" if none is ready.'),
            'notes': NOTES,
        })

    def default_decision(self, ds, report):
        issues = [i for r in report['runs'] for i in r['issues']] + report['generated_issues']
        sev = {i['severity'] for i in issues}
        verdict = 'fail' if 'fail' in sev else 'warning' if 'warning' in sev else 'ok'
        diff = [r for r in report['runs'] if r['kind'] == 'diffusion' and r.get('best_epoch')]
        best = min(diff, key=lambda r: r['best_val_loss']) if diff else None
        return {'verdict': verdict, 'issues': issues,
                'recommended_checkpoint': f"{best['run']} epoch {best['best_epoch']}" if best else '',
                'notes': (f'{len(issues)} issues across {len(report["runs"])} runs; '
                          f'{report["excluded_inputs"]} input images excluded.')}

    def prompt(self, ds, report):
        return ('Review the training runs and their latest previews against real images. Say '
                'whether each problem is real, what caused it, and the concrete fix (stop and '
                'restart from an epoch, exclude images, choose another checkpoint).')

    def apply(self, ds, d):
        ds.update_config(qc={'verdict': d['verdict'], 'recommended_checkpoint': d['recommended_checkpoint']})
        return {'summary': f"QC {d['verdict']}: {len(d['issues'])} issues", 'verdict': d['verdict']}
