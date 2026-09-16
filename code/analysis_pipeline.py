# The machinery both per-model analysis pipelines run on.
#
# evaluate_diffusion_model.py (roots) and evaluate_seed_diffusion_model.py
# (seeds) each list their own analyses; this file checks a plan of them before
# anything runs, runs each in its own process, skips steps already done for the
# model, and keeps the log. It holds no analyses and no paths of its own, so
# which dataset is being analysed is still decided by which pipeline file you run.

import ast
import importlib
import json
import multiprocessing
import time
from pathlib import Path

import pandas as pd

from paths import CODE_DIR, resolve_output


# The names a script's cfg block defines, read from source without importing it.
def cfg_settings(module_name):
    path = CODE_DIR / (module_name.replace('.', '/') + '.py')
    if not path.exists():
        return None, f"no file at {path}"
    tree = ast.parse(path.read_text(encoding='utf-8'))
    main = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main'), None)
    if main is None:
        return None, "has no main()"
    if [a.arg for a in main.args.args] != ['overrides']:
        return None, "main() does not accept overrides"
    cfg = next((n for n in main.body if isinstance(n, ast.ClassDef) and n.name == 'cfg'), None)
    if cfg is None:
        return None, "main() has no cfg block"
    return {t.id for n in cfg.body if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name)}, None


# Every problem with the plan, found before anything runs. A misnamed setting
# would otherwise only surface when its step started, which on a cluster can be
# hours into the job.
def check_plan(plan):
    problems = []
    for name, (module, overrides, _) in plan.items():
        settings, error = cfg_settings(module)
        if error:
            problems.append(f"{name}: {module} {error}")
            continue
        for key in overrides:
            if key not in settings:
                problems.append(f"{name}: {module} has no setting {key!r}")
    return problems


# A folder name that says exactly which weights were analysed. A _best.pt file is
# overwritten as training improves, so its epoch is added; otherwise analysing it
# again later would overwrite a different model's results under the same name.
def model_label(checkpoint_path, epoch):
    stem = checkpoint_path.stem
    if stem.endswith('_best') and epoch is not None:
        return f'{stem}_epoch{epoch}'
    return stem


# Runs in a child process. Top level so it can be sent to one.
def run_step(module_name, overrides):
    importlib.import_module(module_name).main(overrides)


# Makes the model's results folder and records what is being analysed in it.
#
# Root checkpoints predate the dataset field, so a checkpoint without one is a
# root model. Checked here because every step would otherwise fail on its own,
# one after another, over a checkpoint from the wrong dataset.
def prepare_model_dir(checkpoint, output_root, expected_dataset):
    import torch
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    dataset = ckpt.get('dataset', 'roots')
    if dataset != expected_dataset:
        raise SystemExit(f"{checkpoint.name} was trained on {dataset}, but this pipeline "
                         f"is for models trained on {expected_dataset}")

    epoch = ckpt.get('epoch')
    label = model_label(checkpoint, epoch)
    model_dir = resolve_output(Path(output_root) / label)
    model_dir.mkdir(parents=True, exist_ok=True)

    # What was analysed, kept with the results so the folder explains itself.
    info = {
        'checkpoint': str(checkpoint), 'label': label, 'dataset': dataset, 'epoch': epoch,
        'model_size': ckpt.get('model_size'), 'encoding': ckpt.get('encoding'),
        'unet_config': ckpt.get('unet_config'),
        'snp_encoder_config': ckpt.get('snp_encoder_config'),
        'n_held_out_genotypes': len(ckpt.get('val_genotypes', [])),
        'train_loss': ckpt.get('loss'), 'val_loss': ckpt.get('val_loss'),
        'genotype_gain_held_out_during_training': ckpt.get('genotype_gain_heldout'),
        'genotype_gain_trained_during_training': ckpt.get('genotype_gain_trained'),
    }
    (model_dir / 'model_info.json').write_text(json.dumps(info, indent=2, default=str))

    print(f"Model:   {checkpoint.name}  (epoch {epoch})")
    print(f"Results: {model_dir}\n")
    return model_dir, label


# Checks and runs the steps a pipeline's cfg selects, from all_steps, a dict of
# name -> (module, overrides, names of steps whose output it reads).
def run_pipeline(cfg, all_steps, model_dir, label):
    for field in ('steps', 'step_settings', 'rerun'):
        unknown = [s for s in getattr(cfg, field) if s not in all_steps]
        if unknown:
            raise SystemExit(f"{field} names unknown step(s) {unknown}; "
                             f"choose from {list(all_steps)}")

    plan = {}
    for name in cfg.steps:
        module, overrides, needs = all_steps[name]
        plan[name] = (module, {**overrides, **cfg.step_settings.get(name, {})}, needs)

    problems = check_plan(plan)
    if problems:
        print("The plan has problems, so nothing was run:")
        for p in problems:
            print(f"  {p}")
        raise SystemExit(1)
    print(f"Plan checked: {len(plan)} steps, every setting exists in its script\n")

    # A spawned process per step, rather than calling each main() here. Every
    # step loads its own models, so this releases the GPU memory in between,
    # and a step that crashes cannot take the rest of the run down with it.
    context = multiprocessing.get_context('spawn')
    log_path = model_dir / 'pipeline_log.csv'

    # The log carries over between runs, one row per step, so a step that was
    # not part of this run keeps the row from the run that did it.
    log = {}
    if log_path.exists():
        for row in pd.read_csv(log_path, keep_default_na=False).to_dict('records'):
            log[row['step']] = row

    def write_log():
        order = [s for s in all_steps if s in log]
        pd.DataFrame([log[s] for s in order]).to_csv(log_path, index=False)

    failed, ran = set(), set()

    for i, (name, (module, overrides, needs)) in enumerate(plan.items(), start=1):
        header = f"[{i}/{len(plan)}] {name}"
        broken = [n for n in needs if n in failed]
        if broken:
            print(f"{header}: skipped, needs {broken} which failed\n")
            log[name] = {'step': name, 'status': 'skipped', 'seconds': 0.0,
                         'note': f'needs {broken}'}
            write_log()
            continue

        # A folder alone is not proof a step finished: one that crashed or was
        # killed partway leaves a folder too. The log tells those apart - every
        # step is marked running before it starts, and only overwritten once it
        # ends - so the folder is trusted only when the log does not contradict it.
        output_dir = overrides['output_dir']
        has_output = output_dir.is_dir() and any(output_dir.iterdir())
        last_status = log.get(name, {}).get('status')
        rerun_reason = ('in rerun' if name in cfg.rerun
                        else f"last run {last_status}"
                        if last_status in ('failed', 'running', 'skipped')
                        else f"reads {[n for n in needs if n in ran]}, run this time"
                        if any(n in ran for n in needs) else None)
        if cfg.skip_completed and has_output and rerun_reason is None:
            print(f"{header}: already done, skipped\n")
            if last_status is None:
                log[name] = {'step': name, 'status': 'ok', 'seconds': 0.0,
                             'note': 'output folder existed before logging'}
                write_log()
            continue

        note = f"  ({rerun_reason})" if has_output and rerun_reason else ''
        print(f"{header}  ->  {output_dir.relative_to(model_dir)}{note}")
        log[name] = {'step': name, 'status': 'running', 'seconds': 0.0, 'note': ''}
        write_log()
        start = time.time()
        process = context.Process(target=run_step, args=(module, overrides))
        process.start()
        process.join()
        seconds = time.time() - start
        ran.add(name)

        status = 'ok' if process.exitcode == 0 else 'failed'
        print(f"{header}: {status} in {seconds / 60:.1f} min\n")
        log[name] = {'step': name, 'status': status, 'seconds': round(seconds, 1),
                     'note': '' if status == 'ok' else f'exit code {process.exitcode}'}
        write_log()

        if status == 'failed':
            failed.add(name)
            if not cfg.continue_on_error:
                print("Stopping: continue_on_error is False")
                break

    print(f"Finished {label}  (ran {len(ran)} of {len(plan)} steps this time)")
    for name in plan:
        if name in log:
            row = log[name]
            when = 'this run' if name in ran else 'earlier'
            print(f"  {row['status']:8s} {name:26s} {float(row['seconds']) / 60:6.1f} min"
                  f"  {when:8s}  {row['note']}")
    print(f"\nResults, model_info.json and pipeline_log.csv in {model_dir}")
    if failed:
        raise SystemExit(1)
