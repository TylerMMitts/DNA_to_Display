# Runs every analysis on one root diffusion model and files the results together.
#
# Point cfg.checkpoint at any diffusion checkpoint and this runs the same set of
# analyses that were done on the first one-hot model - trait fidelity through the
# segmenter, the train-versus-held-out split, genotype memorisation, attention,
# SNP contribution and founder archetypes - writing all of it under
#     results/model_analysis/<checkpoint name>/
# so two models are never mixed in one folder.
#
# Each analysis is still its own script and still runs on its own. This file only
# redirects each one's checkpoint and output folder, and runs them in order.

import ast
import json
import multiprocessing
import sys
import time
from pathlib import Path

import pandas as pd

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import (
    CODE_DIR, DIFFUSION_ONEHOT_MODEL, MODEL_ANALYSIS_DIR, apply_overrides,
    resolve_input, resolve_output,
)


# Every analysis this can run, in the order they have to run in.
#
# Each entry names the script, the settings to redirect for one model, and any
# earlier step whose output it reads. Settings are built from the checkpoint and
# the model's folder, so everything a step writes - including caches - lands
# inside that folder.
#
# Four of these cache model-specific results: the sensitivity analyses cache a
# population sensitivity map, and the legacy numeric path caches a PCA. Those
# caches are pointed into the model's folder too. Left at their defaults, a
# second model would silently reuse the first model's cached values.
def build_steps(checkpoint, d):
    pca = d / 'pca.pkl'
    return {
        # Whether the genotype describes the plant or recalls it. Cheap, and the
        # one result that decides how to read every other number here.
        'memorization': ('latent_diffusion.analysis.analyze_genotype_memorization', {
            'checkpoint': checkpoint, 'output_dir': d / 'memorization',
        }, []),
        'snp_encoder': ('latent_diffusion.validation.test_snp_encoder', {
            'checkpoint': checkpoint, 'output_dir': d / 'snp_encoder',
        }, []),
        # Generates an image for every real one and measures both with the
        # segmenter. The slowest step, and the source of the next two.
        'genetic_fidelity': ('feature_segmentation.evaluation.genetic_fidelity_test', {
            'checkpoint': checkpoint, 'output_dir': d / 'genetic_fidelity', 'pca_cache': pca,
        }, []),
        # Splits the fidelity result by whether the genotype was trained on.
        # Also reads results/reconstruction_fidelity, which does not depend on
        # the diffusion model and so is shared rather than rerun per model.
        'train_vs_test': ('feature_segmentation.evaluation.train_vs_test_accuracy', {
            'checkpoint': checkpoint,
            'measurements': d / 'genetic_fidelity' / 'per_image_measurements.csv',
            'output_dir': d / 'train_vs_test',
        }, ['genetic_fidelity']),
        'latent_comparison': ('feature_segmentation.evaluation.latent_comparison', {
            'generated_dir': d / 'genetic_fidelity' / 'generated',
            'output_dir': d / 'latent_comparison',
        }, ['genetic_fidelity']),
        'attention': ('latent_diffusion.analysis.analyze_snp_attention', {
            'checkpoint': checkpoint, 'output_dir': d / 'attention', 'pca_cache': pca,
        }, []),
        'genotype_contribution': ('latent_diffusion.analysis.analyze_genotype_contribution', {
            'checkpoint': checkpoint, 'output_dir': d / 'genotype_contribution',
            'pca_cache': pca,
            'sensitivity_cache': d / 'genotype_contribution' / 'population_sensitivity.csv',
        }, []),
        'snp_spatial_contribution': ('latent_diffusion.analysis.analyze_snp_spatial_contribution', {
            'checkpoint': checkpoint, 'output_dir': d / 'snp_spatial_contribution',
            'pca_cache': pca,
            'sensitivity_cache': d / 'snp_spatial_contribution' / 'population_sensitivity.csv',
        }, []),
        'snp_ranking': ('latent_diffusion.analysis.rank_snp_contributions', {
            'checkpoint': checkpoint, 'output_dir': d / 'snp_ranking', 'pca_cache': pca,
            'sensitivity_cache': d / 'snp_ranking' / 'population_sensitivity.csv',
        }, []),
        'snp_diverse_maps': ('latent_diffusion.analysis.select_diverse_snp_maps', {
            'checkpoint': checkpoint, 'output_dir': d / 'snp_diverse_maps', 'pca_cache': pca,
            'sensitivity_cache': d / 'snp_diverse_maps' / 'population_sensitivity.csv',
        }, []),
        'snp_output_contribution': ('latent_diffusion.analysis.snp_output_contribution', {
            'checkpoint': checkpoint, 'output_dir': d / 'snp_output_contribution',
            'diverse_csv': d / 'snp_diverse_maps' / 'diverse_snp_details.csv',
        }, ['snp_diverse_maps']),
        'parent_archetypes': ('latent_diffusion.generation.generate_parent_archetypes', {
            'checkpoint': checkpoint, 'output_dir': d / 'parent_archetypes', 'pca_cache': pca,
        }, []),
        'founder_strategies': ('latent_diffusion.generation.founder_archetype_strategies', {
            'checkpoint': checkpoint, 'output_dir': d / 'founder_strategies', 'pca_cache': pca,
        }, []),
        # Side-by-side real and generated images for browsing. Off by default:
        # genetic_fidelity already generates an image for every real one, so
        # this repeats the slowest work in the pipeline for a gallery.
        'gallery': ('latent_diffusion.generation.generate_from_dataset', {
            'checkpoint': checkpoint, 'output_dir': d / 'gallery', 'pca_cache': pca,
        }, []),
    }


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
    import importlib
    importlib.import_module(module_name).main(overrides)


def main(overrides=None):
    # Edit these values, then run:
    #     python code/evaluate_diffusion_model.py
    # On a cluster, evaluate_diffusion_model_hellbender.sbatch runs this on a GPU.
    class cfg:
        # The model to analyse. Any root diffusion checkpoint.
        checkpoint = DIFFUSION_ONEHOT_MODEL
        output_root = MODEL_ANALYSIS_DIR

        # Comment a step out to skip it. Order matters where one step reads
        # another's output; the steps that depend on another say so above.
        steps = [
            'memorization',
            'snp_encoder',
            'genetic_fidelity',
            'train_vs_test',
            'latent_comparison',
            'attention',
            'genotype_contribution',
            'snp_spatial_contribution',
            'snp_ranking',
            'snp_diverse_maps',
            'snp_output_contribution',
            'parent_archetypes',
            'founder_strategies',
            # 'gallery',
        ]

        # Extra settings for individual steps, on top of the checkpoint and
        # output redirection. Useful for a quick first pass, e.g.
        #     {'genetic_fidelity': {'max_images': 40}}
        step_settings = {}

        # Carry on with the remaining steps when one fails, and report every
        # failure at the end. A step that needs a failed step is skipped.
        continue_on_error = True

    apply_overrides(cfg, overrides)

    checkpoint = resolve_input(cfg.checkpoint, 'diffusion checkpoint')

    import torch
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    epoch = ckpt.get('epoch')
    label = model_label(checkpoint, epoch)
    model_dir = resolve_output(Path(cfg.output_root) / label)
    model_dir.mkdir(parents=True, exist_ok=True)

    # What was analysed, kept with the results so the folder explains itself.
    info = {
        'checkpoint': str(checkpoint), 'label': label, 'epoch': epoch,
        'model_size': ckpt.get('model_size'), 'encoding': ckpt.get('encoding'),
        'unet_config': ckpt.get('unet_config'),
        'snp_encoder_config': ckpt.get('snp_encoder_config'),
        'n_held_out_genotypes': len(ckpt.get('val_genotypes', [])),
        'train_loss': ckpt.get('loss'), 'val_loss': ckpt.get('val_loss'),
        'genotype_gain_held_out_during_training': ckpt.get('genotype_gain_heldout'),
        'genotype_gain_trained_during_training': ckpt.get('genotype_gain_trained'),
    }
    del ckpt
    (model_dir / 'model_info.json').write_text(json.dumps(info, indent=2, default=str))

    print(f"Model:   {checkpoint.name}  (epoch {epoch})")
    print(f"Results: {model_dir}\n")

    all_steps = build_steps(checkpoint, model_dir)
    unknown = [s for s in cfg.steps if s not in all_steps]
    if unknown:
        raise SystemExit(f"unknown step(s) {unknown}; choose from {list(all_steps)}")
    unknown = [s for s in cfg.step_settings if s not in all_steps]
    if unknown:
        raise SystemExit(f"step_settings names unknown step(s) {unknown}")

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
    log, failed = [], set()

    for i, (name, (module, overrides, needs)) in enumerate(plan.items(), start=1):
        header = f"[{i}/{len(plan)}] {name}"
        broken = [n for n in needs if n in failed]
        if broken:
            print(f"{header}: skipped, needs {broken} which failed\n")
            log.append({'step': name, 'status': 'skipped', 'seconds': 0.0,
                        'note': f'needs {broken}'})
            continue

        print(f"{header}  ->  {overrides['output_dir'].relative_to(model_dir)}")
        start = time.time()
        process = context.Process(target=run_step, args=(module, overrides))
        process.start()
        process.join()
        seconds = time.time() - start

        status = 'ok' if process.exitcode == 0 else 'failed'
        print(f"{header}: {status} in {seconds / 60:.1f} min\n")
        log.append({'step': name, 'status': status, 'seconds': round(seconds, 1),
                    'note': '' if status == 'ok' else f'exit code {process.exitcode}'})
        pd.DataFrame(log).to_csv(model_dir / 'pipeline_log.csv', index=False)

        if status == 'failed':
            failed.add(name)
            if not cfg.continue_on_error:
                print("Stopping: continue_on_error is False")
                break

    frame = pd.DataFrame(log)
    frame.to_csv(model_dir / 'pipeline_log.csv', index=False)
    print(f"Finished {label}")
    for row in frame.itertuples():
        print(f"  {row.status:8s} {row.step:26s} {row.seconds / 60:6.1f} min  {row.note}")
    print(f"\nResults, model_info.json and pipeline_log.csv in {model_dir}")
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
