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
# Rerunning for the same model skips every step whose folder already exists, so
# a newly added step is the only one that runs.

import sys
from pathlib import Path

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import DIFFUSION_ONEHOT_MODEL, MODEL_ANALYSIS_DIR, apply_overrides, resolve_input

# Checking, skipping, logging and the per-step processes, shared with the seed
# pipeline. The steps themselves are listed below.
from analysis_pipeline import prepare_model_dir, run_pipeline


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
        # Every held-out root beside seven generated images of its genotype, for
        # judging by eye what the model does with genotypes it never trained on.
        'held_out_seeds': ('latent_diffusion.generation.generate_held_out_seeds', {
            'checkpoint': checkpoint, 'output_dir': d / 'held_out_seeds', 'pca_cache': pca,
        }, []),
        # Side-by-side real and generated images for browsing. Off by default:
        # genetic_fidelity already generates an image for every real one, so
        # this repeats the slowest work in the pipeline for a gallery.
        'gallery': ('latent_diffusion.generation.generate_from_dataset', {
            'checkpoint': checkpoint, 'output_dir': d / 'gallery', 'pca_cache': pca,
        }, []),
    }


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
            'held_out_seeds',
            # 'gallery',
        ]

        # Skip a step whose output folder already exists for this model, so
        # rerunning after adding a step runs only that step. A step the last run
        # logged as failed or left unfinished is run again regardless, and so is
        # any step that reads the output of a step run this time.
        skip_completed = True

        # Steps to run again even though their folder exists, e.g. after
        # changing one of their settings: ['genetic_fidelity'].
        rerun = []

        # Extra settings for individual steps, on top of the checkpoint and
        # output redirection. Useful for a quick first pass, e.g.
        #     {'genetic_fidelity': {'max_images': 40}}
        step_settings = {}

        # Carry on with the remaining steps when one fails, and report every
        # failure at the end. A step that needs a failed step is skipped.
        continue_on_error = True

    apply_overrides(cfg, overrides)

    checkpoint = resolve_input(cfg.checkpoint, 'diffusion checkpoint')
    model_dir, label = prepare_model_dir(checkpoint, cfg.output_root, 'roots')
    run_pipeline(cfg, build_steps(checkpoint, model_dir), model_dir, label)


if __name__ == '__main__':
    main()
