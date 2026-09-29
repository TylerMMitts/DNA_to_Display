# Runs every analysis on one seed diffusion model and files the results together.
#
# The seed counterpart of evaluate_diffusion_model.py. Point cfg.checkpoint at any
# checkpoint train_seeds.py wrote and this runs the seed analyses - genotype
# memorisation, kernel size and colour fidelity, and generated kernels for every
# held-out genotype - writing all of it under
#     results/seeds/model_analysis/<checkpoint name>/
# so two models are never mixed in one folder.
#
# The root segmenter has nothing to find in a kernel, so trait fidelity measures
# kernels from their pixels instead: size from the non-white mask at the common
# px/mm scale, colour from the pixels inside it. The per-gene maps and the
# sampling trajectory are the same scripts the root pipeline runs, reading the
# dataset from the checkpoint. Rerunning for the same model
# skips every step whose folder already exists, so a newly added step is the
# only one that runs.

import sys
from pathlib import Path

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import SEED_DIFFUSION_MODEL, SEED_MODEL_ANALYSIS_DIR, apply_overrides, resolve_input

# Checking, skipping, logging and the per-step processes, shared with the root
# pipeline. The steps themselves are listed below.
from analysis_pipeline import prepare_model_dir, run_pipeline


# Every analysis this can run, in order. Each entry names the script, the
# settings to redirect for one model, and any earlier step whose output it reads.
# Everything a step writes lands inside the model's folder.
def build_steps(checkpoint, d):
    pca = d / 'pca.pkl'
    return {
        # Whether the genotype describes the kernel or recalls it. Cheap, and the
        # one result that decides how to read every other number here.
        'memorization': ('latent_diffusion.analysis.analyze_seed_memorization', {
            'checkpoint': checkpoint, 'output_dir': d / 'memorization', 'pca_cache': pca,
        }, []),
        # Generates every genotype from several seeds and compares kernel size,
        # shape and colour with the real kernel, trained against held out, with
        # the LiteVAE reconstruction as the ceiling. The slowest step.
        'kernel_traits': ('kernel_traits.kernel_trait_fidelity', {
            'checkpoint': checkpoint, 'output_dir': d / 'kernel_traits', 'pca_cache': pca,
        }, []),
        # Every held-out kernel beside seven generated kernels of its genotype,
        # for judging by eye what the model does with genotypes it never saw.
        'held_out_kernels': ('latent_diffusion.generation.generate_held_out_kernels', {
            'checkpoint': checkpoint, 'output_dir': d / 'held_out_kernels', 'pca_cache': pca,
        }, []),
        # The reverse process for trained and held-out genotypes: the latent at
        # each shown step on top, decoded underneath.
        'sampling_trajectory': ('latent_diffusion.analysis.sampling_trajectory', {
            'checkpoint': checkpoint, 'output_dir': d / 'sampling_trajectory', 'pca_cache': pca,
        }, []),
        # Per-gene maps, the same two scripts the root pipeline runs. Both read
        # the dataset from the checkpoint, so they pick up the seed SNP table and
        # the seed LiteVAE themselves. The first picks loci whose attention
        # patterns differ most and draws each one's attention maps; the second
        # regenerates kernels with each locus flipped and maps where the decoded
        # kernel changes.
        'snp_diverse_maps': ('latent_diffusion.analysis.select_diverse_snp_maps', {
            'checkpoint': checkpoint, 'output_dir': d / 'snp_diverse_maps', 'pca_cache': pca,
            'sensitivity_cache': d / 'snp_diverse_maps' / 'population_sensitivity.csv',
        }, []),
        # A run of 50 markers is flipped rather than one: on the root model a
        # single SNP moved the decoded image too little to localise, and the
        # tissue split it gave did not reproduce between genotype samples. The
        # root segmenter has nothing to find in a kernel, so there is no tissue
        # breakdown or tissue screen here.
        'snp_output_contribution': ('latent_diffusion.analysis.snp_output_contribution', {
            'checkpoint': checkpoint, 'output_dir': d / 'snp_output_contribution',
            'diverse_csv': d / 'snp_diverse_maps' / 'diverse_snp_details.csv',
            'block_size': 50, 'top_n': 5,
        }, ['snp_diverse_maps']),
    }


def main(overrides=None):
    # Edit these values, then run:
    #     python code/evaluate_seed_diffusion_model.py
    # On a cluster, evaluate_seed_diffusion_model_hellbender.sbatch runs this on a GPU.
    class cfg:
        # The model to analyse. Any checkpoint train_seeds.py wrote.
        checkpoint = SEED_DIFFUSION_MODEL
        output_root = SEED_MODEL_ANALYSIS_DIR

        # Comment a step out to skip it.
        steps = [
            'memorization',
            'kernel_traits',
            'held_out_kernels',
            'sampling_trajectory',
            'snp_diverse_maps',
            'snp_output_contribution',
        ]

        # Skip a step whose output folder already exists for this model, so
        # rerunning after adding a step runs only that step. A step the last run
        # logged as failed or left unfinished is run again regardless, and so is
        # any step that reads the output of a step run this time.
        skip_completed = True

        # Steps to run again even though their folder exists, e.g. after
        # changing one of their settings: ['kernel_traits'].
        rerun = []

        # Extra settings for individual steps, on top of the checkpoint and
        # output redirection. Useful for a quick first pass, e.g.
        #     {'kernel_traits': {'max_genotypes': 60}}
        step_settings = {}

        # Carry on with the remaining steps when one fails, and report every
        # failure at the end. A step that needs a failed step is skipped.
        continue_on_error = True

    apply_overrides(cfg, overrides)

    checkpoint = resolve_input(cfg.checkpoint, 'seed diffusion checkpoint')
    model_dir, label = prepare_model_dir(checkpoint, cfg.output_root, 'seeds')
    run_pipeline(cfg, build_steps(checkpoint, model_dir), model_dir, label)


if __name__ == '__main__':
    main()
