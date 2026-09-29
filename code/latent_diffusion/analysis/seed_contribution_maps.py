# Per-gene contribution maps for a seed diffusion model, in one command.
#
# Runs the same two scripts the seed pipeline runs, with the seed settings
# already filled in:
#   select_diverse_snp_maps.py   picks the genes whose attention patterns differ
#                                most and draws each one's attention maps
#   snp_output_contribution.py   regenerates kernels with each of those genes
#                                flipped and maps where the decoded kernel changes
# Both read the dataset from the checkpoint and load the seed SNP table and seed
# LiteVAE themselves; this file only points them at a seed checkpoint and a
# folder of its own.
#
# Writes, under results/seeds/contribution_maps/<checkpoint name>/:
#   snp_diverse_maps/diverse_snp_maps/   attention maps, one figure per gene
#   snp_output_contribution/             locus_comparison.png and maps/, one per gene
#
# evaluate_seed_diffusion_model.py runs the same two steps as part of the full
# seed analysis; this is for running them on their own.

import sys
from pathlib import Path

import torch

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import (
    SEED_DIFFUSION_MODEL, SEED_RESULTS_DIR, apply_overrides, pick_device,
    resolve_input, resolve_output,
)

from latent_diffusion.utils.dataset_inputs import checkpoint_dataset


def main(overrides=None):
    # Edit these values, then run:
    #     python code/latent_diffusion/analysis/seed_contribution_maps.py
    class cfg:
        # Any checkpoint train_seeds.py wrote.
        checkpoint = SEED_DIFFUSION_MODEL
        # None -> results/seeds/contribution_maps/<checkpoint name>/, so two
        # models never share a folder.
        output_dir = None

        # How many genes get a map.
        top_n = 5
        # Markers flipped together, starting at each chosen gene. On the root
        # model a single SNP moved the decoded image too little to localise and
        # its effect did not reproduce between genotype samples; 50 did. Set 1 to
        # see the single-SNP effect.
        block_size = 50
        # Genotypes each contribution map is averaged over. The run costs two
        # full generations per genotype per gene, so this sets the running time.
        n_genotypes = 8
        # A list of gene names for the contribution maps, e.g.
        # ['Zm00001eb000010']. None -> the genes the attention step picked. This
        # only steers the contribution maps; the attention step always makes its
        # own selection, so set run_attention_maps = False when naming genes.
        genes = None

        # Either step can be switched off, e.g. to redraw the contribution maps
        # without redoing the selection. The second reads the first's output.
        run_attention_maps = True
        run_contribution_maps = True

        device = pick_device()

    apply_overrides(cfg, overrides)

    checkpoint = resolve_input(cfg.checkpoint, 'seed diffusion checkpoint')
    dataset = checkpoint_dataset(checkpoint)
    if dataset != 'seeds':
        raise SystemExit(f"{checkpoint.name} is a {dataset} model; this runner is "
                         "for seed checkpoints. For a root model use "
                         "evaluate_diffusion_model.py, which runs the same two steps.")

    out = resolve_output(cfg.output_dir or
                         SEED_RESULTS_DIR / 'contribution_maps' / checkpoint.stem)
    out.mkdir(parents=True, exist_ok=True)
    diverse_dir = out / 'snp_diverse_maps'
    diverse_csv = diverse_dir / 'diverse_snp_details.csv'
    print(f"Seed checkpoint: {checkpoint}\nOutput: {out}\n")

    if cfg.run_attention_maps:
        from latent_diffusion.analysis import select_diverse_snp_maps
        print("=== 1/2  choosing genes and drawing their attention maps ===")
        select_diverse_snp_maps.main({
            'checkpoint': checkpoint,
            'output_dir': diverse_dir,
            'pca_cache': out / 'pca.pkl',
            'sensitivity_cache': diverse_dir / 'population_sensitivity.csv',
            'top_n': cfg.top_n,
            'device': cfg.device,
        })
        # Both steps load the model; the first one's copy is released before the
        # second loads its own, so they fit on one GPU in the same process.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if cfg.run_contribution_maps:
        from latent_diffusion.analysis import snp_output_contribution
        if not cfg.genes and not diverse_csv.exists():
            raise SystemExit(f"no {diverse_csv} to read the genes from - run with "
                             "run_attention_maps = True first, or name them in cfg.genes")
        print("\n=== 2/2  mapping where each gene changes the decoded kernel ===")
        settings = {
            'checkpoint': checkpoint,
            'output_dir': out / 'snp_output_contribution',
            'diverse_csv': diverse_csv,
            'top_n': cfg.top_n,
            'block_size': cfg.block_size,
            'n_genotypes': cfg.n_genotypes,
            'device': cfg.device,
        }
        if cfg.genes:
            settings['snp_names'] = list(cfg.genes)
        snp_output_contribution.main(settings)

    print(f"\nDone. Attention maps in {diverse_dir / 'diverse_snp_maps'}\n"
          f"      Contribution maps in {out / 'snp_output_contribution'}")


if __name__ == '__main__':
    main()
