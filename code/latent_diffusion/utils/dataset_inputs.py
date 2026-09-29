# Which SNP table, image metadata and LiteVAE belong to a diffusion checkpoint.
#
# Root and seed models share the UNet, the SNP encoder and the analysis code,
# but not their data. Seeds are genotyped against B73v5 in a different parquet,
# read by a different loader, listed in their own metadata and decoded by a
# LiteVAE trained on kernels. train_seeds.py records dataset='seeds' in every
# checkpoint it writes, so the analyses read the dataset from there rather than
# being told it, and a seed latent can never be decoded through the root
# autoencoder by mistake. A checkpoint with no record is a root model, the same
# rule analysis_pipeline.prepare_model_dir applies.

import torch

from paths import (
    IMAGE_METADATA, LITEVAE_MODEL, SEED_LITEVAE_MODEL, SEED_SCALED_METADATA,
    SEED_SNP_PARQUET, SNP_PARQUET, resolve_input,
)


def checkpoint_dataset(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    dataset = ckpt.get('dataset', 'roots')
    if dataset not in ('roots', 'seeds'):
        raise SystemExit(f"{checkpoint_path} records dataset={dataset!r}, which "
                         "none of the analyses know how to read")
    return dataset


def snp_parquet_for(dataset):
    return SEED_SNP_PARQUET if dataset == 'seeds' else SNP_PARQUET


def litevae_for(dataset):
    return SEED_LITEVAE_MODEL if dataset == 'seeds' else LITEVAE_MODEL


# (metadata csv, filename column). The two datasets name the column differently.
def metadata_for(dataset):
    return ((SEED_SCALED_METADATA, 'filename') if dataset == 'seeds'
            else (IMAGE_METADATA, 'new_filename'))


# (sample names, SNP names, matrix) from the table this dataset was trained on.
# parquet overrides the default path, for a copy kept somewhere else.
def load_snp_table(dataset, parquet=None):
    from latent_diffusion.models.snp_encoder import (
        load_seed_snp_data_from_parquet, load_snp_data_from_parquet,
    )
    loader = (load_seed_snp_data_from_parquet if dataset == 'seeds'
              else load_snp_data_from_parquet)
    return loader(resolve_input(parquet or snp_parquet_for(dataset), f'{dataset} SNP parquet'))


# The frozen decoder this dataset's diffusion model was trained against. Both
# autoencoders are built with the same arguments; only the weights differ.
def load_decoder(dataset, device, latent_channels, checkpoint=None):
    from litevae.models import LiteVAEDecoder
    path = resolve_input(checkpoint or litevae_for(dataset), f'{dataset} LiteVAE checkpoint')
    ckpt = torch.load(path, map_location=device, weights_only=False)
    decoder = LiteVAEDecoder(latent_channels=latent_channels, output_channels=3,
                             base_channels=512, num_res_blocks=2)
    decoder.load_state_dict(ckpt['decoder_state_dict'])
    return decoder.to(device).eval()
