# Writes a copy of the root SNP table with the synthetic stele genes added.
#
# The original parquet is read and never written. The copy holds every original
# row unchanged plus block_size added genes per genotype, each carrying that
# genotype's stele code (synthetic_genes.py), in the same long format, so
# load_snp_data_from_parquet and everything downstream read it like the real
# table. After writing, the copy is read back and checked: the real genes must
# come out identical to the original, and the added ones as intended.
#
# Writes to dataset/synthetic_stele/:
#   MEMA_gene_matrix_synthetic_stele_k<block_size>.parquet
#   synthetic_codes_k<block_size>.csv    the code and measured stele per genotype
#   manifest_k<block_size>.json          what was built from what

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import (
    SNP_PARQUET, SYNTHETIC_STELE_DIR, SYNTHETIC_STELE_RESULTS_DIR, apply_overrides,
    resolve_input, resolve_output,
)

from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from synthetic_stele.synthetic_genes import NAME_PREFIX, gene_names, stele_codes


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main(overrides=None):
    # Edit these values, then run:
    #     python code/synthetic_stele/build_synthetic_snps.py
    class cfg:
        snp_parquet = SNP_PARQUET
        ranking_csv = SYNTHETIC_STELE_RESULTS_DIR / 'real_root_traits' / 'genotype_stele_ranking.csv'
        # None -> the block size block_size_sweep.py recommended.
        block_size = None
        sweep_summary = SYNTHETIC_STELE_RESULTS_DIR / 'block_size_sweep' / 'summary.json'
        output_dir = SYNTHETIC_STELE_DIR
        n_founders = 8
        seed = 0

    apply_overrides(cfg, overrides)

    k = cfg.block_size
    if k is None:
        k = json.loads(Path(resolve_input(cfg.sweep_summary, 'sweep summary')).read_text())[
            'recommended_block_size']
        print(f"Block size from the sweep: {k}")
    k = int(k)
    if k < 1:
        raise SystemExit("block_size must be at least 1")

    source = Path(resolve_input(cfg.snp_parquet, 'SNP parquet'))
    out = resolve_output(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f'MEMA_gene_matrix_synthetic_stele_k{k}.parquet'
    if target.resolve() == source.resolve():
        raise SystemExit("the output would overwrite the source table")
    before = sha256(source)

    original = pd.read_parquet(source)
    if original['gene_model'].str.startswith(NAME_PREFIX).any():
        raise SystemExit(f"the source already has {NAME_PREFIX} genes; point at the original table")

    # Codes come from genotype names as the loader strips them, but the copy has
    # to keep the IDs exactly as the source writes them.
    ids = original['ID'].drop_duplicates().tolist()
    names = [i.replace('_TC', '') for i in ids]
    codes, measured = stele_codes(resolve_input(cfg.ranking_csv, 'stele ranking'), names,
                                  cfg.n_founders, cfg.seed)

    genes = gene_names(k)
    added = pd.DataFrame({
        'gene_model': np.repeat(genes, len(ids)),
        'ID': np.tile(ids, k),
        'value': np.tile(codes, k).astype(original['value'].dtype),
    })
    pd.concat([original, added[original.columns]], ignore_index=True).to_parquet(target, index=False)

    # Read the copy back the way training will, and check it.
    sample_names, snp_names, matrix = load_snp_data_from_parquet(target)
    o_names, o_snps, o_matrix = load_snp_data_from_parquet(source)
    matrix, o_matrix = np.asarray(matrix), np.asarray(o_matrix)
    snp_names = list(snp_names)
    if list(sample_names) != list(o_names):
        raise SystemExit("genotype order changed between the source and the copy")
    if snp_names[:k] != genes:
        raise SystemExit("the added genes are not the first columns of the copy")
    if snp_names[k:] != list(o_snps) or not np.array_equal(matrix[:, k:], o_matrix):
        raise SystemExit("the real genes in the copy differ from the source")
    code_of = dict(zip(names, codes))
    expected = np.array([code_of[g] for g in sample_names])
    if not (matrix[:, :k] == expected[:, None]).all():
        raise SystemExit("the added genes do not carry the intended codes")
    if sha256(source) != before:
        raise SystemExit("the source table changed while building the copy")

    ranking = pd.read_csv(resolve_input(cfg.ranking_csv, 'stele ranking'))
    stele = dict(zip(ranking['genotype'], ranking['stele_diameter_mean']))
    pd.DataFrame({'genotype': names, 'code': codes.astype(int), 'measured': measured,
                  'stele_diameter_mean': [stele.get(g, np.nan) for g in names]}
                 ).sort_values(['code', 'stele_diameter_mean']).to_csv(
        out / f'synthetic_codes_k{k}.csv', index=False)

    per_code = {int(c): {'n_measured': int((measured & (codes == c)).sum()),
                         'stele_range_px': [float(min(stele[g] for g, cc, m in zip(names, codes, measured) if m and cc == c)),
                                            float(max(stele[g] for g, cc, m in zip(names, codes, measured) if m and cc == c))]}
                for c in range(1, cfg.n_founders + 1)}
    manifest = {
        'built': date.today().isoformat(),
        'block_size': k,
        'gene_names': f'{genes[0]} .. {genes[-1]}',
        'source_parquet': str(source),
        'source_sha256': before,
        'ranking_csv': str(cfg.ranking_csv),
        'n_genotypes': len(names),
        'n_with_measured_stele': int(measured.sum()),
        'unmeasured_genotypes': 'random code, seed %d - never trained on, no image' % cfg.seed,
        'codes': per_code,
        'output': str(target),
    }
    (out / f'manifest_k{k}.json').write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {target.name}: {len(original):,} original rows + {len(added):,} added "
          f"({k} genes x {len(ids)} genotypes)")
    print("Checked: source unchanged, real genes identical, added genes first and as intended")
    for c, v in per_code.items():
        print(f"  founder {c}: {v['n_measured']} measured lines, stele "
              f"{v['stele_range_px'][0]:.1f}-{v['stele_range_px'][1]:.1f} px")


if __name__ == '__main__':
    main()
