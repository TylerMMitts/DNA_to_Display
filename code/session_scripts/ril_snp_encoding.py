# code/widiv/analyze_snp_encoding.py run on the RIL v5 checkpoints, unchanged
# except for what the RIL rules and files require:
#   - "test" in that script = the 198 validation lines here; the delivered test
#     lines are never scored (the sender forbids using them for anything)
#   - fitting lines = training side = the 1,780 trained lines
#   - QTL = the answer key's designed QTL at their nearest markers; targets = the
#     answer key's true genetic values (g_) and observed phenotypes (mean_), with
#     test-line rows dropped before reading
# Output: results/ril_v5/snp_encoding_analysis/
import os
os.environ['WIDIV_DATASET'] = 'ril_v5'
import sys
from pathlib import Path
sys.path.insert(0, 'E:/D2D/DNA_to_Display/code/widiv')
sys.path.insert(0, 'E:/D2D/DNA_to_Display/code')
import pandas as pd
import analyze_snp_encoding as A
from widiv_paths import WIDIV_RESULTS_DIR, WIDIV_SPLIT

SCRATCH = Path(__file__).parent
KEY = Path('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5')
CK = ['E:/D2D/DNA_to_Display/models/from_hellbender/diffusion_ril_v5_medium_random_vaev3_epoch_800.pt',
      'E:/D2D/DNA_to_Display/models/from_hellbender/diffusion_ril_v5_medium_random_windows200_pc32_sp1e-06_vaev3_epoch_800.pt']

split = pd.read_csv(WIDIV_SPLIT).set_index('line').split_random
nontest = lambda d: d[d.genotype.map(split) == 'train']

ph = nontest(pd.read_csv(KEY / 'phenotypes_with_blind_trait.csv').rename(columns={'line': 'genotype'}))
gv = nontest(pd.read_csv(KEY / 'genetic_values.csv').rename(columns={'line': 'genotype'}))
q = pd.read_csv(KEY / 'qtl_and_scan_baseline__qtl.csv')
q = q.rename(columns={'nearest_marker': 'snp', 'qtl': 'role'})[['snp', 'trait', 'role', 'pct_genetic_variance']]
for name, d in (('ril_nontest_phenotypes.csv', ph), ('ril_nontest_genetic_values.csv', gv), ('ril_qtl.csv', q)):
    d.to_csv(SCRATCH / name, index=False)

A.TRAITS = ['stele_diam_um', 'mx_n', 'mx_diam_um', 'cortex_width_um', 'rca_pct', 'mcs_layers', 'n_lacunae']
A.WIDIV_GENOTYPE_MEANS = SCRATCH / 'ril_nontest_phenotypes.csv'
A.WIDIV_GENETIC_VALUES = SCRATCH / 'ril_nontest_genetic_values.csv'
A.WIDIV_QTL = SCRATCH / 'ril_qtl.csv'
A.OUT_DIR = WIDIV_RESULTS_DIR / 'snp_encoding_analysis'
A.RETENTION_KS = [10, 32, 50, 100, 124, 200, 400, 800, 1200]

first = A.load_checkpoint(CK[0])
val = set(first['val_genotypes'])
trained = set(split.index[split == 'train']) - val
data = A.Data(trained, trained, val)          # fit = train side = trained; "test" = validation
A.log('covariance spectrum and per-SNP retention on the fitting lines')
spectrum = A.covariance_spectrum(data)
summaries = [A.analyze_checkpoint(Path(p), data, spectrum)[0] for p in CK]
pd.DataFrame(summaries).to_csv(A.OUT_DIR / 'checkpoint_summary.csv', index=False, float_format='%.4g')
print(A.run_alternatives(data, spectrum, 124).to_string(index=False))
