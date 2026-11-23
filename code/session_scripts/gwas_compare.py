# Side-by-side MLM GWAS on generated line means: full e400 (reference), full / literal / strict
# e500, plus the real line means. Per set, over the 30 QTL on the five imaged traits: QTL found
# (Bonferroni hit within 1 Mb), false peaks (Bonferroni peaks near no QTL), inflation, the
# median -log10 p at the QTL SNP, how many QTL have p < 0.05, and how many estimated effects
# have the designed sign.
import numpy as np
import pandas as pd

G = 'E:/D2D/DNA_to_Display/results/widiv/gwas'
RUNS = {'full e400': 'random_vaev3_e400', 'full e500': 'random_vaev3_e500',
        'literal e500': 'random_noqtl0kb_vaev3_e500', 'strict e500': 'random_noqtl1000kb_orth_vaev3_e500'}
SKIP = ['ccfn', 'cort_cell_um']
rows = []
for lab, run in RUNS.items():
    s = pd.read_csv(f'{G}/{run}/summary.csv')
    q = pd.read_csv(f'{G}/{run}/qtl_recovery.csv')
    for src in ('real', 'generated'):
        if src == 'real' and lab != 'full e500':
            continue
        for st in ('all', 'training', 'test'):
            x = s[(s.source == src) & (s.set == st) & ~s.trait.isin(SKIP)]
            y = q[(q.source == src) & (q.set == st) & ~q.trait.isin(SKIP)]
            rows.append({'model': 'real data' if src == 'real' else lab, 'set': st, 'lines': int(x.lines.iloc[0]),
                         'qtl_found_of_30': int(x.qtl_found.sum()), 'false_peaks': int(x.peaks_near_no_qtl.sum()),
                         'inflation': f'{x.inflation.min():.2f}-{x.inflation.max():.2f}',
                         'median_log10p_at_qtl': round(float(np.median(-np.log10(y.p_at_qtl))), 2),
                         'qtl_p_below_0.05': int((y.p_at_qtl < 0.05).sum()),
                         'right_sign_of_30': int((np.sign(y.beta_per_allele) == np.sign(y.designed_per_allele)).sum())})
R = pd.DataFrame(rows)
order = {'all': 0, 'training': 1, 'test': 2}
R = R.sort_values(['set', 'model'], key=lambda c: c.map(order) if c.name == 'set' else c, kind='stable')
pd.set_option('display.width', 220)
print(R.to_string(index=False))
R.to_csv(f'{G}/noqtl_comparison/e500/gwas_comparison.csv', index=False)
for run in RUNS.values():
    p = pd.read_csv(f'{G}/{run}/peaks.csv')
    p = p[(p.source == 'generated') & ~p.trait.isin(SKIP)]
    print(f'\n{run} generated peaks:')
    print(p[['set', 'trait', 'snp', 'p', 'near']].to_string(index=False) if len(p) else '  none')
q = pd.read_csv(f'{G}/random_noqtl1000kb_orth_vaev3_e500/qtl_recovery.csv')
print('\nstrict e500 found QTL:'); print(q[q.found & (q.source == 'generated')][['set', 'trait', 'snp', 'min_p_within_1mb', 'beta_per_allele', 'designed_per_allele']].to_string(index=False))
