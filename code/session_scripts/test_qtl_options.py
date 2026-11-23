# Checks the QTL options without training: orthogonalised windows / hybrid
# layouts hide the 14 designed QTL from the encoder's inputs, the QTL-only
# projector gives exactly those 14, encoders build and round-trip.
import os
os.environ['WIDIV_DATASET'] = 'ril_v5'
import sys
sys.path.insert(0, 'E:/D2D/DNA_to_Display/code/widiv')
sys.path.insert(0, 'E:/D2D/DNA_to_Display/code')
import numpy as np, pandas as pd, torch
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_SNP_MAP, WIDIV_SPLIT
from widiv_snp_encoding import QTLOnlyProjector, DosagePCAEncoder, load_widiv_genotypes, qtl_columns
from widiv_data import load_split
from widiv_window_encoding import DosageWindowLayout, GatedDosageWindowEncoder, load_widiv_encoder

QT = 'E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/designed_qtl_markers.csv'
ck = torch.load('E:/D2D/DNA_to_Display/models/from_hellbender/diffusion_ril_v5_medium_random_windows200_pc32_sp1e-06_vaev3_epoch_800.pt',
                map_location='cpu', weights_only=False)
val = sorted(ck['val_genotypes'])
split = load_split(WIDIV_SPLIT)
train = sorted(set(split.index[split.split_random == 'train']) - set(val))
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
row_of = {g: i for i, g in enumerate(lines)}
fit_rows = [row_of[g] for g in train]
va_rows = [row_of[g] for g in val]
_, qcols = qtl_columns(snps, QT, WIDIV_SNP_MAP, 0)
print('QTL columns', len(qcols))
smap = pd.read_csv(WIDIV_SNP_MAP).set_index('snp').loc[snps]
lay = DosageWindowLayout(200, 32).fit(dosage[fit_rows], snps, smap.chr.to_numpy(), smap.pos.to_numpy(),
                                      orthogonalize_columns=qcols)
enc = GatedDosageWindowEncoder.for_layout(lay, embedding_dim=256, num_tokens=8, hidden_dim=512)
enc.set_layout(lay).eval()
print('encoder config has n_qtl:', enc.config().get('n_qtl'))


def leak(Zin):
    # Ridge from the encoder's whole standardised input -> each QTL's true standardised dosage.
    zq = lay.pca._standardize(dosage)[:, qcols]
    A = Zin
    n = len(fit_rows)
    K = A[fit_rows] @ A[fit_rows].T / A.shape[1]
    Kv = A[va_rows] @ A[fit_rows].T / A.shape[1]
    out = []
    for j in range(len(qcols)):
        y = zq[fit_rows, j]
        best = -9
        for lam in (0.01, 0.1, 1):
            al = np.linalg.solve(K + lam * np.eye(n), y - y.mean())
            p = y.mean() + Kv @ al
            t = zq[va_rows, j]
            best = max(best, 1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum())
        out.append(best)
    return np.round(out, 3)


with torch.no_grad():
    z_enc = enc.standardize(torch.as_tensor(dosage.astype(np.int8))).numpy()
print('encoder standardize == layout.standardized:', np.allclose(z_enc, lay.standardized(dosage), atol=1e-4))
print('QTL readable from orthogonalised SNPs, R^2 per QTL (validation):', leak(z_enc))
pcs = z_enc @ lay.pca.components_.T
print('QTL readable from the 32 PCs, max R^2:', leak(pcs).max())
plain = DosageWindowLayout(200, 32).fit(dosage[fit_rows], snps, smap.chr.to_numpy(), smap.pos.to_numpy(),
                                        verbose=False)
print('without orthogonalising, median R^2:', np.median(leak(plain.standardized(dosage))))
# Old checkpoints still load (no n_qtl buffers).
e_old, p_old = load_widiv_encoder(ck)
print('old windows checkpoint loads; n_qtl', e_old.n_qtl, '| layout n_qtl', p_old.n_qtl)
# Round trip of an orthogonalised layout + encoder.
fake = {'snp_projector': lay.state_dict(), 'snp_encoder_config': enc.config(), 'snp_encoder_state_dict': enc.state_dict()}
e2, p2 = load_widiv_encoder(fake)
x = torch.as_tensor(dosage[va_rows[:4]].astype(np.int8))
with torch.no_grad():
    print('round trip identical tokens:', torch.allclose(e2(x), enc(x)))
# QTL only.
qp = QTLOnlyProjector().fit(dosage[fit_rows], snps, qcols)
xq = qp.transform(dosage[va_rows])
print('QTL-only input', xq.shape, '| columns are the QTL:', sorted(np.asarray(snps)[qp.columns]) == sorted(pd.read_csv(QT).snp))
qe = DosagePCAEncoder(input_dim=qp.output_dim, embedding_dim=256, num_tokens=8, hidden_dim=512).eval()
fake = {'snp_projector': qp.state_dict(), 'snp_encoder_config': qe.config(), 'snp_encoder_state_dict': qe.state_dict()}
qe2, qp2 = load_widiv_encoder(fake)
with torch.no_grad():
    print('QTL-only round trip identical tokens:', torch.allclose(qe2(torch.tensor(qp2.transform(dosage[va_rows[:4]]))),
                                                                   qe(torch.tensor(xq[:4]))))
