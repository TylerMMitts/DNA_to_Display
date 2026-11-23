# Checks the hybrid encoding without training: fit the projector on the RIL's
# training lines as train_widiv.py would, compare with the option-1 check, run one
# forward pass through encoder + UNet, round-trip a checkpoint dict.
import os
os.environ['WIDIV_DATASET'] = 'ril_v5'
import sys, time
sys.path.insert(0, 'E:/D2D/DNA_to_Display/code/widiv')
sys.path.insert(0, 'E:/D2D/DNA_to_Display/code')
import numpy as np, pandas as pd, torch
from PIL import Image
from widiv_paths import (WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_SCALED_DIR, WIDIV_SCALED_METADATA,
                         WIDIV_SNP_MAP, WIDIV_SPLIT)
from widiv_snp_encoding import load_widiv_genotypes
from widiv_data import load_split, choose_val_lines
from widiv_hybrid_encoding import HybridDosageEncoder, HybridLayout, tissue_layout_summary
from widiv_window_encoding import load_widiv_encoder
from latent_diffusion.models.unet import DenoisingUNet

t0 = time.time()
dev = torch.device('cuda')
split_table = load_split(WIDIV_SPLIT)
ck = torch.load('E:/D2D/DNA_to_Display/models/from_hellbender/diffusion_ril_v5_medium_random_windows200_pc32_sp1e-06_vaev3_epoch_800.pt',
                map_location='cpu', weights_only=False)
val = set(ck['val_genotypes'])
assert val == choose_val_lines(split_table, 'split_random', 0.1, 0), 'validation lines differ from train_widiv'
train = set(split_table.index[split_table.split_random == 'train']) - val
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
row_of = {g: i for i, g in enumerate(lines)}
fit_rows = [row_of[g] for g in sorted(train) if g in row_of]
meta = pd.read_csv(WIDIV_SCALED_METADATA, usecols=['scaled_filename', 'genotype'])
meta = meta[meta.genotype.isin(train)]
summ = {}
for f, g in zip(meta.scaled_filename, meta.genotype):
    rgb = torch.from_numpy(np.asarray(Image.open(WIDIV_SCALED_DIR / f).convert('RGB')).copy()).to(dev)
    summ.setdefault(g, []).append(tissue_layout_summary(rgb))
summ = {g: np.mean(v, 0) for g, v in summ.items()}
print(f'summaries {len(summ)} lines, dim {len(next(iter(summ.values())))} ({time.time() - t0:.0f}s)')
snp_map = pd.read_csv(WIDIV_SNP_MAP).set_index('snp').loc[snps]
proj = HybridLayout(200, 32, 256).fit(dosage, fit_rows, snps, snp_map.chr.to_numpy(), snp_map.pos.to_numpy(),
                                      lines, summ)
print(f'fit done ({time.time() - t0:.0f}s); output_dim {proj.output_dim}, windows {proj.n_windows}')

# Validation lines: predicted summary PCs vs their real summary PCs (option-1 check 2 gave PC1 0.78).
vl = sorted(val)
real = []
for g in vl:
    f = pd.read_csv(WIDIV_SCALED_METADATA, usecols=['scaled_filename', 'genotype']).set_index('genotype').loc[g, 'scaled_filename']
    rgb = torch.from_numpy(np.asarray(Image.open(WIDIV_SCALED_DIR / f).convert('RGB')).copy()).to(dev)
    real.append(tissue_layout_summary(rgb))
real = ((np.stack(real) - proj.summary_mean) / proj.summary_scale) @ proj.summary_components.T
pred = proj.predict_summary(dosage[[row_of[g] for g in vl]])
print('val summary PC1-8 r:', [round(np.corrcoef(pred[:, j], real[:, j])[0, 1], 2) for j in range(8)])
# Training lines get OOF predictions, unseen lines the full fit: check the rows differ accordingly.
tr_names = sorted(train)[:5]
rows_oof = proj.project_lines(dosage[[row_of[g] for g in tr_names]], tr_names)
rows_full = proj.transform(dosage[[row_of[g] for g in tr_names]])
print('training line: OOF vs in-sample summary differ:', not np.allclose(rows_oof[:, -256:], rows_full[:, -256:]),
      '| dosage part identical:', np.array_equal(rows_oof[:, :proj.n_snps], dosage[[row_of[g] for g in tr_names]]))
x = proj.project_lines(dosage, lines)
print('encoder input', x.shape, x.dtype, 'summary part spread on training lines (should be ~1):',
      round(float(x[fit_rows, proj.n_snps:].astype(np.float32).std()), 2))

# Encoder + UNet forward, as train_widiv builds them (medium preset).
cfg = HybridDosageEncoder.for_layout(proj, window_dim=4, embedding_dim=256, num_tokens=8, hidden_dim=512,
                                     summary_tokens=8, window_token_dropout=0.5).config()
enc = HybridDosageEncoder(**cfg).set_layout(proj).to(dev)
unet = DenoisingUNet(**dict(ck['unet_config'], channel_mult=tuple(ck['unet_config']['channel_mult']))).to(dev)
unet.set_store_attention(False)
xb = torch.tensor(x[fit_rows[:4]], dtype=torch.float16, device=dev)
enc.train()
tok = enc(xb)
print('tokens', tuple(tok.shape))
null_rows = (tok[:, 8:] == enc.null_window_tokens).all(-1).all(-1)
print('training: window tokens dropped for', int(null_rows.sum()), 'of 4 samples (random)')
enc.eval()
with torch.no_grad():
    out = unet(torch.randn(4, 4, 64, 64, device=dev), torch.full((4,), 500, device=dev), enc(xb))
    enc.drop_windows = True
    t2 = enc(xb)
print('unet out', tuple(out.shape), '| drop_windows gives null window tokens:',
      bool((t2[:, 8:] == enc.null_window_tokens).all()))
enc.drop_windows = False
# Checkpoint round trip through load_widiv_encoder.
fake = {'snp_projector': proj.state_dict(), 'snp_encoder_config': cfg, 'snp_encoder_state_dict': enc.state_dict()}
torch.save(fake, 'C:/Users/TylerM/AppData/Local/Temp/claude/5cea7d8aa087c68d/196eda90-cf5d-4266-a917-5c0f45e4b1a0/scratchpad/hybrid_fake.pt')
back = torch.load('C:/Users/TylerM/AppData/Local/Temp/claude/5cea7d8aa087c68d/196eda90-cf5d-4266-a917-5c0f45e4b1a0/scratchpad/hybrid_fake.pt', weights_only=False)
enc2, proj2 = load_widiv_encoder(back, dev)
x2 = proj2.project_lines(dosage[[row_of[g] for g in vl[:4]]], vl[:4])
x1 = proj.project_lines(dosage[[row_of[g] for g in vl[:4]]], vl[:4])
with torch.no_grad():
    d = (enc2(torch.tensor(x2, dtype=torch.float32, device=dev)) - enc(torch.tensor(x1, dtype=torch.float32, device=dev))).abs().max()
print('reloaded encoder max token difference:', float(d), '(float16 weights in the checkpoint)')
print(f'checkpoint projector size {os.path.getsize("C:/Users/TylerM/AppData/Local/Temp/claude/5cea7d8aa087c68d/196eda90-cf5d-4266-a917-5c0f45e4b1a0/scratchpad/hybrid_fake.pt") / 1e6:.0f} MB')
print(f'done ({time.time() - t0:.0f}s)')
