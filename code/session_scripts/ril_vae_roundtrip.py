# Is the LiteVAE the bottleneck? Real validation images measured as they are and
# after an encode -> decode round trip (mean latent, and a sampled latent as the
# diffusion model trains on), each line's traits vs the true genetic value.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, load_tables, measure
import numpy as np
import pandas as pd
import torch
from PIL import Image
from widiv_data import make_transforms
from widiv_litevae import load_widiv_litevae
from widiv_paths import WIDIV_SCALED_DIR

dev = torch.device('cuda')
meta, split = load_tables()
val = pd.read_csv(OUT / 'validation_lines.csv').genotype.tolist()
G = pd.read_csv('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/genetic_values.csv')
G = G[G.line.map(split) == 'train'].set_index('line')
enc, dec = load_widiv_litevae(Path('E:/D2D/DNA_to_Display/models/from_hellbender/litevae_ril_v5_v3_best.pt'), dev)
tf = make_transforms(False)
T = ['stele_diam_um', 'mx_n', 'mx_diam_um', 'cortex_width_um', 'rca_pct', 'mcs_layers']
rows = []
torch.manual_seed(0)
for g in val:
    x = tf(Image.open(WIDIV_SCALED_DIR / meta.loc[g, 'scaled_filename']).convert('RGB'))[None].to(dev)
    with torch.no_grad():
        _, zm, zl = enc(x)
        outs = {'original': x,
                'recon_mean': dec(zm, save_steps=False),
                'recon_sample': dec(zm + torch.exp(0.5 * zl) * torch.randn_like(zm), save_steps=False)}
    for kind, im in outs.items():
        rgb = ((im[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        rows.append({'genotype': g, 'kind': kind, **measure(rgb)})
d = pd.DataFrame(rows)
d.to_csv(OUT / 'vae_roundtrip.csv', index=False)
res = []
for t in T:
    orig = d[d.kind == 'original'].set_index('genotype')[t]
    row = {'trait': t}
    for kind in ('original', 'recon_mean', 'recon_sample'):
        v = d[d.kind == kind].set_index('genotype')[t].reindex(val)
        row[f'{kind}_vs_g'] = np.corrcoef(v, G.loc[val, f'g_{t}'])[0, 1]
        if kind != 'original':
            row[f'{kind}_vs_original'] = np.corrcoef(v, orig.reindex(val))[0, 1]
            row[f'{kind}_mean_shift'] = float((v - orig.reindex(val)).mean())
    res.append(row)
res = pd.DataFrame(res).set_index('trait')
res.to_csv(OUT / 'vae_roundtrip_summary.csv')
pd.set_option('display.width', 220)
print(res.round(3).to_string())
