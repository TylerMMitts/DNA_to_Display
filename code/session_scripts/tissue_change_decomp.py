# What "tissue gained / lost" is made of: for each QTL map (original images and generated),
# net change of each tissue and of background over the whole frame, and a radial profile.
import numpy as np, pandas as pd
GM = 'E:/D2D/DNA_to_Display/results/ril_v5/gene_maps/'
COMP = ['vessels', 'stele', 'aerenchyma', 'outer', 'cortex']
R = np.load(GM + 'qtl_maps_original_images.npz'); G = np.load(GM + 'gene_maps_methods.npz')
UM = 2 * float(R['half']) / 256 * 5.3798          # um per canonical pixel (original frame)
yy, xx = np.mgrid[:256, :256]; rad = np.hypot(yy - 127.5, xx - 127.5)
keys = ['shared_stele_mx', 'mx_diam_add', 'rca_linked1', 'lac_q'] + [f'ctx_q{i}' for i in range(1, 9)]
rows = []
for src, Z in (('original', R), ('generated', G)):
    for k in keys:
        m = Z[f'{k}__tissue']
        full = np.concatenate([m, -m.sum(0, keepdims=True)])          # + background
        net = full.reshape(6, -1).sum(1)
        rows.append({'source': src, 'qtl': k, **{f'net_{c}': v for c, v in zip(COMP + ['background'], net)}})
D = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print('net change in pixels (original frame: 1 px =', round(UM ** 2, 1), 'um^2)')
print(D.round(0).to_string(index=False))
# radial profile of ctx_q1 / ctx_q2, original images
tmpl = R['template']
for k in ('ctx_q1', 'ctx_q2'):
    m = R[f'{k}__tissue']; full = np.concatenate([m, -m.sum(0, keepdims=True)])
    print('\n', k, 'radial profile, original images (mean change per pixel; template tissue share)')
    print(' r_um   ' + '  '.join(f'{c[:6]:>7}' for c in COMP + ['backgr']) + '   | template cortex  outer  stele')
    for r0 in range(0, 128, 8):
        ring = (rad >= r0) & (rad < r0 + 8)
        print(f'{(r0 + 4) * UM:6.0f} ' + '  '.join(f'{full[j][ring].mean():+7.3f}' for j in range(6)) +
              f'   | {tmpl[4][ring].mean():.2f}  {tmpl[3][ring].mean():.2f}  {(tmpl[0][ring] + tmpl[1][ring]).mean():.2f}')
