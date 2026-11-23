# Per-image stats of the scaled IMPC X-rays, to look for outliers that could
# throw a training step (inverted, blank, dark background, odd size/mode).
import sys
from pathlib import Path
import numpy as np
from PIL import Image
d = Path(r'E:/D2D/DNA_to_Display\results\impc_bcm\scaled_512')
rows = []
for p in sorted(d.glob('*')):
    im = Image.open(p)
    a = np.asarray(im.convert('RGB'), dtype=np.float32) / 255
    g = a.min(2)
    rows.append((p.name, im.mode, im.size, a.mean(), (g < 0.98).mean(), a[:8].mean(), a[-8:].mean()))
print(len(rows), 'images')
m = np.array([r[3] for r in rows]); f = np.array([r[4] for r in rows]); e = np.array([(r[5]+r[6])/2 for r in rows])
print('modes', {r[1] for r in rows}, 'sizes', {r[2] for r in rows})
for name, v in [('mean', m), ('tissue frac', f), ('edge mean', e)]:
    print(name, np.percentile(v, [0, 0.1, 1, 50, 99, 99.9, 100]).round(3))
print('lowest mean:'); [print(' ', r[0], round(r[3],3), round(r[4],3)) for r in sorted(rows, key=lambda r: r[3])[:8]]
print('highest tissue frac:'); [print(' ', r[0], round(r[3],3), round(r[4],3)) for r in sorted(rows, key=lambda r: -r[4])[:8]]
print('lowest tissue frac:'); [print(' ', r[0], round(r[3],3), round(r[4],3)) for r in sorted(rows, key=lambda r: r[4])[:5]]
