# Stacked MU mark from the reversed Mizzou signature (brand.missouri.edu Color-Sig-FC-W,
# full-size upload) -> transparent PNG for the poster's black title band. Only the mark and
# its (R) are kept: the columns left of the first wide gap (the space before "Mizzou").
# The source is drawn on black, so alpha comes from the brightest channel and colour is
# un-premultiplied; gold and white stay fully opaque in their own colours. No pixels of
# the mark are redrawn or recoloured.
import sys
import numpy as np
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
a = np.asarray(Image.open(src).convert('RGB'), dtype=np.float32) / 255
alpha = np.clip((a.max(axis=2) - 0.06) / (0.6 - 0.06), 0, 1)
ink = alpha > 0.15
ink[:10], ink[-10:], ink[:, :10], ink[:, -10:] = False, False, False, False   # the file has a 1 px frame
alpha = np.where(ink | (np.arange(a.shape[1])[None] > 10), alpha, 0)
cols = np.flatnonzero(ink.any(axis=0))
gap = np.flatnonzero(np.diff(cols) > 40)                 # first wide gap = mark | wordmark
x0, x1 = cols[0], cols[gap[0]] + 1
rows = np.flatnonzero(ink[:, x0:x1].any(axis=1))
y0, y1 = rows[0], rows[-1] + 1
pad = 8
y0, y1, x0, x1 = max(y0 - pad, 0), min(y1 + pad, a.shape[0]), max(x0 - pad, 0), min(x1 + pad, a.shape[1])
rgb = np.clip(a / np.maximum(alpha, 1e-3)[..., None], 0, 1)
out = np.dstack([rgb, alpha])[y0:y1, x0:x1]
Image.fromarray((out * 255).round().astype(np.uint8), 'RGBA').save(dst)
print(dst, out.shape[1], 'x', out.shape[0], 'aspect w/h', round(out.shape[1] / out.shape[0], 4))
