# Contact sheet of downloaded IMPC X-ray samples (data only, read as images).
import glob
import os
import sys
from PIL import Image, ImageDraw

src, out = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(os.path.join(src, '*.jpg')) + glob.glob(os.path.join(src, '*.png')))
W, H, cols = 260, 340, 6
sheet = Image.new('L', (W * cols, H * ((len(fs) + cols - 1) // cols)), 255)
for i, f in enumerate(fs):
    im = Image.open(f).convert('L')
    im.thumbnail((W, H))
    x, y = (i % cols) * W, (i // cols) * H
    sheet.paste(im, (x, y))
    ImageDraw.Draw(sheet).text((x + 4, y + 4), os.path.basename(f)[:14], fill=255)
sheet.save(out)
print(len(fs), 'images ->', out)
