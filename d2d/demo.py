# A small synthetic dataset for trying the pipeline end to end in a minute:
# 120 lines x 2,000 markers, three images per line. Each image is a round
# "cross-section" whose outer size, inner core size and number of dark spots are
# set by a few markers plus noise, so every stage has real signal to find.
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .project import ROOT


def make_demo(folder=None, n_lines=120, n_markers=2000, images_per_line=3, seed=0):
    rng = np.random.default_rng(seed)
    folder = folder or ROOT / 'demo_source'
    (folder / 'images').mkdir(parents=True, exist_ok=True)
    # Two founders, recombinant lines: blocks of shared alleles, so markers are in LD.
    blocks = rng.integers(0, 2, size=(n_lines, n_markers // 50))
    x = np.repeat(blocks, 50, axis=1) * 2
    flip = rng.random(x.shape) < 0.02
    x[flip] = 2 - x[flip]
    lines = [f'L{i:03d}' for i in range(n_lines)]
    geno = pd.DataFrame(x, index=pd.Index(lines, name='line'), columns=[f'm{j:04d}' for j in range(n_markers)])
    geno.to_csv(folder / 'genotypes.csv')
    pd.DataFrame({'snp': geno.columns, 'chr': 1 + np.arange(n_markers) // 500,
                  'pos': (np.arange(n_markers) % 500) * 20000}).to_csv(folder / 'marker_map.csv', index=False)
    size_g = (x[:, 100] + x[:, 900]) / 4          # 0..1
    core_g = x[:, 1500] / 2
    spots_g = x[:, 300] / 2
    rows = []
    for i, line in enumerate(lines):
        for k in range(images_per_line):
            r_out = 150 + 70 * size_g[i] + rng.normal(0, 8)
            r_in = r_out * (0.3 + 0.2 * core_g[i] + rng.normal(0, 0.02))
            n_spots = int(round(6 + 10 * spots_g[i] + rng.normal(0, 1.5)))
            im = Image.new('RGB', (600, 600), (250, 250, 250))
            d = ImageDraw.Draw(im)
            cx, cy = 300 + rng.normal(0, 15), 300 + rng.normal(0, 15)
            d.ellipse([cx - r_out, cy - r_out, cx + r_out, cy + r_out], fill=(120, 150, 200))
            d.ellipse([cx - r_in, cy - r_in, cx + r_in, cy + r_in], fill=(70, 90, 150))
            for s in range(n_spots):
                a = 2 * np.pi * s / max(n_spots, 1)
                px, py = cx + 0.8 * r_in * np.cos(a), cy + 0.8 * r_in * np.sin(a)
                d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(30, 30, 60))
            name = f'{line}_rep{k + 1}.png'
            im.save(folder / 'images' / name)
            rows.append({'line': line, 'outer_diameter': 2 * r_out, 'core_diameter': 2 * r_in, 'spots': n_spots})
    t = pd.DataFrame(rows).groupby('line').mean().reset_index()
    t.to_csv(folder / 'line_traits.csv', index=False)
    return folder
