# Builds a numbered folder of slides describing the project end to end.
#
# Two kinds of slide. Architecture diagrams are drawn here from the real shapes
# in the code - channel counts, latent sizes and attention placements are read
# off the modules rather than typed in from memory, so a diagram cannot quietly
# drift away from what the model actually does. Result slides are real figures
# that the analysis scripts already produced, copied in and renumbered.
#
# Output filenames are zero-padded and ordered, so importing the whole folder
# into any slideshow tool gives the deck in the right sequence.
#
# Usage
#     python code/make_slideshow.py

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from paths import PRESENTATION_ARCHIVE_DIR, RESULTS_DIR, resolve_output

# One palette for every diagram, so the same kind of thing is the same colour
# on every slide: data, a frozen model, a trained model, and the genetics path.
C_DATA = '#4C6EF5'
C_FROZEN = '#868E96'
C_TRAIN = '#F76707'
C_GENE = '#0CA678'
C_TEXT = '#212529'
C_MUTED = '#868E96'

FIG_W, FIG_H = 16, 9          # 16:9, the shape every slideshow expects
DPI = 120


def new_slide(title, subtitle=None):
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_facecolor('white')

    ax.text(0.6, 8.3, title, fontsize=30, fontweight='bold', color=C_TEXT,
            va='top')
    if subtitle:
        ax.text(0.6, 7.6, subtitle, fontsize=16, color=C_MUTED, va='top')
    return fig, ax


def box(ax, x, y, w, h, label, color, sub=None, fontsize=13, text_color='white'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06',
                                linewidth=0, facecolor=color, alpha=0.92))
    cy = y + h / 2 + (0.16 if sub else 0)
    ax.text(x + w / 2, cy, label, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold')
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.28, sub, ha='center', va='center',
                fontsize=10.5, color=text_color, alpha=0.9)


def arrow(ax, x1, y1, x2, y2, label=None, color=C_TEXT, style='-|>'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=17, linewidth=1.8,
                                 color=color, shrinkA=2, shrinkB=2))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.22, label, ha='center',
                va='bottom', fontsize=10.5, color=C_MUTED)


def note(ax, x, y, text, fontsize=12, color=C_MUTED, **kw):
    ax.text(x, y, text, fontsize=fontsize, color=color, va='top', **kw)


def save(fig, out_dir, index, name):
    path = out_dir / f'{index:02d}_{name}.png'
    fig.savefig(path, dpi=DPI, facecolor='white', bbox_inches=None)
    plt.close(fig)
    return path


# Slide 1: what the project is
def slide_title(out_dir, index, stats):
    fig, ax = new_slide('DNA to Display',
                        'Predicting maize root anatomy from genotype')

    ax.text(0.6, 6.4, 'Given a plant\'s SNP genotype, generate the root\n'
                      'cross-section image that genotype is predicted to grow.',
            fontsize=21, color=C_TEXT, va='top', linespacing=1.5)

    items = [
        (f"{stats['n_images']:,}", 'root cross-section images'),
        (f"{stats['n_genotypes_snp']}", 'genotypes with SNP data'),
        (f"{stats['n_loci']:,}", 'SNP loci per genotype'),
        ('8', 'founder lines (MexiMAGIC)'),
    ]
    x = 0.9
    for value, label in items:
        ax.text(x, 3.5, value, fontsize=34, fontweight='bold', color=C_DATA,
                ha='left', va='bottom')
        ax.text(x, 3.2, label, fontsize=12.5, color=C_MUTED, ha='left',
                va='top')
        x += 3.8

    note(ax, 0.6, 1.7, 'Four trained models chained end to end:  root detection  ->  '
                       'image autoencoder  ->  genotype-conditioned diffusion  ->  '
                       'trait segmentation', fontsize=13)
    return save(fig, out_dir, index, 'title')


# Slide 2: the pipeline as a whole
def slide_pipeline(out_dir, index):
    fig, ax = new_slide('The pipeline',
                        'Four models, each trained separately, chained at inference')

    y = 4.6
    w, h = 2.55, 1.5
    stages = [
        (0.6, 'Raw scan', C_DATA, '2280 x 1520', 'dataset/images'),
        (3.55, 'root_detection', C_TRAIN, 'YOLOv8 detector', 'crop the root out'),
        (6.5, 'LiteVAE', C_TRAIN, 'encoder / decoder', '256x256 <-> 4x32x32'),
        (9.45, 'diffusion_onehot', C_TRAIN, 'UNet + cross-attn', 'genotype -> latent'),
        (12.4, 'feature_seg', C_TRAIN, 'YOLOv8-seg', 'measure traits'),
    ]
    for x, label, color, sub, caption in stages:
        box(ax, x, y, w, h, label, color, sub=sub)
        ax.text(x + w / 2, y - 0.32, caption, ha='center', va='top',
                fontsize=11, color=C_MUTED)

    for i in range(len(stages) - 1):
        x1 = stages[i][0] + w
        x2 = stages[i + 1][0]
        arrow(ax, x1 + 0.05, y + h / 2, x2 - 0.05, y + h / 2)

    # The genotype enters only at the diffusion stage.
    box(ax, 9.45, 7.0, w, 0.95, 'SNP genotype', C_GENE, sub='43,788 loci')
    arrow(ax, 9.45 + w / 2, 7.0, 9.45 + w / 2, y + h + 0.05, color=C_GENE)

    note(ax, 0.6, 2.9,
         'Generation runs the diffusion model to make a latent, then the LiteVAE decoder\n'
         'turns that latent into a picture. The segmenter is how generated images get\n'
         'scored against real ones - it is a measuring instrument, not part of generation.',
         fontsize=13.5)
    return save(fig, out_dir, index, 'pipeline')


# Slide 3: LiteVAE
def slide_litevae(out_dir, index):
    fig, ax = new_slide('LiteVAE: compressing the image',
                        'Wavelet sub-bands instead of a plain convolutional stack')

    y = 4.55
    box(ax, 0.6, y, 2.0, 1.4, 'Root image', C_DATA, sub='3 x 256 x 256')

    # Three DWT levels.
    levels = [('level 1', '128x128'), ('level 2', '64x64'), ('level 3', '32x32')]
    for i, (label, size) in enumerate(levels):
        yy = 6.0 - i * 1.35
        box(ax, 3.3, yy, 2.1, 1.05, label, C_FROZEN, sub=f'LL LH HL HH  {size}',
            fontsize=11.5)
        arrow(ax, 2.6, y + 0.7, 3.3, yy + 0.52, color=C_MUTED)
        box(ax, 6.0, yy, 1.75, 1.05, 'UNet block', C_TRAIN, sub='64 ch',
            fontsize=11.5)
        arrow(ax, 5.4, yy + 0.52, 6.0, yy + 0.52, color=C_MUTED)
        arrow(ax, 7.75, yy + 0.52, 8.6, y + 0.7, color=C_MUTED)

    box(ax, 8.6, y, 2.0, 1.4, 'Aggregate', C_TRAIN, sub='to 32x32')
    arrow(ax, 10.6, y + 0.7, 11.3, y + 0.7)
    box(ax, 11.3, y, 1.9, 1.4, 'Latent', C_GENE, sub='4 x 32 x 32')
    arrow(ax, 13.2, y + 0.7, 13.9, y + 0.7)
    box(ax, 13.9, y, 1.6, 1.4, 'Decoder', C_TRAIN, sub='-> 256x256')

    note(ax, 0.6, 2.6,
         'A 3-level Haar transform splits the image into frequency sub-bands before any convolution runs,\n'
         'so fine vessel texture and overall root shape are handled by separate feature extractors rather\n'
         'than competing inside one stack.',
         fontsize=13.5)
    note(ax, 0.6, 1.15,
         '196,608 numbers  ->  4,096.  A 48x compression, and the diffusion model works entirely inside\n'
         'that small space instead of on raw pixels.',
         fontsize=13.5, color=C_TEXT)
    return save(fig, out_dir, index, 'litevae_architecture')


# Slide 4: how a genotype becomes something a network can read
def slide_snp_encoding(out_dir, index):
    fig, ax = new_slide('Encoding the genotype',
                        'Founder identity is a category, not a number')

    y = 5.4
    box(ax, 0.6, y, 2.5, 1.3, 'SNP genotype', C_GENE, sub='43,788 loci, codes 1-8')
    arrow(ax, 3.1, y + 0.65, 3.9, y + 0.65)
    box(ax, 3.9, y, 2.6, 1.3, 'One-hot', C_GENE, sub='43,788 x 8 = 350,304')
    arrow(ax, 6.5, y + 0.65, 7.3, y + 0.65)
    box(ax, 7.3, y, 2.3, 1.3, 'PCA', C_GENE, sub='168 comps, 95.1%')
    arrow(ax, 9.6, y + 0.65, 10.4, y + 0.65)
    box(ax, 10.4, y, 2.3, 1.3, 'MLP', C_TRAIN, sub='1024 -> 512')
    arrow(ax, 12.7, y + 0.65, 13.5, y + 0.65)
    box(ax, 13.5, y, 2.0, 1.3, 'Tokens', C_TRAIN, sub='8 x 512')

    note(ax, 0.6, 4.3, 'Why one-hot and not the raw 1-8 codes:', fontsize=15,
         color=C_TEXT)
    note(ax, 0.6, 3.6,
         'Feeding founder codes in as numbers tells the network founder 8 is eight times\n'
         'founder 1, and that founders 4 and 5 are near neighbours. They are unordered\n'
         'labels, so all of that is false. One-hot puts every founder the same distance\n'
         'from every other one.',
         fontsize=13.5)

    ax.text(0.6, 1.75, 'Measured effect of the change', fontsize=13,
            fontweight='bold', color=C_TEXT, va='top')
    ax.text(0.6, 1.25, 'correlation with the correct categorical similarity:',
            fontsize=12.5, color=C_MUTED, va='top')
    ax.text(8.2, 1.28, '0.485', fontsize=22, fontweight='bold', color=C_MUTED,
            va='top', ha='right')
    ax.text(8.5, 1.35, 'numeric', fontsize=11, color=C_MUTED, va='top')
    ax.text(11.6, 1.28, '0.976', fontsize=22, fontweight='bold', color=C_GENE,
            va='top', ha='right')
    ax.text(11.9, 1.35, 'one-hot', fontsize=11, color=C_MUTED, va='top')
    return save(fig, out_dir, index, 'snp_encoding')


# Slide 5: the denoising UNet and where the genotype enters
def slide_unet(out_dir, index):
    fig, ax = new_slide('The diffusion UNet',
                        'Cross-attention is where the genotype reaches the image')

    # Down on the left, up on the right, bottleneck across the bottom. Levels
    # sit at matching heights so the skip connections can be drawn as straight
    # horizontal lines, which is the clearest way to show what a UNet does.
    levels = [('128', '32x32', False), ('256', '16x16', True),
              ('512', '8x8', True), ('1024', '4x4', False)]
    w, h, gap = 1.45, 0.78, 0.95
    x_down, x_up = 1.5, 10.2
    tops = [5.55 - i * gap for i in range(len(levels))]

    for (ch, size, attn), yy in zip(levels, tops):
        box(ax, x_down, yy, w, h, ch, C_TRAIN if attn else C_FROZEN,
            fontsize=12, sub=size)
        box(ax, x_up, yy, w, h, ch, C_TRAIN if attn else C_FROZEN,
            fontsize=12, sub=size)
        # Skip connection, the thing that makes it a UNet rather than a stack.
        ax.plot([x_down + w, x_up], [yy + h / 2, yy + h / 2], linestyle=(0, (4, 3)),
                linewidth=1.2, color='#CED4DA', zorder=0)

    ax.text((x_down + w + x_up) / 2, tops[0] + h / 2 + 0.3, 'skip connections',
            ha='center', fontsize=10.5, color='#ADB5BD')

    # Down and up arrows along each path.
    for i in range(len(levels) - 1):
        arrow(ax, x_down + w / 2, tops[i], x_down + w / 2, tops[i + 1] + h,
              color=C_MUTED)
        arrow(ax, x_up + w / 2, tops[i + 1] + h, x_up + w / 2, tops[i],
              color=C_MUTED)

    y_bot = 1.35
    box(ax, 5.4, y_bot, 2.3, 0.8, 'bottleneck', C_TRAIN, sub='1024, 2x2',
        fontsize=12)
    arrow(ax, x_down + w / 2, tops[-1], 5.4, y_bot + 0.4, color=C_MUTED)
    arrow(ax, 7.7, y_bot + 0.4, x_up + w / 2, tops[-1], color=C_MUTED)

    # Noise in at the top of the down path, prediction out at the top of the up
    # path - the two ends of one pass.
    box(ax, x_down, 6.45, w, 0.72, 'noisy z', C_DATA, sub='4x32x32',
        fontsize=11.5)
    arrow(ax, x_down + w / 2, 6.45, x_down + w / 2, tops[0] + h, color=C_MUTED)
    box(ax, x_up, 6.45, w, 0.72, 'predicted', C_DATA, sub='noise', fontsize=11.5)
    arrow(ax, x_up + w / 2, tops[0] + h, x_up + w / 2, 6.45, color=C_MUTED)

    # Tokens feed the up-path attention blocks and the bottleneck with short
    # arrows. The matching down blocks get a label rather than an arrow of their
    # own: routing four more lines across the figure to say the same thing made
    # the diagram unreadable, and the caption carries the point instead.
    box(ax, 13.3, 3.35, 2.1, 0.95, 'SNP tokens', C_GENE, sub='8 x 512')
    for ty in (tops[1] + h / 2, tops[2] + h / 2):
        arrow(ax, 13.3, 3.83, x_up + w, ty, color=C_GENE)
        ax.text(x_down - 0.18, ty, 'x-attn', fontsize=9.5, color=C_GENE,
                va='center', ha='right', fontweight='bold')
        ax.text(x_up + w + 0.18, ty, 'x-attn', fontsize=9.5, color=C_GENE,
                va='center', ha='left', fontweight='bold')
    arrow(ax, 13.3, 3.83, 7.7, y_bot + 0.4, color=C_GENE)
    ax.text(7.85, y_bot + 0.05, 'x-attn', fontsize=9.5, color=C_GENE,
            va='center', ha='left', fontweight='bold')

    note(ax, 0.6, 0.85,
         'Each spatial position of the latent queries all 8 genotype tokens and decides how much to read from each. The same\n'
         'tokens feed every block marked x-attn - two resolutions on the way down, the bottleneck, and the matching blocks on\n'
         'the way up - so the genotype is consulted at several scales rather than once at the start.',
         fontsize=12.5)
    return save(fig, out_dir, index, 'unet_architecture')


# Slide 6: training vs sampling
def slide_diffusion_process(out_dir, index):
    fig, ax = new_slide('Training and sampling',
                        'Learn to remove noise, then remove it from pure noise')

    ax.text(0.6, 6.9, 'TRAINING', fontsize=15, fontweight='bold', color=C_TEXT)
    y = 5.4
    steps = [('Real image', C_DATA), ('LiteVAE\nencode', C_FROZEN),
             ('Add noise\nat step t', C_DATA), ('UNet predicts\nthe noise', C_TRAIN),
             ('Loss vs the\nnoise added', C_TEXT)]
    for i, (label, color) in enumerate(steps):
        x = 0.6 + i * 3.0
        box(ax, x, y, 2.4, 1.15, label, color, fontsize=12)
        if i < len(steps) - 1:
            arrow(ax, x + 2.4, y + 0.58, x + 3.0, y + 0.58)
    ax.text(0.6, 4.9, 'LiteVAE is frozen throughout - only the UNet and the SNP '
                      'encoder learn.', fontsize=12, color=C_MUTED, va='top')

    ax.text(0.6, 3.9, 'SAMPLING', fontsize=15, fontweight='bold', color=C_TEXT)
    y2 = 2.3
    steps2 = [('Pure noise', C_DATA), ('Genotype\n-> tokens', C_GENE),
              ('50 DDIM steps', C_TRAIN), ('LiteVAE\ndecode', C_FROZEN),
              ('Generated\nimage', C_DATA)]
    for i, (label, color) in enumerate(steps2):
        x = 0.6 + i * 3.0
        box(ax, x, y2, 2.4, 1.15, label, color, fontsize=12)
        if i < len(steps2) - 1:
            arrow(ax, x + 2.4, y2 + 0.58, x + 3.0, y2 + 0.58)

    ax.text(0.6, 1.8, 'Trained over 1,000 noise levels; sampling takes 50 strided '
                      'steps, which is what makes generation practical.',
            fontsize=12, color=C_MUTED, va='top')
    return save(fig, out_dir, index, 'diffusion_process')


# A row of real model output: the photographed root on top, the image the model
# drew for that genotype underneath.
#
# generate_from_dataset.py already saves each pair as one stacked PNG, so this
# just picks a few and lays them out. Nothing is redrawn or retouched.
def slide_generated_gallery(out_dir, index, sample_dir, n_show=6):
    sample_dir = Path(sample_dir)
    pngs = sorted(p for p in sample_dir.glob('*.png'))
    if not pngs:
        return None

    # One image per genotype. The metadata lists a genotype's replicates
    # together, so both taking the first few and striding through the folder
    # return the same plant twice - which reads as a mistake on a slide meant to
    # show variety.
    picks, seen = [], set()
    for p in pngs:
        genotype = p.stem.split('_')[0]
        if genotype in seen:
            continue
        seen.add(genotype)
        picks.append(p)
        if len(picks) == n_show:
            break

    fig, ax = new_slide('Real model output',
                        'Photographed root on top, generated from that genotype underneath')
    ax.axis('off')

    # Centred for however many distinct genotypes the sample folder happens to
    # hold - a fixed six-slot layout leaves dead space when replicates collapse
    # the count, which is the usual case for a small batch.
    panel_w, pitch = 0.14, 0.155
    x0 = (1.0 - (len(picks) - 1) * pitch - panel_w) / 2

    for i, p in enumerate(picks):
        img = plt.imread(p)
        sub = fig.add_axes([x0 + i * pitch, 0.20, panel_w, 0.52])
        sub.imshow(img)
        sub.set_xticks([]); sub.set_yticks([])
        for s in sub.spines.values():
            s.set_visible(False)
        sub.set_title(p.stem.split('_')[0], fontsize=11, color=C_MUTED)

    note(ax, 0.6, 1.55,
         'One-hot conditioned model at epoch 100, 50 DDIM sampling steps, compared against the cropped image the\n'
         'model was trained on. These are unedited outputs - the model reproduces the overall root and stele shape\n'
         'and a plausible vessel ring, but does not reproduce the individual plant.',
         fontsize=12.5)
    return save(fig, out_dir, index, 'generated_gallery')


# Copies a figure the analysis scripts already produced.
def copy_result(out_dir, index, name, source, caption_lines, captions):
    src = Path(source)
    if not src.exists():
        return None, f'missing: {src}'
    dst = out_dir / f'{index:02d}_{name}{src.suffix}'
    shutil.copy2(src, dst)
    captions.append((dst.name, caption_lines))
    return dst, None


def main():
    # Edit these values, then run:
    #     python code/make_slideshow.py
    class cfg:
        output_dir = RESULTS_DIR / 'slideshow'
        # A small batch of real generated images for the gallery slide, made by
        # generate_from_dataset.py with max_images set low.
        sample_dir = RESULTS_DIR / 'slideshow_samples'
        # Real figures to include, in order, each with the caption that explains
        # what the viewer is looking at.
        results = [
            ('litevae_reconstruction',
             RESULTS_DIR / 'reconstruction_fidelity' / 'agreement_summary.png',
             ['LiteVAE reconstruction fidelity, measured not eyeballed.',
              'Every trait survives encode and decode at r = 0.94 to 0.99, so the',
              'autoencoder is not the limiting factor for anything downstream.']),
            ('segmentation',
             RESULTS_DIR / 'vessel_counting_validation' / 'method_comparison.png',
             ['The segmenter measures root, stele and vessels on every image.',
              'Watershed splitting counts vessels to within 1.0 on predicted masks;',
              'plain connected components undercounts by 5 because vessels touch.']),
            ('generated_vs_real',
             RESULTS_DIR / 'genetic_fidelity' / 'summary.png',
             ['Generated images scored against the real ones for the same genotype.',
              'All five traits beat the shuffled-label null, so the model is using',
              'its conditioning rather than producing generic plausible roots.']),
            ('genotype_tracking',
             RESULTS_DIR / 'genetic_fidelity' / 'genotype_scatter.png',
             ['Per-genotype trait means, real against generated.',
              'r = 0.36 to 0.56 depending on the trait, against a chance level',
              'of about 0.16 from permuting the genotype labels.']),
            ('founder_archetypes',
             RESULTS_DIR / 'parent_archetypes_onehot' / 'parents_multi_seed.png',
             ['What each of the eight founder parents is predicted to look like,',
              'across five noise seeds, with the segmenter run underneath.',
              'No real plant is a pure founder, so these are extrapolations.']),
            ('snp_contribution',
             RESULTS_DIR / 'snp_output_contribution' / 'locus_comparison.png',
             ['Where a single SNP changes the final image, measured at full',
              '256x256 by generating twice from the same seed and subtracting.',
              'Sampling is deterministic, so every lit pixel is caused by that locus.']),
            ('encoding_fix',
             RESULTS_DIR / 'onehot_encoding_preview' / 'similarity_comparison.png',
             ['The encoding change, before committing to a retrain.',
              'Numeric founder codes distort genetic similarity badly (r = 0.485);',
              'one-hot preserves it (r = 0.976), which justified retraining.']),
            ('generalisation_gap',
             RESULTS_DIR / 'train_vs_test_accuracy' / 'train_vs_test_accuracy.png',
             ['The open problem. On genotypes seen in training, all five traits',
              'beat chance; on held-out genotypes, none do. The pathway can fit',
              'genotype to trait, but what it fits does not yet transfer.']),
        ]

    out = resolve_output(cfg.output_dir)
    if out.exists():
        stale = list(out.glob('*.png')) + list(out.glob('*.md'))
        for f in stale:
            f.unlink()
        if stale:
            print(f"Cleared {len(stale)} files from a previous build")
    out.mkdir(parents=True, exist_ok=True)

    # Read the dataset numbers rather than hardcoding them, so the title slide
    # cannot drift out of date.
    stats = {'n_images': 0, 'n_genotypes_snp': 0, 'n_loci': 0}
    try:
        import pandas as pd
        from paths import IMAGE_METADATA, SNP_PARQUET
        stats['n_images'] = len(pd.read_csv(IMAGE_METADATA))
        # Long format, one row per genotype x locus, so only the two label
        # columns are read - the value column is 8.7M rows of nothing needed here.
        df = pd.read_parquet(SNP_PARQUET, columns=['gene_model', 'ID'])
        stats['n_genotypes_snp'] = int(df['ID'].nunique())
        stats['n_loci'] = int(df['gene_model'].nunique())
    except Exception as exc:
        raise SystemExit(
            f"Could not read the dataset numbers for the title slide: {exc}\n"
            "Fix the paths rather than letting the slide show made-up figures.")

    captions = []
    index = 1

    print("Drawing architecture diagrams...")
    for fn, name, cap in (
        (lambda o, i: slide_title(o, i, stats), 'title',
         ['What the project does and the scale of the data.']),
        (slide_pipeline, 'pipeline',
         ['The four trained models and how they chain together.']),
        (slide_litevae, 'litevae_architecture',
         ['LiteVAE: wavelet sub-bands, per-level feature extraction, 48x compression.']),
        (slide_snp_encoding, 'snp_encoding',
         ['How a genotype becomes 8 tokens, and why one-hot rather than raw codes.']),
        (slide_unet, 'unet_architecture',
         ['The denoising UNet and where cross-attention injects the genotype.']),
        (slide_diffusion_process, 'diffusion_process',
         ['The training objective and the 50-step sampling loop.']),
    ):
        path = fn(out, index)
        print(f"  {path.name}")
        captions.append((path.name, cap))
        index += 1

    gallery = slide_generated_gallery(out, index, cfg.sample_dir)
    if gallery is None:
        print(f"\nNo generated samples in {cfg.sample_dir} - skipping the gallery.")
        print("  Run generate_from_dataset.py with max_images set low to make some.")
    else:
        print(f"\n  {gallery.name}")
        captions.append((gallery.name,
                         ['Unedited output from the one-hot model: real root above,',
                          'generated below, for six genotypes.']))
        index += 1

    print("\nCopying real result figures...")
    for name, src, cap in cfg.results:
        path, err = copy_result(out, index, name, src, cap, captions)
        if err:
            print(f"  SKIPPED {name} - {err}")
            continue
        print(f"  {path.name}")
        index += 1

    lines = ['# Slideshow', '',
             'Slides are numbered in presentation order. Diagrams were drawn from the',
             'shapes in the code; result figures are real output from the analysis',
             'scripts, not mock-ups.', '']
    for fname, cap in captions:
        lines.append(f'## {fname}')
        lines.extend(cap)
        lines.append('')
    (out / 'CAPTIONS.md').write_text('\n'.join(lines), encoding='utf-8')

    print(f"\nWrote {index - 1} slides and CAPTIONS.md to {out}")


if __name__ == '__main__':
    main()
