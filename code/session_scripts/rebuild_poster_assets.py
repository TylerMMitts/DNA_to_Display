# Rebuilds presentation/poster_assets/ after the D: drive failure (2026-11-23).
#
# EXACT: cut or copied from surviving originals, the same pixels as before.
#   pipe_*                       the talk deck's pipeline figure (image12.png in
#                                Downloads/DNA_to_Display_talk_updated.pptx, md5 50024ae9),
#                                cut with the original pipecrop.py boxes
#   pipe_tokens_v2, hero_which_is_real   images recorded in the chat transcripts at full size
#   mu_stacked_mark_reversed     make_mu_logo.py on the downloaded Mizzou signature
#   qr_project_page              the QR image from the chat
# STAND-IN: lower resolution until the models are back from Hellbender; rerun the
# script named in brackets to replace each one.
#   noise_to_root_mema_frame0-6  cut from the transcript copy of the hero strip, 275 px
#                                instead of 512 [mema_poster_heroes.py]
#   planted_dial_code1/3/6/8     MEMA020 seed 7 at 400 px with the thin first-pass outline
#                                [planted_clean_examples.py, then redraw_dial.py]
#   planted_dial_code8_raw       no unoutlined copy survived: the outlined one stands in
#                                [redraw_dial.py]
#   poster_scatter, poster_gene_maps   transcript copies, 2000 px wide instead of 3520 / 2780
#                                [poster_scatter.py, poster_gene_maps_v2.py]
import shutil
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'presentation' / 'poster_assets'
OUT.mkdir(parents=True, exist_ok=True)
REC = Path('E:/D2D/recovered/images')
SCRATCH = Path(r'C:\Users\TylerM\AppData\Local\Temp\claude\5cea7d8aa087c68d'
               r'\196eda90-cf5d-4266-a917-5c0f45e4b1a0')
TALK = Path(r'C:\Users\TylerM\Downloads\DNA_to_Display_talk_updated.pptx')

# Pipeline pieces, exact.
fig = Image.open(BytesIO(zipfile.ZipFile(TALK).read('ppt/media/image12.png'))).convert('RGB')
assert fig.size == (2306, 1345), fig.size
boxes = {'pipe_real_root': (84, 231, 314, 461), 'pipe_wavelets': (688, 231, 917, 461),
         'pipe_latent': (1291, 231, 1521, 461), 'pipe_recon_root': (1896, 231, 2125, 461),
         'pipe_genotype': (75, 857, 346, 1067), 'pipe_tokens': (605, 891, 790, 1032),
         'pipe_noise': (910, 818, 1101, 1009), 'pipe_gen_latent': (1489, 818, 1680, 1009),
         'pipe_gen_root': (1979, 794, 2214, 1030)}
for name, box in boxes.items():
    fig.crop(box).save(OUT / f'{name}.png')

shutil.copy(REC / 'project/presentation/poster_assets/pipe_tokens_v2.png', OUT / 'pipe_tokens_v2.png')
Image.open(REC / 'project/presentation/poster_assets/hero_which_is_real.png.jpeg').convert('RGB').save(
    OUT / 'hero_which_is_real.png')
shutil.copy(SCRATCH / 'images' / '1.png', OUT / 'qr_project_page.png')
subprocess.run([sys.executable, str(Path(__file__).parent / 'make_mu_logo.py'),
                str(SCRATCH / 'scratchpad' / 'mu_logo_dl' / 'orig_Color-Sig-FC-W.jpg'),
                str(OUT / 'mu_stacked_mark_reversed.png')], check=True)

# Stand-ins.
strip = Image.open(REC / 'project/presentation/poster_assets/hero_noise_to_root_mema.png.jpeg').convert('RGB')
s = strip.width / 3728                       # original: 7 cells of 512 with 24 px gaps
for k in range(7):
    x0 = round(k * (512 + 24) * s)
    strip.crop((x0, 0, x0 + round(512 * s), strip.height)).save(OUT / f'noise_to_root_mema_frame{k}.png')
strip.save(OUT / 'hero_noise_to_root_mema.png')

top2 = Image.open(SCRATCH / 'scratchpad' / 'stele_top2.png').convert('RGB')   # row 2 = MEMA020 seed 7
for i, c in enumerate((1, 3, 6, 8)):
    top2.crop((i * 400, 400, (i + 1) * 400, 800)).save(OUT / f'planted_dial_code{c}.png')
shutil.copy(OUT / 'planted_dial_code8.png', OUT / 'planted_dial_code8_raw.png')

Image.open(REC / 'project/presentation/poster_assets/poster_scatter.png.jpeg').convert('RGB').save(
    OUT / 'poster_scatter.png')
Image.open(REC / 'project/presentation/poster_assets/poster_gene_maps.png.jpeg').convert('RGB').save(
    OUT / 'poster_gene_maps.png')
print('poster assets in', OUT)
for f in sorted(OUT.glob('*.png')):
    print(f'  {f.name:36s} {Image.open(f).size}')
