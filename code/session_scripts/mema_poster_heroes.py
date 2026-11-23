# Poster hero candidates from the MexMAGIC (MEMA) model, epoch 300 - inference only.
#   hero_noise_to_root_mema.png  the reverse diffusion for held-out MEMA042, seed 6: frames
#                                from the starting noise to the finished root
#   hero_which_is_real.png       3x3 grid of real and generated held-out roots, numbered,
#                                plus which_is_real_answers.json
# Writes to presentation/poster_assets/.
import sys
import json
sys.path.insert(0, r'E:/D2D/DNA_to_Display\code')
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from model_toolkit import load_session, _litevae, _scheduler

OUT = __import__('pathlib').Path(r'E:/D2D/DNA_to_Display\presentation\poster_assets')
OUT.mkdir(parents=True, exist_ok=True)
s = load_session(300)
dev = s.device
_, decoder, _ = _litevae(dev)
sched = _scheduler(dev)
C = s.unet_cfg['latent_channels']

# 1. Noise to root.
G, SEED, STEPS = 'MEMA042', 6, 50
keep = [0, 24, 34, 40, 44, 47, 49]
z = torch.randn(C, 32, 32, generator=torch.Generator().manual_seed(SEED))[None].to(dev)
frames = []
with torch.no_grad():
    emb = s.encoder(torch.tensor(s.codes(G), device=dev))
    ts = sched.get_timesteps(STEPS, dev)
    for i, t in enumerate(ts):
        tb = torch.full((1,), int(t.item()), device=dev, dtype=torch.long)
        tp = int(ts[i + 1].item()) if i + 1 < len(ts) else -1
        z = sched.denoise_step(z, s.unet(z, tb, emb), tb, torch.full((1,), tp, device=dev, dtype=torch.long))
        if i in keep:
            img = torch.clamp((decoder(z, save_steps=False)[0] + 1) / 2, 0, 1)
            frames.append((img.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
cell, gap, bg = 512, 24, (8, 12, 28)
strip = Image.new('RGB', (len(frames) * cell + (len(frames) - 1) * gap, cell), bg)
for k, f in enumerate(frames):
    strip.paste(Image.fromarray(f).resize((cell, cell), Image.LANCZOS), (k * (cell + gap), 0))
strip.save(OUT / 'hero_noise_to_root_mema.png')
for k, f in enumerate(frames):
    Image.fromarray(f).resize((cell, cell), Image.LANCZOS).save(OUT / f'noise_to_root_mema_frame{k}.png')
print('noise to root done; held-out?', s.group(G))

# 2. Which is real? Held-out genotypes only (never in this model's training).
real_tiles = []
for g in ('MEMA310', 'MEMA233', 'MEMA105', 'MEMA296'):
    im = s.real_images(g, 1)
    if len(im):
        real_tiles.append((g, 'real', im[0]))
gen_plan = [('MEMA042', 6), ('MEMA042', 10), ('MEMA310', 7), ('MEMA233', 1), ('MEMA105', 8)]
gen_tiles = []
for g, sd in gen_plan:
    gen_tiles.append((g, 'generated', s.generate(g, seeds=(sd,))[0, 0]))
tiles = real_tiles + gen_tiles
order = np.random.default_rng(3).permutation(len(tiles))
tiles = [tiles[i] for i in order][:9]
cell, gap = 512, 24
grid = Image.new('RGB', (3 * cell + 2 * gap, 3 * cell + 2 * gap), (255, 255, 255))
answers = {}
try:
    font = ImageFont.truetype('arialbd.ttf', 44)
except OSError:
    font = ImageFont.load_default()
for k, (g, kind, im) in enumerate(tiles):
    x, y = (k % 3) * (cell + gap), (k // 3) * (cell + gap)
    grid.paste(Image.fromarray(im).resize((cell, cell), Image.LANCZOS), (x, y))
    d = ImageDraw.Draw(grid)
    d.ellipse([x + 16, y + 16, x + 80, y + 80], fill=(255, 255, 255))
    d.text((x + 48, y + 48), str(k + 1), fill=(8, 12, 28), font=font, anchor='mm')
    answers[k + 1] = {'genotype': g, 'kind': kind}
grid.save(OUT / 'hero_which_is_real.png')
(OUT / 'which_is_real_answers.json').write_text(json.dumps(answers, indent=1))
print('which is real done:', {k: v['kind'] for k, v in answers.items()})
