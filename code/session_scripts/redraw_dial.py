# Redraw the chosen dial example (MEMA020, seed 7) with a bolder poster-gold outline.
# Same segmentation as planted_clean_examples.py (512 px input, largest stele contour).
import sys
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0, r'E:/D2D/DNA_to_Display\code')
from paths import SEGMENTATION_MODEL
from feature_segmentation.evaluation.reconstruction_fidelity_test import segment
from ultralytics import YOLO

SRC = r'E:/D2D/DNA_to_Display\results\synthetic_stele\planted_gene_clean_examples_epoch_350\MEMA020_seed7'
DST = r'E:/D2D/DNA_to_Display\presentation\poster_assets'
GOLD = (241, 184, 45)
seg = YOLO(str(SEGMENTATION_MODEL))
for c in (1, 3, 6, 8):
    raw = np.array(Image.open(f'{SRC}/code{c}_raw.png').convert('RGB'))
    up = np.array(Image.fromarray(raw).resize((512, 512), Image.BICUBIC))
    if c == 8:   # section 1's "generated" picture: the same root before the outline
        Image.fromarray(up).save(f'{DST}/planted_dial_code8_raw.png')
    traits, masks = segment(seg, up, 0.25, 'cuda', 4, 2)
    m = max(masks[1], key=lambda x: x.sum()).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cont = max(cs, key=cv2.contourArea)
    out = up.copy()
    cv2.drawContours(out, [cont], -1, (0, 0, 0), 13, lineType=cv2.LINE_AA)   # dark halo for contrast
    cv2.drawContours(out, [cont], -1, GOLD, 8, lineType=cv2.LINE_AA)
    Image.fromarray(out).save(f'{DST}/planted_dial_code{c}.png')
    print(c, round(traits['stele_diameter_px'] / 2, 1))
