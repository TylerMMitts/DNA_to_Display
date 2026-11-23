# Redraws the planted-gene example outlines from the saved raw images, keeping
# only the largest stele contour (the segmenter adds small stray specks).
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r'E:/D2D/DNA_to_Display\code')
from paths import SEGMENTATION_MODEL, SYNTHETIC_STELE_RESULTS_DIR
from feature_segmentation.evaluation.reconstruction_fidelity_test import segment
from ultralytics import YOLO

OUT = SYNTHETIC_STELE_RESULTS_DIR / 'planted_gene_examples_epoch_350'
EA = Path(r'E:/D2D/DNA_to_Display\presentation\edit_assets')
ex = json.load(open(OUT / 'summary.json'))['example']
g, seed = ex['genotype'], ex['seed']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seg = YOLO(str(SEGMENTATION_MODEL))
AMBER, BLUE = (242, 166, 90), (120, 170, 255)


def main_contour(img):
    traits, masks = segment(seg, img, 0.25, device, 4, 2)
    m = max(masks[1], key=lambda x: x.sum()).astype(np.uint8)
    m = cv2.resize(m, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea), traits['stele_diameter_px']


def draw(img, contours):
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    for c, col in contours:
        cv2.drawContours(big, [c], -1, col, 3)
    return big


raw = {c: np.array(Image.open(OUT / f'{g}_seed{seed}_code{c}_raw.png').convert('RGB')) for c in range(1, 9)}
cont = {c: main_contour(raw[c]) for c in raw}
for c in raw:
    Image.fromarray(draw(raw[c], [(cont[c][0], AMBER)])).save(EA / f'planted_dial_code{c}.png')
    print(c, round(cont[c][1], 1))
Image.fromarray(draw(raw[1], [(cont[1][0], BLUE)])).save(EA / 'planted_pair_code1.png')
Image.fromarray(draw(raw[8], [(cont[8][0], AMBER)])).save(EA / 'planted_pair_code8.png')
Image.fromarray(draw(raw[8], [(cont[1][0], BLUE), (cont[8][0], AMBER)])).save(EA / 'planted_pair_overlay.png')
