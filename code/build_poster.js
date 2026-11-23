// Conference poster, 34 x 46 in portrait, one slide, Mizzou black and gold.
//   NODE_PATH=node_modules PPTX_SKILL_DIR=<pptx skill dir> node code/build_poster.js
// Images come from presentation/poster_assets (made by the poster asset scripts);
// the poster is written to presentation/DNA_to_Display_poster.pptx.
const path = require('path');
const pptxgen = require('pptxgenjs');

const SKILL = process.env.PPTX_SKILL_DIR;   // optional: writes the theme colours/fonts
const ROOT = path.resolve(__dirname, '..');
const A = (f) => path.join(ROOT, 'presentation', 'poster_assets', f);
const OUT = path.join(ROOT, 'presentation', 'DNA_to_Display_poster.pptx');

const BLACK = '111111', GOLD = 'F1B82D', GOLDTINT = 'FDF3DA', PANEL = 'F2F2F2', INK = '1A1A1A', MUTED = '5F6062',
  GREY = '6E7073', LIGHT = 'D9D9D9', WHITE = 'FFFFFF';
const FONT = 'Arial';
const THEME = {
  name: 'DNA to Display poster (Mizzou)', headFontFace: FONT, bodyFontFace: FONT,
  colors: { dk1: INK, lt1: WHITE, dk2: BLACK, lt2: PANEL, accent1: GOLD, accent2: GREY, accent3: 'A7A9AC', accent4: GOLDTINT,
            accent5: MUTED, accent6: '7A5C00', hlink: '7A5C00', folHlink: MUTED },
};

const pres = new pptxgen();
pres.defineLayout({ name: 'POSTER', width: 34, height: 46 });
pres.layout = 'POSTER';
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
pres.title = 'DNA to Display';
const s = pres.addSlide();
s.background = { color: WHITE };
const text = (t, o) => s.addText(t, { isTextBox: true, fontFace: FONT, color: INK, margin: 0, valign: 'top', ...o });
const img = (f, o) => s.addImage({ path: A(f), ...o });
const arrow = (x, y, w, h, o = {}) => s.addShape(pres.shapes.LINE, { x, y, w, h, line: { color: BLACK, width: 3, endArrowType: 'triangle', ...o } });

// ---------- Title band and hero ----------
s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 34, h: 10.15, fill: { color: BLACK }, line: { color: BLACK }, objectName: 'title band' });
text('DNA to Display', { x: 1, y: 0.55, w: 25.5, h: 1.7, fontSize: 108, bold: true, color: WHITE, objectName: 'title' });
text('Generating images of a plant\'s anatomy directly from its DNA', { x: 1, y: 2.3, w: 25.5, h: 0.85, fontSize: 44, color: GOLD, objectName: 'subtitle' });
text([
  { text: 'Tyler Mitts', options: { bold: true } },
  { text: '¹ · Jagdeep Sidhu¹ · Erik Amezquita¹ · Mo Akbar²', options: { breakLine: true } },
  { text: '¹University of Missouri, Columbia, MO  ·  ²Levity Turtle LLC', options: { fontSize: 24, color: LIGHT } },
], { x: 1, y: 3.3, w: 25.5, h: 1.2, fontSize: 30, color: WHITE, paraSpaceAfter: 4, objectName: 'authors' });
// Stacked MU mark, reversed (gold on black, white outline), cropped from the reversed
// Mizzou signature on brand.missouri.edu; right edge on the 1 in margin. 296 x 305 px.
const MARK_H = 3.6, MARK_W = MARK_H * 296 / 305;
img('mu_stacked_mark_reversed.png', { x: 33 - MARK_W, y: 0.75, w: MARK_W, h: MARK_H, objectName: 'MU logo',
  altText: 'University of Missouri stacked MU logo' });
const FS = 4.357, FG = 0.25, FY = 4.75;
for (let k = 0; k < 7; k++) {
  img(`noise_to_root_mema_frame${k}.png`, { x: 1 + k * (FS + FG), y: FY, w: FS, h: FS, objectName: `hero frame ${k}`,
    altText: k === 0 ? 'Starting noise' : k === 6 ? 'Finished generated root cross-section' : `Denoising step ${k}` });
}
const AY = FY + FS + 0.3;
s.addShape(pres.shapes.LINE, { x: 1 + FS / 2, y: AY, w: 6 * (FS + FG), h: 0, line: { color: GOLD, width: 3, endArrowType: 'triangle' }, objectName: 'hero arrow' });
text('random noise', { x: 1, y: AY + 0.15, w: FS, h: 0.5, fontSize: 24, color: GOLD, align: 'center', objectName: 'hero start label' });
text('generated root', { x: 1 + 6 * (FS + FG), y: AY + 0.15, w: FS, h: 0.5, fontSize: 24, color: GOLD, align: 'center', objectName: 'hero end label' });
text('One model run, guided only by the plant\'s DNA. The model never saw this genotype during training.', {
  x: 1 + FS + 0.3, y: AY + 0.15, w: 32 - 2 * FS - 0.6, h: 0.5, fontSize: 24, color: WHITE, italic: true, align: 'center', objectName: 'hero caption' });

// ---------- Section header ----------
function header(n, title, x, y, w = 15) {
  s.addShape(pres.shapes.OVAL, { x, y: y + 0.05, w: 0.95, h: 0.95, fill: { color: GOLD }, line: { color: GOLD }, objectName: `section ${n} badge` });
  text(String(n), { x, y: y + 0.05, w: 0.95, h: 0.95, fontSize: 38, bold: true, color: BLACK, align: 'center', valign: 'middle', objectName: `section ${n} number` });
  text(title, { x: x + 1.25, y, w, h: 1.05, fontSize: 44, bold: true, color: BLACK, valign: 'middle', objectName: `section ${n} title` });
}

// ---------- 1. The idea ----------
header(1, 'The idea: an image, not just a number', 1, 10.75, 20);
{
  // Two rows: usual genomic prediction vs this work.
  const rowLabel = (t, y, color) => text(t, { x: 1, y, w: 3.3, h: 1.3, fontSize: 22, bold: true, color, valign: 'middle', objectName: `idea label ${t}` });
  const dna = (y, n) => img('pipe_genotype.png', { x: 4.4, y: y + 0.1, w: 1.4, h: 1.1, objectName: `idea dna ${n}` });
  // row 1
  const y1 = 12.0;
  rowLabel('Genomic prediction', y1, MUTED);
  dna(y1, 1); arrow(5.95, y1 + 0.65, 0.75, 0);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.85, y: y1 + 0.1, w: 3.6, h: 1.1, fill: { color: PANEL }, line: { color: PANEL }, rectRadius: 0.15, objectName: 'idea number card' });
  text('stele = 0.95 mm', { x: 6.85, y: y1 + 0.1, w: 3.6, h: 1.1, fontSize: 24, bold: true, color: INK, align: 'center', valign: 'middle', objectName: 'idea number' });
  text('one number per trait you planned to measure', { x: 10.65, y: y1 + 0.1, w: 4.9, h: 1.1, fontSize: 19, color: MUTED, valign: 'middle', objectName: 'idea number note' });
  // row 2
  const y2 = 13.55;
  rowLabel('DNA to Display', y2, '7A5C00');
  dna(y2, 2); arrow(5.95, y2 + 0.7, 0.75, 0);
  // Same root as the measured one beside it (MEMA020, seed 7, gene = 8), before the stele outline.
  img('planted_dial_code8_raw.png', { x: 6.85, y: y2, w: 1.45, h: 1.45, objectName: 'idea generated root' });
  arrow(8.45, y2 + 0.7, 0.75, 0);
  img('planted_dial_code8.png', { x: 9.35, y: y2, w: 1.45, h: 1.45, objectName: 'idea measured root' });
  text('an image of the plant: measure any trait, even unplanned ones', { x: 10.95, y: y2, w: 4.6, h: 1.45, fontSize: 19, color: INK, valign: 'middle', objectName: 'idea image note' });
  // Right: the planted-gene dial.
  text('Change one gene → the stele grows', { x: 16.4, y: 11.95, w: 16.6, h: 0.55, fontSize: 26, bold: true, color: BLACK, objectName: 'dial title' });
  const codes = [1, 3, 6, 8], DS = 2.05, DX = 16.4, DY = 12.55;
  codes.forEach((c, i) => {
    const x = DX + i * (DS + 0.2);
    img(`planted_dial_code${c}.png`, { x, y: DY, w: DS, h: DS, objectName: `dial code ${c}`, altText: `Generated root with the artificial gene set to ${c}; stele outlined` });
    text(`gene = ${c}`, { x, y: DY + DS + 0.05, w: DS, h: 0.4, fontSize: 20, bold: true, color: INK, align: 'center', objectName: `dial label ${c}` });
  });
  const SX = DX + 4 * (DS + 0.2) + 0.2;
  text('+19%', { x: SX, y: DY + 0.05, w: 33 - SX, h: 1.0, fontSize: 60, bold: true, color: BLACK, objectName: 'dial stat' });
  text('stele width from gene 1 to 8, on a line the model never saw. Only that artificial gene changes; our segmentation model outlines the stele. It grew in 99% of 256 line × seed tests.', {
    x: SX, y: DY + 1.1, w: 33 - SX, h: 1.4, fontSize: 18, color: MUTED, objectName: 'dial note' });
}

// ---------- 2. How it works (pipeline) ----------
header(2, 'How it works', 1, 15.3);
const panel = (x, y, w, h, letter, title, name) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: PANEL }, line: { color: PANEL }, rectRadius: 0.25, objectName: `${name} panel` });
  s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.25, w: 0.75, h: 0.75, fill: { color: BLACK }, line: { color: BLACK }, objectName: `${name} badge` });
  text(letter, { x: x + 0.3, y: y + 0.25, w: 0.75, h: 0.75, fontSize: 28, bold: true, color: GOLD, align: 'center', valign: 'middle', objectName: `${name} letter` });
  text(title, { x: x + 1.3, y: y + 0.25, w: w - 1.6, h: 0.75, fontSize: 26, bold: true, color: BLACK, valign: 'middle', objectName: `${name} title` });
};
const box = (x, y, w, h, label, name, gold = false) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: gold ? GOLD : WHITE }, line: { color: BLACK, width: 2 }, rectRadius: 0.12, objectName: `${name} box` });
  text(label, { x, y, w, h, fontSize: 22, bold: true, color: BLACK, align: 'center', valign: 'middle', objectName: `${name} label` });
};
// Main label under a picture or box, plus a smaller grey detail line (the talk version's specifics).
const cap = (x, y, w, t, name, detail) => {
  text(t, { x: x - 0.45, y, w: w + 0.9, h: 0.42, fontSize: 22, bold: true, color: INK, align: 'center', objectName: name });
  if (detail) text(detail, { x: x - 0.6, y: y + 0.42, w: w + 1.2, h: 0.38, fontSize: 18, color: MUTED, align: 'center', objectName: `${name} detail` });
};
const sub = (x, y, w, t, name) => text(t, { x: x - 0.5, y, w: w + 1.0, h: 0.38, fontSize: 18, color: MUTED, align: 'center', objectName: name });

// A: autoencoder - one centred chain across the full width
panel(1, 16.45, 32, 4.45, 'A', 'Learn a compact image code  (LiteVAE autoencoder)', 'pipeline A');
{
  const y = 17.4, S = 2.45, G = 1.05, BW = 2.7, BH = 1.2;
  const chain = [['img', 'pipe_real_root.png', 'real image', '(3, 256, 256)'], ['box', 'wavelet\ntransform', 'fixed'],
                 ['img', 'pipe_wavelets.png', 'wavelet bands', 'LL   LH / HL   HH'], ['box', 'encoder', 'learned'],
                 ['img', 'pipe_latent.png', 'compact code', '(4, 32, 32): 48× smaller'], ['box', 'decoder', 'learned'],
                 ['img', 'pipe_recon_root.png', 'rebuilt image', '(3, 256, 256)']];
  const total = chain.reduce((t, c) => t + (c[0] === 'img' ? S : BW), 0) + G * (chain.length - 1);
  let x = 1 + (32 - total) / 2;
  chain.forEach((c, i) => {
    if (c[0] === 'img') {
      img(c[1], { x, y, w: S, h: S, objectName: `A ${c[2]}` });
      cap(x, y + S + 0.04, S, c[2], `A cap ${c[2]}`, c[3]);
      x += S;
    } else {
      box(x, y + S / 2 - BH / 2, BW, BH, c[1], `A ${c[1]}`);
      sub(x, y + S / 2 + BH / 2 + 0.05, BW, c[2], `A sub ${c[1]}`);
      x += BW;
    }
    if (i < chain.length - 1) { arrow(x + 0.12, y + S / 2, G - 0.24, 0); x += G; }
  });
}
// B: genome encoder
panel(1, 21.15, 12.6, 4.45, 'B', 'Encode the genome  (SNP encoder)', 'pipeline B');
const BY = 22.25;
img('pipe_genotype.png', { x: 1.6, y: BY, w: 2.75, h: 2.15, objectName: 'B genotype' });
cap(1.6, BY + 2.2, 2.75, 'genotype', 'B cap dna', '43,788 markers');
arrow(4.47, BY + 1.07, 0.7, 0);
box(5.3, BY + 0.47, 2.7, 1.2, 'SNP encoder', 'B encoder');
sub(5.3, BY + 1.72, 2.7, 'one-hot → PCA → MLP', 'B sub encoder');
arrow(8.13, BY + 1.07, 0.7, 0);
img('pipe_tokens_v2.png', { x: 8.95, y: BY + 0.1, w: 2.6, h: 1.98, objectName: 'B tokens',
  altText: 'Eight gene tokens, each a row of numbers: the 10 most variable of 256 shown, gold above and blue below the average genotype' });
cap(8.95, BY + 2.2, 2.6, 'gene tokens', 'B cap tokens', '(8, 256)');
// C: generator
panel(13.9, 21.15, 19.1, 4.45, 'C', 'Generate: denoise step by step, steered by the gene tokens', 'pipeline C');
const CY = 22.6, CS = 2.2;
img('pipe_noise.png', { x: 14.5, y: CY, w: CS, h: CS, objectName: 'C noise' }); cap(14.5, CY + CS + 0.04, CS, 'random noise', 'C cap noise', '~ N(0, 1)');
arrow(16.82, CY + CS / 2, 0.66, 0);
box(17.6, CY + CS / 2 - 0.65, 3.3, 1.3, 'denoising\nU-Net', 'C unet', true);
sub(17.6, CY + CS / 2 + 0.7, 3.3, '50 steps', 'C sub unet');
arrow(21.02, CY + CS / 2, 0.66, 0);
img('pipe_gen_latent.png', { x: 21.8, y: CY, w: CS, h: CS, objectName: 'C latent' }); cap(21.8, CY + CS + 0.04, CS, 'new code', 'C cap code', '(4, 32, 32)');
arrow(24.12, CY + CS / 2, 0.66, 0);
box(24.9, CY + CS / 2 - 0.6, 2.3, 1.2, 'decoder', 'C decoder');
sub(24.9, CY + CS / 2 + 0.65, 2.3, 'from A, frozen', 'C sub decoder');
arrow(27.32, CY + CS / 2, 0.66, 0);
img('noise_to_root_mema_frame6.png', { x: 28.1, y: CY, w: CS, h: CS, objectName: 'C root' }); cap(28.1, CY + CS + 0.04, CS, 'generated root', 'C cap root', 'steps shown in top strip');
// tokens steer the network: B tokens -> up -> across -> down into the network box
const ty = BY + 1.07, ly = 22.32, ux = 17.6 + 1.65, topBox = CY + CS / 2 - 0.65;
const STEER = '7A5C00';
s.addShape(pres.shapes.LINE, { x: 11.6, y: ty, w: 1.75, h: 0, line: { color: STEER, width: 4 }, objectName: 'steer 1' });
s.addShape(pres.shapes.LINE, { x: 13.35, y: ly, w: 0, h: ty - ly, line: { color: STEER, width: 4 }, objectName: 'steer 2' });
s.addShape(pres.shapes.LINE, { x: 13.35, y: ly, w: ux - 13.35, h: 0, line: { color: STEER, width: 4 }, objectName: 'steer 3' });
s.addShape(pres.shapes.LINE, { x: ux, y: ly, w: 0, h: topBox - ly, line: { color: STEER, width: 4, endArrowType: 'triangle' }, objectName: 'steer 4' });
text('read at every step', { x: ux + 0.15, y: ly + 0.04, w: 2.4, h: 0.4, fontSize: 18, bold: true, color: STEER, objectName: 'steer label' });

// ---------- 3. Accuracy (left) ----------
const L = 1, LW = 17.4, R = 19.1, RW = 13.9;
header(3, 'Accurate where the answer is known', L, 26.2, LW);
text('Simulated maize population with known genes  ·  trained on 1,780 lines  ·  tested on 198 unseen lines', {
  x: L, y: 27.4, w: LW, h: 0.95, fontSize: 24, objectName: 'accuracy text' });
const stats = [['0.81', 'average r over six traits'], ['0.89 vs 0.51', 'metaxylem diameter: two interacting genes, model vs GBLUP'],
               ['198', 'lines never seen in training']];
const cw = (LW - 0.6) / 3;
stats.forEach(([n, l], i) => {
  const x = L + i * (cw + 0.3);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 28.5, w: cw, h: 1.95, fill: { color: GOLDTINT }, line: { color: GOLDTINT }, rectRadius: 0.2, objectName: `stat card ${i + 1}` });
  text(n, { x: x + 0.3, y: 28.6, w: cw - 0.6, h: 1.0, fontSize: i === 1 ? 44 : 56, bold: true, color: BLACK, valign: 'middle', objectName: `stat ${i + 1} number` });
  text(l, { x: x + 0.3, y: 29.6, w: cw - 0.6, h: 0.8, fontSize: 19, color: INK, objectName: `stat ${i + 1} label` });
});
img('poster_scatter.png', { x: L, y: 30.75, w: 16.6, h: 11.13, objectName: 'accuracy scatter',
  altText: 'Six scatter plots of predictions against true genetic values: generated roots in gold, GBLUP in grey' });
text('Each dot is one unseen line.  Gold: read from 8 generated images.  Grey: genomic prediction (GBLUP).', {
  x: L, y: 41.95, w: LW, h: 0.45, fontSize: 20, italic: true, color: MUTED, objectName: 'scatter caption' });

// ---------- QR, acknowledgments, references (left bottom) ----------
// Points to https://tylermitts.com/dna-to-display. Its own white margin is the quiet zone.
img('qr_project_page.png', { x: L, y: 42.75, w: 2.6, h: 2.6, objectName: 'QR code',
  altText: 'QR code linking to tylermitts.com/dna-to-display' });
text([
  { text: 'Scan for the project page: tylermitts.com/dna-to-display', options: { bold: true, fontSize: 24, color: BLACK, breakLine: true } },
  // From the talk's acknowledgements slide; Sidhu, Amezquita and Akbar are already authors.
  { text: 'Acknowledgments: ', options: { bold: true, color: INK } },
  { text: 'Ruairidh Sawers (Penn State) · Sergio Perez-Limon (Penn State / NC State) · Alejandra Rougon (UNAM)', options: { breakLine: true } },
  { text: 'Contact: ', options: { bold: true, color: INK } },
  { text: 'tmitts0206@gmail.com' },
], { x: L + 2.95, y: 42.75, w: LW - 2.95, h: 2.6, fontSize: 18, color: MUTED, valign: 'middle', paraSpaceAfter: 6, objectName: 'credits' });

// ---------- 4. Real images (right) ----------
header(4, 'Real microscope images (MexMAGIC)', R, 26.2, RW - 1.3);
const GS = 8.0, GY = 27.45;
img('hero_which_is_real.png', { x: R, y: GY, w: GS, h: GS, objectName: 'which is real grid',
  altText: 'Nine root cross-sections, some real photographs and some generated from DNA' });
text([{ text: 'Which ones are real?', options: { bold: true, breakLine: true, fontSize: 30 } },
      { text: 'Some are photos. The rest were drawn from DNA the model never saw.', options: { fontSize: 22 } }],
     { x: R + GS + 0.35, y: GY, w: RW - GS - 0.35, h: 4.0, paraSpaceAfter: 8, objectName: 'which is real prompt' });
text('Real: 2, 3, 4, 8\nGenerated: 1, 5, 6, 7, 9', {
  x: R + GS + 0.35, y: GY + GS - 1.3, w: RW - GS - 0.35, h: 1.25, fontSize: 20, color: MUTED, rotate: 180, objectName: 'which is real answer' });

// ---------- 5. Change a gene (right) ----------
header(5, 'Genotype-to-phenotype maps', R, 35.85, RW - 1.3);
// Two lines tall: at one line the last word wrapped under the figure and was cut off.
text([
  { text: 'Swap one gene, map where its tissue is gained or lost: 12 of 14 designed genes point to the right tissue. ', options: { bold: true, color: '7A5C00' } },
  { text: '96 line-seed pairs per gene, unseen lines.', options: { color: MUTED } },
], { x: R, y: 36.9, w: RW, h: 0.85, fontSize: 21, valign: 'top', objectName: 'g2p subtitle' });
const MW = 13.6;
img('poster_gene_maps.png', { x: R + (RW - MW) / 2, y: 37.85, w: MW, h: MW * 1540 / 2780, objectName: 'genotype-to-phenotype maps',
  altText: 'Eight maps of where a gene\'s own tissue is gained (gold) or lost (blue) when only that gene is swapped: stele, vessels, cortex, air channels and lacunae; the control region stays pale' });

s.addNotes([
  'Opening (top strip): one run of the model. It starts from random noise and, guided only by the plant\'s DNA, draws a root cross-section of a genotype it never saw.',
  'Idea: genomic prediction gives a number per trait. We generate the image instead, so any trait can be measured afterwards, even ones nobody planned to measure. Roots are the test case; the method is not root-specific.',
  'Gene dial (if asked): we added an artificial block of markers to each MexMAGIC genotype whose value (1-8) tracks that plant\'s real stele size, and trained on it. Changing only that block on one generated root of a held-out line (MEMA020, same noise) grows the stele by 19% (125 to 148 px) while the root stays the same size; the gold outline is our segmentation model\'s stele. This example was chosen for clean outlines; across 256 line x seed pairs the stele grew in 99%, by a median of 4.5% (6 px).',
  'How it works: (A) an autoencoder learns a compact image code; (B) a genome encoder turns the DNA markers into 8 gene tokens; (C) a diffusion model starts from noise and removes it over 50 steps, steered by the tokens, then the decoder turns the code into an image.',
  'Accuracy: simulated population with known genetic values; 198 lines never seen. Average r 0.81. Metaxylem diameter is set by two interacting genes: 0.89 for the model, 0.51 for GBLUP. GBLUP stays ahead on simple additive traits, and aerenchyma and sclerenchyma are the hardest to draw.',
  'Real images: MexMAGIC roots imaged by laser ablation tomography. Ask them to guess which are real.',
  'Genotype-to-phenotype maps: for each gene we swap its chromosome region to the other parent on 48 unseen lines (2 seeds, same noise), align the roots, classify every pixel into tissues, and average where each tissue is gained (gold) or lost (blue), with every swap oriented the same way so random jitter cancels. The shared gene grows the stele, the vessel-width gene shrinks the vessels, the cortex gene changes the cortex band, the aerenchyma gene adds air channels, and the lacuna gene - a trait nobody measured or gave the model - rearranges the air channels. A region with no designed gene stays blank. Validated on all 14 designed genes: 12 point to the right tissue (10 of 11 confident calls), and no control region is called. Plain pixel-difference maps get only 7 of 14, because every edit jitters the high-contrast vessels and root edge.',
  'Beyond roots (say it, not on the poster): the same recipe works for any image tied to a genotype, such as seeds, leaves or ears.',
  'Limits if asked: the model partly memorises training images; final scores on the 495 hidden test lines are still to come.',
].join('\n\n'));

(async () => {
  await pres.writeFile({ fileName: OUT });
  if (SKILL) {
    const { applyTheme } = require(path.join(SKILL, 'scripts', 'apply_theme.js'));
    await applyTheme(OUT, THEME);
  }
  console.log('wrote', OUT);
})();
