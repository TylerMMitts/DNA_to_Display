// Builds the conference talk: 16 slides, about 20 minutes.
//
//     uv run code/make_hook_assets.py
//     uv run code/make_talk_assets.py
//     node code/build_talk_deck.js
//
// The shape of the talk:
//   hook    four roots that were never grown, from DNA alone
//   why     an image is expensive, a genotype is cheap, and here is the data
//   how     the three pieces, one slide each: what goes in, what comes out, and
//           why that piece was chosen over the obvious alternative
//   where   nothing in the middle is about roots
//   close   one slide, then questions
//
// Written for a plant audience: the pieces are described by their inputs and
// outputs and the reasoning behind them, not by their internals. Numbers on the
// slides are measured, and the comment above each one says where it came from.

const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');

const ROOT = path.resolve(__dirname, '..');
const HOOK = path.join(ROOT, 'results', 'hook_slides', 'assets_epoch300');
const TALK = path.join(ROOT, 'results', 'talk_slides', 'assets_epoch300');
const OUT = path.join(ROOT, 'results', 'talk_slides', 'DNA_to_Display_talk_epoch300.pptx');

const BG = '0B1020';
const PANEL = '17203A';
const INK = 'CADCFC';
const WHITE = 'FFFFFF';
const AMBER = 'F2A65A';
const MUTED = '7C8DB0';
const ROSE = 'E8607A';
const GREEN = '5FD3A6';

const SERIF = 'Cambria';
const SANS = 'Calibri';
const W = 13.333;

const FOUNDER_COLORS = ['4C9BE8', 'F2A65A', '5FD3A6', 'E8607A',
                        'B48BE8', 'F2D65A', '5FC8D3', 'E88A4C'];

const hook = (n) => path.join(HOOK, n);
const talk = (n) => path.join(TALK, n);

// Written by make_hook_assets.py, so the slides name the genotype and counts
// that script actually used.
const HOOK_INFO = JSON.parse(fs.readFileSync(hook('hook.json'), 'utf8'));
const G = HOOK_INFO.genotype;

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.author = 'Tyler Mitts';
pres.title = 'DNA to Display';

function slide() {
  const s = pres.addSlide();
  s.background = { color: BG };
  return s;
}

// Places an image as large as fits in the box without distorting it, centred.
function imageFit(s, file, x, y, w, h) {
  const buf = fs.readFileSync(file);
  const ratio = buf.readUInt32BE(16) / buf.readUInt32BE(20);
  let iw = w, ih = w / ratio;
  if (ih > h) { ih = h; iw = h * ratio; }
  s.addImage({ path: file, x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih });
}

function eyebrow(s, text, y = 0.55) {
  s.addText(text, {
    x: 0.7, y: y, w: 11.0, h: 0.3, isTextBox: true, margin: 0,
    fontFace: SANS, fontSize: 14, bold: true, color: AMBER, charSpacing: 3,
  });
}

function title(s, text, y = 0.92, size = 36) {
  s.addText(text, {
    x: 0.7, y: y, w: 11.9, h: 0.8, isTextBox: true, margin: 0,
    fontFace: SERIF, fontSize: size, color: WHITE,
  });
}

function body(s, text, opts) {
  s.addText(text, Object.assign({
    isTextBox: true, margin: 0, fontFace: SANS, fontSize: 15, color: INK,
  }, opts));
}

function stat(s, value, label, x, y, w, color = AMBER) {
  s.addText(value, {
    x: x, y: y, w: w, h: 0.62, isTextBox: true, margin: 0,
    fontFace: SANS, fontSize: 32, bold: true, color: color,
  });
  s.addText(label, {
    x: x, y: y + 0.6, w: w, h: 0.5, isTextBox: true, margin: 0,
    fontFace: SANS, fontSize: 12.5, color: INK,
  });
}

function panel(s, x, y, w, h, accent = false) {
  s.addShape(pres.ShapeType.roundRect, {
    x: x, y: y, w: w, h: h, rectRadius: 0.1,
    fill: { color: PANEL }, line: { color: accent ? AMBER : '24304F', width: accent ? 1.25 : 1 } });
}

// The in/out/why row the three architecture slides share, so each piece is read
// the same way: what it is handed, what it hands on, and the decision behind it.
function pieceCards(s, inText, outText, whyTitle, whyText, y = 5.0) {
  const w = 3.83, gap = 0.22;
  const cards = [
    ['IN', inText, MUTED, false],
    ['OUT', outText, MUTED, false],
    [whyTitle, whyText, AMBER, true],
  ];
  cards.forEach((c, i) => {
    const x = 0.7 + i * (w + gap);
    panel(s, x, y, w, 1.85, c[3]);
    s.addText([
      { text: c[0] + '\n', options: { fontSize: 11, bold: true, color: c[2], charSpacing: 2, breakLine: true } },
      { text: c[1], options: { fontSize: 13.5, color: c[3] ? WHITE : INK } },
    ], { x: x + 0.28, y: y + 0.22, w: w - 0.56, h: 1.5, isTextBox: true, margin: 0,
         fontFace: SANS, lineSpacingMultiple: 1.1 });
  });
}

const generated = [0, 1, 2, 3].map((i) => hook(`generated_${i}.png`));

/* ---------------------------------------------------------------- act 1: hook */

// 1 - title
{
  const s = slide();
  // Centred over the strip below, which is the only other thing on the slide.
  s.addText('DNA to Display', {
    x: 0.7, y: 1.6, w: 11.93, h: 1.2, isTextBox: true, margin: 0, align: 'center',
    fontFace: SERIF, fontSize: 54, color: WHITE });
  s.addText('Generating maize root anatomy from genotype', {
    x: 0.7, y: 2.85, w: 11.93, h: 0.6, isTextBox: true, margin: 0, align: 'center',
    fontFace: SANS, fontSize: 22, color: INK });
  s.addText('Tyler Mitts', {
    x: 0.7, y: 3.75, w: 11.93, h: 0.45, isTextBox: true, margin: 0, align: 'center',
    fontFace: SANS, fontSize: 18, bold: true, color: AMBER });
  // One real run, left to right: pure noise to a root. The talk comes back to it
  // on the generator slide, where the same strip is labelled.
  imageFit(s, talk('fig_denoise_strip.png'), 0.7, 4.75, 11.93, 1.7);
  body(s, 'ONE RUN OF THE MODEL, LEFT TO RIGHT', {
    x: 0.7, y: 6.5, w: 11.93, h: 0.25, align: 'center', fontSize: 10, color: MUTED, charSpacing: 2 });
  s.addNotes('I work on predicting what a plant looks like on the inside from its DNA. I am going to show you what that looks like, then take the system apart piece by piece - what each piece is handed, what it hands on, and why it is there.');
}

// 2 - the question
{
  const s = slide();
  s.addText('What if you could see inside a plant’s roots\nwithout ever growing one?', {
    x: 0.9, y: 2.15, w: 11.6, h: 2.2, isTextBox: true, margin: 0,
    fontFace: SERIF, fontSize: 40, color: WHITE, lineSpacingMultiple: 1.2 });
  s.addText('Just from a sample of its DNA.', {
    x: 0.9, y: 4.5, w: 11.6, h: 0.6, isTextBox: true, margin: 0,
    fontFace: SANS, fontSize: 24, color: AMBER });
  s.addNotes('Here is the question I want to start with. What a root looks like in cross-section decides how a plant takes up water and nutrients - but measuring it means growing the plant, harvesting the root, sectioning it and putting it under a microscope. So: what if you could see inside a plant\'s roots without ever growing one, just from a sample of its DNA?\n\n[Beat, then move on.]');
}

// 3 - four roots
{
  const s = slide();
  eyebrow(s, 'ONE MAIZE LINE');
  title(s, 'Four primary roots', 0.92, 40);
  const size = 2.75, gap = 0.36;
  const x0 = (W - (size * 4 + gap * 3)) / 2;
  generated.forEach((file, i) => s.addImage({ path: file, x: x0 + i * (size + gap), y: 2.15, w: size, h: size }));
  body(s, 'Cross-sections through four primary roots of the same maize line: the ring of water-carrying vessels, the cortex around it, the air spaces in between.', {
    x: 0.7, y: 5.25, w: 11.9, h: 0.6, fontSize: 16 });
  s.addNotes('Here are four primary roots of one maize line, in cross-section. The dark circles in the middle are the vessels that carry water up the plant, the ring around them is the cortex, and the gaps in it are air spaces. Each is a little different, the way replicates of one plant are.\n\n[Pause. Do not say yet where they came from.]');
}

// 4 - the confession
{
  const s = slide();
  s.addText('One confession:', {
    x: 0.9, y: 2.45, w: 11.5, h: 0.7, isTextBox: true, margin: 0,
    fontFace: SANS, fontSize: 24, color: AMBER });
  s.addText('None of these roots\nwere ever grown.', {
    x: 0.9, y: 3.15, w: 11.5, h: 2.0, isTextBox: true, margin: 0,
    fontFace: SERIF, fontSize: 54, color: WHITE, lineSpacingMultiple: 1.1 });
  s.addNotes('One confession: none of these roots were ever grown.\n\n[Beat.]');
}

// 5 - how they were made
{
  const s = slide();
  title(s, 'No seed was planted. No section was cut.', 0.75, 32);
  body(s, 'A computer drew all four, from this line’s DNA and nothing else.', {
    x: 0.7, y: 1.45, w: 11.9, h: 0.5, fontSize: 17 });

  s.addImage({ path: hook('snp_strip_zoom.png'), x: 0.7, y: 3.54, w: 3.1, h: 0.62 });
  body(s, 'ITS DNA, AS THE MODEL READS IT', {
    x: 0.7, y: 4.24, w: 3.4, h: 0.25, fontSize: 10, color: MUTED, charSpacing: 2 });
  s.addShape(pres.ShapeType.rightArrow, { x: 4.0, y: 3.72, w: 0.7, h: 0.26, fill: { color: AMBER } });
  s.addShape(pres.ShapeType.roundRect, {
    x: 4.95, y: 3.29, w: 2.5, h: 1.12, rectRadius: 0.12,
    fill: { color: PANEL }, line: { color: AMBER, width: 1.25 } });
  s.addText('the model', {
    x: 4.95, y: 3.29, w: 2.5, h: 1.12, isTextBox: true, align: 'center',
    valign: 'middle', fontFace: SANS, fontSize: 16, bold: true, color: WHITE });
  s.addShape(pres.ShapeType.rightArrow, { x: 7.7, y: 3.72, w: 0.7, h: 0.26, fill: { color: AMBER } });

  const size = 1.62, gap = 0.14;
  generated.forEach((file, i) => s.addImage({ path: file,
    x: 8.9 + (i % 2) * (size + gap), y: 2.16 + Math.floor(i / 2) * (size + gap), w: size, h: size }));
  body(s, 'FOUR TRIES, ONE GENOTYPE', {
    x: 8.9, y: 5.62, w: 4.0, h: 0.25, fontSize: 10, color: MUTED, charSpacing: 2 });
  body(s, 'The DNA is the only input. The four differ because each one starts from a different patch of random noise — nothing about the plant changed between them.', {
    x: 0.7, y: 6.25, w: 11.9, h: 0.6 });
  s.addNotes('No seed was planted, no section was cut. A computer drew all four from this line\'s DNA and nothing else. The coloured bar is that DNA in the form the model reads - I will come back to what the colours mean. The four images differ only because each starts from a different patch of random noise.');
}

// 6 - real vs generated
{
  const s = slide();
  eyebrow(s, G);
  title(s, 'And this is one of its real roots.', 0.92, 36);
  // The reveal carries one line and nothing else. The training-set counts used
  // to sit here, but they are the subject of the data slide two along, and on
  // this slide they pull the eye away from the comparison.
  const size = 2.05, gap = 0.16;
  const gridW = size * 2 + gap;
  const rightX = 12.63 - gridW;
  body(s, 'GENERATED', { x: 0.7, y: 1.98, w: gridW, h: 0.3, fontSize: 11, bold: true, color: AMBER, charSpacing: 2 });
  body(s, 'REAL', { x: rightX, y: 1.98, w: gridW, h: 0.3, fontSize: 11, bold: true, charSpacing: 2 });
  generated.forEach((file, i) => s.addImage({ path: file,
    x: 0.7 + (i % 2) * (size + gap), y: 2.45 + Math.floor(i / 2) * (size + gap), w: size, h: size }));
  s.addImage({ path: hook('real_root.png'), x: rightX, y: 2.45, w: gridW, h: gridW });
  s.addText('The model had\nnever seen\nthis line.', {
    x: 0.7 + gridW + 0.2, y: 3.75, w: rightX - (0.7 + gridW) - 0.4, h: 1.7,
    isTextBox: true, margin: 0, align: 'center',
    fontFace: SERIF, fontSize: 26, color: WHITE, lineSpacingMultiple: 1.12 });
  s.addNotes(`And this is a real root from the same line, photographed and sectioned. The model never saw it, or any other image of this line: it learned from ${HOOK_INFO.n_trained_images} cross-sections across ${HOOK_INFO.n_trained_genotypes} other lines.\n\n[For you, not the audience: this line is one of 27 the model never trained on, and it is the closest-looking of them - the best example, not a typical one.]`);
}

// 7 - why bother
{
  const s = slide();
  eyebrow(s, 'WHY BOTHER');
  title(s, 'One of these is cheap. The other is not.');
  panel(s, 0.7, 2.0, 5.85, 3.0);
  s.addText([
    { text: 'AN IMAGE\n', options: { fontSize: 11, color: MUTED, charSpacing: 2, breakLine: true } },
    { text: 'grow the plant · harvest the root · fix and section it · image the section\n\n', options: { fontSize: 15, color: WHITE, breakLine: true } },
    { text: 'weeks of work, one plant at a time', options: { fontSize: 13, color: ROSE } },
  ], { x: 1.0, y: 2.3, w: 5.3, h: 2.4, isTextBox: true, margin: 0, fontFace: SANS, lineSpacingMultiple: 1.2 });

  panel(s, 6.78, 2.0, 5.85, 3.0, true);
  s.addText([
    { text: 'A GENOTYPE\n', options: { fontSize: 11, color: MUTED, charSpacing: 2, breakLine: true } },
    { text: 'take a leaf punch · sequence it once · reuse it forever\n\n', options: { fontSize: 15, color: WHITE, breakLine: true } },
    { text: 'already done for thousands of lines', options: { fontSize: 13, color: GREEN } },
  ], { x: 7.08, y: 2.3, w: 5.3, h: 2.4, isTextBox: true, margin: 0, fontFace: SANS, lineSpacingMultiple: 1.2 });

  body(s, 'If the cheap measurement could stand in for the expensive one — even partly — anatomy could be screened at the scale genotyping already works at. That is the whole motivation, and it is not specific to roots.', {
    x: 0.7, y: 5.35, w: 11.93, h: 1.0, fontSize: 16 });
  s.addNotes('So why bother? Because the two things are wildly different in cost. An image means growing a plant, harvesting the root, fixing, sectioning and imaging it - weeks, one plant at a time. A genotype is a leaf punch, sequenced once, and for most breeding populations it has already been done. If the cheap measurement can stand in for the expensive one, even partly, you can screen anatomy at the scale genotyping already works at.');
}

/* ------------------------------------------------------- act 2: what it is given */

// 8 - the data
{
  const s = slide();
  eyebrow(s, 'THE DATA');
  title(s, 'Eight founders, 43,788 markers.');

  // One line's genome beside one of its roots: the pairing the whole project
  // rests on, rather than several strips to compare at once.
  imageFit(s, hook('genome_block.png'), 0.7, 2.0, 7.5, 2.5);
  body(s, `${G} · ALL 43,788 MARKERS, COLOURED BY THE FOUNDER EACH CAME FROM`, {
    x: 0.7, y: 4.6, w: 7.5, h: 0.25, fontSize: 10, color: MUTED, charSpacing: 1.5 });
  FOUNDER_COLORS.forEach((c, i) => {
    const x = 0.7 + i * 0.62;
    s.addShape(pres.ShapeType.rect, {
      x: x, y: 5.0, w: 0.2, h: 0.2, fill: { color: c }, line: { color: c } });
    body(s, String(i + 1), { x: x + 0.24, y: 4.96, w: 0.35, h: 0.28, fontSize: 12, valign: 'middle' });
  });

  s.addImage({ path: hook('real_root.png'), x: 9.0, y: 2.0, w: 3.35, h: 3.35 });
  body(s, 'ONE OF ITS ROOTS', {
    x: 9.0, y: 5.45, w: 3.35, h: 0.25, align: 'center', fontSize: 10, color: MUTED, charSpacing: 2 });

  const xs = [0.7, 4.0, 7.3, 10.3];
  stat(s, '1,008', 'root cross-sections', xs[0], 5.9, 3.0);
  stat(s, '200', 'lines with genotype', xs[1], 5.9, 3.0);
  stat(s, '137', 'lines with both', xs[2], 5.9, 2.6);
  stat(s, '27', 'kept out of training', xs[3], 5.9, 2.8, ROSE);
  s.addNotes('So what is the model given? These lines all descend from the same eight founders, so every genome is a mosaic of those eight, and genotyping tells us which founder each stretch came from. Colour them and you get the block on the left - all 43,788 markers of the line from the opening, wrapped into rows so they fit. The wide bands are stretches inherited whole from one founder.\n\nOn the right is one of that line\'s actual roots. That pairing - a genome and a picture of the plant it produced - is the whole dataset, repeated.\n\nThe numbers: a thousand images, two hundred genotyped lines, but only 137 with both. And 27 of those I kept out of training entirely, so there is always a set the model has never seen.');
}

/* --------------------------------------------------------- act 3: the three pieces */

// 9 - overview
{
  const s = slide();
  eyebrow(s, 'HOW IT WORKS  ·  THE WHOLE THING');
  title(s, 'Three pieces, and what each one does.');
  imageFit(s, talk('fig_three_pieces.png'), 0.55, 1.85, 12.25, 4.9);
  s.addNotes('Here is the whole thing on one slide, and then I will take the pieces one at a time. The compressor shrinks an image to something small enough to learn from. The genotype reader turns a genome into eight numbers. The generator builds a new compressed image from noise, steered by those eight numbers, and the compressor expands it back into a picture.');
}

// 10 - piece 1, LiteVAE
{
  const s = slide();
  eyebrow(s, 'PIECE 1  ·  LITEVAE, THE COMPRESSOR');
  title(s, 'Shrink the image before you learn it.');
  // The real thing rather than the schematic: one root beside the four 32x32
  // grids it compresses to, which is the whole point of the piece.
  imageFit(s, talk('fig_latent_channels_real.png'), 0.7, 1.8, 11.93, 3.0);
  pieceCards(s,
    'A root image.\n3 × 256 × 256 = 196,608 numbers.',
    'A compressed version, 4 × 32 × 32 = 4,096 numbers, and a decoder that turns it back into a picture.',
    'WHY',
    'With ~700 training images, learning 196,608 numbers directly is hopeless. Compressing 48× makes the problem small enough — and the traits still survive at r = 0.94–0.99.');
  s.addNotes('Piece one: LiteVAE, the compressor. In goes a root image - nearly 200,000 numbers. Out comes a compressed version of about 4,000, plus a decoder that can turn that back into a picture.\n\nWhy do it at all? Because with roughly 700 training images, no model is going to learn the distribution of 200,000 numbers. Compressing 48-fold makes it tractable. It is trained on root images alone, with no genotype involved, and then frozen.\n\nThe obvious worry is that compression throws away the traits we care about, so we checked: measure each trait on the original, measure it again after compressing and rebuilding, and correlate. Everything survives between 0.94 and 0.99. So if something fails later, we know it is not the compression.');
}

// 11 - inside the compressor
{
  const s = slide();
  eyebrow(s, 'PIECE 1  ·  HOW THE COMPRESSION HAPPENS');
  title(s, 'Watch one root get packed into four small grids.');
  // The architecture with the real activation sitting at every stage, so the
  // boxes and the pictures are one story instead of two slides.
  imageFit(s, talk('fig_encoder_walkthrough_real.png'), 0.55, 1.75, 12.25, 5.35);
  s.addNotes('Before I leave the compressor, this is what it actually does to the root on the last slide - every panel here is the real thing, not a schematic.\n\nIt starts with a wavelet transform, which is fixed, not learned: it splits the image into coarse shape and fine detail at three scales. You can see the vessels stay sharp in the first one and fade as the scales get coarser. A small network reads each scale - that is the bottom row - the three get resampled to the same size and stacked, and one last block turns that stack into the four 32 by 32 grids on the right.\n\nThe thing to notice is that you can still see a root in nearly every panel. Right up to the last step, the anatomy is being packed into fewer numbers rather than thrown away - which is why the traits come back out at 0.94 to 0.99.\n\n[If asked: the model predicts a mean and a spread, and z is a sample from those - the one step in here that is not deterministic.]');
}

// 12 - piece 2, the SNP encoder
{
  const s = slide();
  eyebrow(s, 'PIECE 2  ·  THE SNP ENCODER');
  title(s, 'Turn a genome into eight numbers.');
  imageFit(s, talk('fig_onehot_diagram.png'), 0.7, 1.8, 11.93, 3.0);
  pieceCards(s,
    'One line’s genotype: 43,788 markers, each naming the founder it came from.',
    'Eight tokens — short summaries the image model can read at every step.',
    'WHY AN ENCODER AT ALL',
    '350,000 columns against 110 lines invites memorising, and most markers say nothing about anatomy. Training the encoder with the generator keeps the ones that do.');
  s.addNotes('Piece two: the SNP encoder. In goes a genotype - 43,788 markers, each naming which founder that stretch came from. Out come eight tokens: short summaries that the image model reads.\n\nWhy have this piece at all - why not hand the genome straight to the generator? Two reasons.\n\nThe first is size. Each marker gets eight slots with one ticked, which is 350,000 columns, and I have 110 lines with images. A model given 350,000 inputs and 110 examples does not learn a rule, it memorises the examples - and we have watched it do exactly that. PCA takes that down to 168 numbers while keeping 95 percent of the variation, and a small network turns those into eight tokens.\n\nThe second is that most of the genome is not doing anything here. Very few markers have any bearing on root anatomy, and which ones they are is not known in advance. Because the encoder is trained together with the generator, it is under pressure to keep whatever actually helps predict the image and ignore the rest - rather than being fixed beforehand by me guessing.\n\n[If asked about the one-hot step: founder codes are labels, not amounts. Fed in as 1 to 8 the model is told founder 8 is eight times founder 1. Giving each marker eight slots put every founder the same distance apart, and agreement with true genetic similarity went from 0.49 to 0.98.]');
}

// 12 - piece 3, the diffusion model
{
  const s = slide();
  eyebrow(s, 'PIECE 3  ·  THE DIFFUSION MODEL');
  title(s, 'Start from static. Remove noise until a root is left.');
  imageFit(s, talk('fig_sampling_trajectory_real.png'), 0.7, 1.8, 11.93, 3.0);
  pieceCards(s,
    'A patch of random noise, the eight genotype tokens, and how noisy the picture currently is.',
    'A little less noise each round. After fifty the static has resolved into an outer boundary, a cortex and a ring of vessels — a root cross-section.',
    'WHY DIFFUSION',
    'It trains stably on a few hundred images, it takes conditioning naturally at every step, and a fixed starting noise always gives the same image — which is what makes controlled comparisons possible.');
  s.addNotes('Piece three: the diffusion model, the generator. It is handed a patch of random noise, the eight genotype tokens, and a number saying how noisy the picture currently is, and it takes a little of the noise away. Do that fifty times and the static has turned into something with an outer boundary, a cortex and a ring of vessels in the middle - an actual root cross-section, not a blur that suggests one.\n\nThis strip is a real run, decoded at a few points along the way. Watch where it happens: nothing recognisable for the first two thirds, then the boundary appears, then the stele, and the individual vessels are the last thing to arrive, in the final ten steps or so.\n\nWhy diffusion rather than the alternatives? It trains stably on a few hundred images, where a GAN typically will not. It takes conditioning naturally: the genotype tokens are consulted at every one of the fifty steps rather than only at the start. And it is deterministic - same tokens, same starting noise, same image every time - which is what lets us change one thing and see what moves.');
}

// 13 - under the hood
{
  const s = slide();
  eyebrow(s, 'FOR CONTEXT  ·  UNDER THE HOOD');
  title(s, 'What the generator actually looks like.');
  imageFit(s, talk('fig_unet_diagram.png'), 0.55, 1.9, 12.25, 4.1);
  body(s, 'Every width and resolution here is read from the trained model, not drawn by hand. You do not need to read it — the point is that the genotype tokens are consulted at five places and at several scales, from a 16×16 view down to a 2×2 one, rather than being mentioned once at the start.', {
    x: 0.7, y: 6.15, w: 11.93, h: 1.0, fontSize: 15 });
  s.addNotes('For context, since some of you will want to see it: this is the generator, with every width and resolution read from the trained model rather than drawn by hand. I am not going to walk through it. The one thing worth pointing at is the blue: those are the places the genotype tokens are read, five of them, at several scales - from a 16 by 16 view of the image down to a 2 by 2 one. So the genotype can act on the overall layout and on local detail, rather than being mentioned once at the start and then forgotten.\n\nThe compressor and the genotype reader have diagrams of their own if anyone wants them afterwards.');
}

// 14 - measuring what it drew
{
  const s = slide();
  eyebrow(s, 'MEASURING WHAT IT DREW');
  title(s, 'A generated root can be measured like a real one.');
  imageFit(s, talk('fig_segmentation_demo.png'), 0.55, 1.85, 12.25, 3.9);
  panel(s, 0.7, 5.95, 11.93, 1.05, true);
  s.addText('The same segmentation model measures both: root and stele diameter, vessel count and vessel area. On images it never trained on it is within 0.6 vessels and about 2 px of the hand annotations — so differences between a real root and a generated one can be attributed to the generator, not the ruler.', {
    x: 1.0, y: 6.18, w: 11.3, h: 0.65, isTextBox: true, margin: 0,
    fontFace: SANS, fontSize: 15, color: WHITE });
  s.addNotes('One more piece, and it is the one that turns pictures into data. A second model - a segmenter - outlines the root, the stele and each vessel, and measures them: diameters, vessel count, vessel area. Red is root, green is stele, blue is vessels.\n\nIt runs on a generated image exactly as it runs on a photograph, which is what makes the whole thing quantitative rather than a matter of looking at pictures and nodding. We cross-validated it: every annotated image scored by a model that never trained on it, and it lands within about 0.6 vessels and 2 pixels of the hand annotations. So when a generated root differs from a real one, that difference is the generator, not the ruler.\n\nThe numbers on the right are for this one pair. A single generated image is a sample for that genotype, not a reconstruction of that particular photograph, so per-image differences are expected - it is the genotype-level agreement that matters.');
}

// 15 - where a gene acts
{
  const s = slide();
  eyebrow(s, 'WHAT YOU CAN ASK IT  ·  GENOTYPE TO PHENOTYPE');
  title(s, 'Where the genotype tokens are looking.', 0.88);
  // The generator's own attention rather than a heat map over a decoded root.
  // One row per locus, so the variety between them is the point of the slide
  // and the figure carries its own caveat line.
  imageFit(s, talk('attention_tokens.png'), 0.55, 1.74, 12.25, 4.8);
  body(s, 'One row per locus, eight tokens each. The model looks in visibly different places depending on which part of the genome differs — a tight inner ring, a wide outer ring, a ring through the cortex, a single central spot.', {
    x: 0.7, y: 6.68, w: 11.93, h: 0.6, fontSize: 15 });
  s.addNotes('This is the genotype-to-phenotype question, and it is the reason to build a model like this rather than just predict a number.\n\nNormally you go from genotype to phenotype one trait at a time: you decide in advance to measure root diameter, you run the association, and you get a number per locus saying how much it moves that one trait. If the locus does something you did not think to measure, you never see it.\n\nHere the phenotype is the whole image, so you can ask the question the other way round - and you can go further and look inside the model while it draws.\n\nThe eight genotype tokens are read at five places in the generator. What you are looking at is one of those places, a 16 by 16 view of the image, and how much attention each token puts on each patch. Red is more attention, blue is less. Each row is a different locus, and the four were chosen out of two dozen for how much their patterns differ from one another.\n\nLook down the rows rather than across them. The first is a tight ring right around the stele. The second is a much wider ring out at the edge of the cortex. The third sits through the middle of the cortex. The fourth is not a ring at all - it is a single spot in the centre. Different parts of the genome, different places in the anatomy.\n\nNow the caveat, which is on the figure because I do not want it lost. This compares lines that carry a founder allele against lines that do not. That is a correlation, not an intervention: those lines differ at thousands of other loci too, so I cannot hand you one of these rings and say this locus causes it.\n\nI can do the clean version - flip one locus, change nothing else, regenerate - and I have. At this checkpoint a single locus barely moves the attention and that map comes out as speckle. Being able to tell those two apart is itself the argument for a model you can intervene on, and it is what an association study on its own cannot give you.\n\nAnd the standing caveat: this tells you what the model has learned, which is only as good as the model. Hypotheses worth testing, not results.');
}

/* --------------------------------------------------------- act 4: where it goes */

// 16 - the generic recipe
{
  const s = slide();
  eyebrow(s, 'WHERE ELSE THIS FITS');
  title(s, 'Nothing in the middle is about roots.');
  imageFit(s, talk('fig_generic_recipe.png'), 0.55, 1.9, 12.25, 4.3);
  body(s, 'The requirement is a pairing: a cheap measurement that identifies a subject, and an expensive image of that subject — with enough distinct subjects to learn from.', {
    x: 0.7, y: 6.35, w: 11.93, h: 0.7, fontSize: 16 });
  s.addNotes('Here is the part I find most interesting. Nothing in the middle of this pipeline is about roots. The compressor does not care what it is compressing. The generator does not care what it is generating. The genotype reader is the only piece that knows anything about plants, and it is the smallest of the three.\n\nWhat the recipe needs is a pairing: a cheap measurement that identifies a subject, and an expensive image of that subject. For us that is a genotype and a root cross-section. Swap those ends and the same machinery applies - a patient genotype or diagnosis and a brain scan, a treatment and dose and a histology slide.\n\nThe one requirement that does not move is the number of distinct subjects. We have 110 lines with images to learn from, and that is the constraint we are working against - not the number of images.');
}

// 17 - conclusion
{
  const s = slide();
  // The whole talk in one line, over the one transformation it is about: the
  // same genome block from the data slide, and a root that was never grown.
  s.addText('Genotype in. Anatomy out.', {
    x: 0.7, y: 0.85, w: 11.93, h: 1.0, isTextBox: true, margin: 0, align: 'center',
    fontFace: SERIF, fontSize: 46, color: WHITE });

  imageFit(s, hook('genome_block.png'), 2.74, 2.63, 4.0, 1.34);
  s.addText('→', {
    x: 6.74, y: 2.75, w: 1.1, h: 1.1, isTextBox: true, margin: 0, align: 'center',
    fontFace: SANS, fontSize: 40, color: AMBER });
  s.addImage({ path: generated[0], x: 7.99, y: 2.0, w: 2.6, h: 2.6 });
  body(s, 'ONE GENOTYPE  ·  43,788 MARKERS', {
    x: 2.74, y: 4.75, w: 4.0, h: 0.3, align: 'center', fontSize: 10, color: MUTED, charSpacing: 2 });
  body(s, 'NO PLANT GROWN', {
    x: 7.99, y: 4.75, w: 2.6, h: 0.3, align: 'center', fontSize: 10, color: MUTED, charSpacing: 2 });

  s.addText('Weeks of greenhouse, harvest and microscope.\nOr a few seconds on a laptop.', {
    x: 0.7, y: 5.3, w: 11.93, h: 1.1, isTextBox: true, margin: 0, align: 'center',
    fontFace: SERIF, fontSize: 27, color: WHITE, lineSpacingMultiple: 1.15 });
  body(s, 'And the middle of this pipeline never knew it was looking at roots.', {
    x: 0.7, y: 6.55, w: 11.93, h: 0.5, fontSize: 19, color: AMBER, align: 'center' });
  s.addNotes('So, to close. A genotype goes in and an anatomy comes out. The thing on the left is a text file. The thing on the right is a root that was never grown - and it was drawn in the time it takes to say this sentence, on the laptop in front of me.\n\nThe measurement it stands in for takes weeks: grow the plant, harvest it, fix it, section it, image it, one plant at a time.\n\nAnd the part I want you to leave with: the middle of this pipeline never knew it was looking at roots. Whatever your expensive image is, and whatever cheap label comes with it, the same three pieces are sitting there waiting.\n\nThank you.');
}

// 18 - questions
{
  const s = slide();
  s.addText('Questions?', {
    x: 0.7, y: 2.9, w: 11.93, h: 1.6, isTextBox: true, margin: 0, align: 'center',
    fontFace: SERIF, fontSize: 60, color: WHITE });
  s.addImage({ path: hook('snp_strip_full.png'), x: 0.7, y: 5.6, w: 11.93, h: 0.45 });
  s.addNotes('[Leave this up. Likely questions: how many lines would it take; whether it works on other traits; whether the generated images could be used as training data; what happens with a genotype outside the population.]');
}

pres.writeFile({ fileName: OUT }).then(() => console.log('wrote ' + OUT));
