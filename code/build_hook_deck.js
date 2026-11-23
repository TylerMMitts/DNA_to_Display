// Builds the opening-hook deck for the conference talk.
//
// Run it after code/make_hook_assets.py, which produces the images this reads:
//     python code/make_hook_assets.py
//     node code/build_hook_deck.js
//
// The narration for each slide goes in the speaker notes, so the slides
// themselves stay sparse - the point of the hook is the images, not the text.

const path = require('path');
const pptxgen = require('pptxgenjs');

const ROOT = path.resolve(__dirname, '..');
const A = path.join(ROOT, 'results', 'hook_slides', 'assets');
const OUT = path.join(ROOT, 'results', 'hook_slides', 'DNA_to_Display_hook.pptx');

// Dark throughout: the root images are fluorescence on black, so a dark deck
// lets them sit on the slide without a box around them.
const BG = '0B1020';
const INK = 'CADCFC';
const WHITE = 'FFFFFF';
const AMBER = 'F2A65A';
const MUTED = '7C8DB0';

const SERIF = 'Cambria';
const SANS = 'Calibri';

// Same eight colours make_hook_assets.py paints the SNP strips with, so the
// legend and the strips agree. Drawn as shapes rather than an image: the label
// widths are known here, and it stays editable in PowerPoint.
const FOUNDER_COLORS = ['4C9BE8', 'F2A65A', '5FD3A6', 'E8607A',
                        'B48BE8', 'F2D65A', '5FC8D3', 'E88A4C'];

// Slide is 13.333 x 7.5.
const W = 13.333;
const asset = (name) => path.join(A, name);

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.author = 'Tyler Mitts';
pres.title = 'DNA to Display - hook';

function slide() {
  const s = pres.addSlide();
  s.background = { color: BG };
  return s;
}

// A small amber eyebrow label, the one repeated motif across the deck.
function eyebrow(s, text, y) {
  s.addText(text, {
    x: 0.7, y: y, w: 6.0, h: 0.3, isTextBox: true, margin: 0,
    fontFace: SANS, fontSize: 14, bold: true, color: AMBER, charSpacing: 3,
  });
}

// 1 - the fingerprint
{
  const s = slide();
  eyebrow(s, 'MEMA244', 0.55);
  s.addText('A genetic fingerprint', {
    x: 0.7, y: 0.92, w: 11.9, h: 0.75, isTextBox: true, margin: 0,
    fontFace: SERIF, fontSize: 40, color: WHITE,
  });

  s.addText('WHOLE GENOME  ·  43,788 MARKERS', {
    x: 0.7, y: 2.05, w: 6.0, h: 0.25, isTextBox: true, margin: 0,
    fontFace: SANS, fontSize: 10, color: MUTED, charSpacing: 2,
  });
  s.addImage({ path: asset('snp_strip_full.png'), x: 0.7, y: 2.35, w: 11.93, h: 0.5 });

  s.addText('ZOOMED TO 260 MARKERS', {
    x: 0.7, y: 3.05, w: 6.0, h: 0.25, isTextBox: true, margin: 0,
    fontFace: SANS, fontSize: 10, color: MUTED, charSpacing: 2,
  });
  s.addImage({ path: asset('snp_strip_zoom.png'), x: 0.7, y: 3.35, w: 11.93, h: 0.5 });

  s.addText(
    'Each marker is inherited from one of eight founder lines. The colour blocks are real haplotype segments — a median run is 42 markers long.',
    {
      x: 0.7, y: 4.15, w: 11.9, h: 0.6, isTextBox: true, margin: 0,
      fontFace: SANS, fontSize: 15, color: INK,
    });

  s.addText('FOUNDER', {
    x: 0.7, y: 5.12, w: 1.1, h: 0.3, isTextBox: true, margin: 0, valign: 'middle',
    fontFace: SANS, fontSize: 10, color: MUTED, charSpacing: 2,
  });
  FOUNDER_COLORS.forEach((c, i) => {
    const x = 1.85 + i * 0.78;
    s.addShape(pres.ShapeType.rect, {
      x: x, y: 5.16, w: 0.24, h: 0.24, fill: { color: c }, line: { color: c },
    });
    s.addText(String(i + 1), {
      x: x + 0.3, y: 5.12, w: 0.4, h: 0.3, isTextBox: true, margin: 0,
      valign: 'middle', fontFace: SANS, fontSize: 13, color: INK,
    });
  });
  s.addNotes('Every plant in this study carries a genetic fingerprint: 43,788 markers across its genome, each one inherited from one of eight founder lines. This is the fingerprint of one maize line, MEMA244.');
}

// 2 - four roots from that genotype
{
  const s = slide();
  eyebrow(s, 'MEMA244', 0.55);
  s.addText('Four primary roots', {
    x: 0.7, y: 0.92, w: 11.9, h: 0.75, isTextBox: true, margin: 0,
    fontFace: SERIF, fontSize: 40, color: WHITE,
  });

  // One row of four, 2.75" square with 0.36" gaps, centred on the slide.
  const size = 2.75, gap = 0.36;
  const total = size * 4 + gap * 3;
  const x0 = (W - total) / 2;
  for (let i = 0; i < 4; i += 1) {
    s.addImage({
      path: asset(`generated_seed${i}.png`),
      x: x0 + i * (size + gap), y: 2.15, w: size, h: size,
      rounding: false,
    });
  }

  s.addText(
    'Same genotype, same DNA. Look how much they differ: the vessel ring, the air spaces in the cortex, even the colour.',
    {
      x: 0.7, y: 5.25, w: 11.9, h: 0.6, isTextBox: true, margin: 0,
      fontFace: SANS, fontSize: 16, color: INK,
    });
  s.addNotes('And these are four primary roots from that genotype. Same DNA, but look how much they differ: the vessels, the air spaces in the cortex, even the colour.\n\n[Pause on the grid for a beat.]');
}

// 3 - the confession
{
  const s = slide();
  s.addText('One confession:', {
    x: 0.9, y: 2.45, w: 11.5, h: 0.7, isTextBox: true, margin: 0,
    fontFace: SANS, fontSize: 24, color: AMBER,
  });
  s.addText('None of these roots\nwere ever grown.', {
    x: 0.9, y: 3.15, w: 11.5, h: 2.0, isTextBox: true, margin: 0,
    fontFace: SERIF, fontSize: 54, color: WHITE, lineSpacingMultiple: 1.1,
  });
  s.addNotes('One confession: none of these roots were ever grown.\n\n[Beat.]');
}

// 4 - what actually made them
{
  const s = slide();
  s.addText('No seed was planted. No section was cut.', {
    x: 0.7, y: 0.75, w: 11.9, h: 0.6, isTextBox: true, margin: 0,
    fontFace: SERIF, fontSize: 32, color: WHITE,
  });
  s.addText('Every one of these images was generated from that genetic fingerprint, and nothing else.', {
    x: 0.7, y: 1.45, w: 11.9, h: 0.5, isTextBox: true, margin: 0,
    fontFace: SANS, fontSize: 17, color: INK,
  });

  // fingerprint -> model -> the four images, built natively so it stays
  // editable. The left half is centred on the 2x2 grid to its right.
  s.addImage({ path: asset('snp_strip_zoom.png'), x: 0.7, y: 3.54, w: 3.1, h: 0.62 });
  s.addText('GENOTYPE', {
    x: 0.7, y: 4.24, w: 3.1, h: 0.25, isTextBox: true, margin: 0,
    fontFace: SANS, fontSize: 10, color: MUTED, charSpacing: 2,
  });

  s.addShape(pres.ShapeType.rightArrow, {
    x: 4.0, y: 3.72, w: 0.7, h: 0.26, fill: { color: AMBER },
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 4.95, y: 3.29, w: 2.5, h: 1.12, rectRadius: 0.12,
    fill: { color: '17203A' }, line: { color: AMBER, width: 1.25 },
  });
  s.addText('diffusion\nmodel', {
    x: 4.95, y: 3.29, w: 2.5, h: 1.12, isTextBox: true, align: 'center',
    valign: 'middle', fontFace: SANS, fontSize: 16, bold: true, color: WHITE,
  });

  s.addShape(pres.ShapeType.rightArrow, {
    x: 7.7, y: 3.72, w: 0.7, h: 0.26, fill: { color: AMBER },
  });

  // 2x2 rather than a row: four in a row runs off the right edge of the slide.
  const size = 1.62, gap = 0.14;
  for (let i = 0; i < 4; i += 1) {
    s.addImage({
      path: asset(`generated_seed${i}.png`),
      x: 8.9 + (i % 2) * (size + gap), y: 2.16 + Math.floor(i / 2) * (size + gap),
      w: size, h: size,
    });
  }
  s.addText('FOUR NOISE SEEDS, ONE GENOTYPE', {
    x: 8.9, y: 5.62, w: 4.0, h: 0.25, isTextBox: true, margin: 0,
    fontFace: SANS, fontSize: 10, color: MUTED, charSpacing: 2,
  });

  s.addText('The only input is the genotype. The four differ because each starts from a different random noise draw — not because anything about the plant changed.', {
    x: 0.7, y: 6.25, w: 11.9, h: 0.6, isTextBox: true, margin: 0,
    fontFace: SANS, fontSize: 15, color: INK,
  });
  s.addNotes('No seed was planted. No section was cut. Every one of these images was generated by a computer from that genetic fingerprint, and nothing else.');
}

// 5 - the real root
{
  const s = slide();
  eyebrow(s, 'MEMA244', 0.55);
  s.addText('This is a real root.', {
    x: 0.7, y: 0.92, w: 11.9, h: 0.75, isTextBox: true, margin: 0,
    fontFace: SERIF, fontSize: 40, color: WHITE,
  });
  s.addImage({ path: asset('real_root.png'), x: 4.87, y: 2.1, w: 3.6, h: 3.6 });
  s.addText('MEMA244_primary_2_2 · photographed, sectioned, and never shown to the model', {
    x: 0.7, y: 5.95, w: 11.9, h: 0.4, isTextBox: true, margin: 0, align: 'center',
    fontFace: SANS, fontSize: 13, color: MUTED,
  });
  s.addNotes('This is a real root from MEMA244.');
}

// 6 - side by side, and what the model was trained on
{
  const s = slide();
  s.addText('Generated', {
    x: 0.7, y: 0.75, w: 5.6, h: 0.45, isTextBox: true, margin: 0,
    fontFace: SANS, fontSize: 15, bold: true, color: AMBER, charSpacing: 2,
  });
  s.addText('Real', {
    x: 8.6, y: 0.75, w: 4.0, h: 0.45, isTextBox: true, margin: 0,
    fontFace: SANS, fontSize: 15, bold: true, color: INK, charSpacing: 2,
  });

  // 2x2 of the generated, one large real one beside it
  const size = 1.95, gap = 0.18;
  for (let i = 0; i < 4; i += 1) {
    s.addImage({
      path: asset(`generated_seed${i}.png`),
      x: 0.7 + (i % 2) * (size + gap), y: 1.3 + Math.floor(i / 2) * (size + gap),
      w: size, h: size,
    });
  }
  s.addImage({ path: asset('real_root.png'), x: 8.6, y: 1.3, w: 4.08, h: 4.08 });

  s.addText('The model had never seen this genotype.', {
    x: 5.05, y: 1.55, w: 3.35, h: 1.2, isTextBox: true, margin: 0,
    fontFace: SERIF, fontSize: 23, color: WHITE, lineSpacingMultiple: 1.1,
  });
  s.addText([
    { text: '~700', options: { fontSize: 30, bold: true, color: AMBER, breakLine: true } },
    { text: 'root cross-sections', options: { fontSize: 13, color: INK, breakLine: true } },
    { text: '110', options: { fontSize: 30, bold: true, color: AMBER, breakLine: true } },
    { text: 'other maize lines', options: { fontSize: 13, color: INK } },
  ], {
    x: 5.05, y: 3.05, w: 3.35, h: 2.2, isTextBox: true, margin: 0,
    fontFace: SANS, lineSpacingMultiple: 1.0,
  });

  s.addText('From those it worked out what a maize root looks like, and how to draw one from a genotype it had never encountered.', {
    x: 0.7, y: 5.75, w: 11.96, h: 0.6, isTextBox: true, margin: 0,
    fontFace: SANS, fontSize: 15, color: INK,
  });
  s.addNotes('The model that made those images had never seen this genotype. It learned from about 700 root cross-sections across 110 other lines, and figured out on its own what a maize root looks like and how to draw one from a genotype.');
}

// 7 - the question the talk answers
{
  const s = slide();
  s.addImage({ path: asset('snp_strip_zoom.png'), x: 1.15, y: 1.35, w: 2.7, h: 0.54 });
  s.addText('SNP FINGERPRINT', {
    x: 1.15, y: 1.98, w: 2.7, h: 0.25, isTextBox: true, margin: 0, align: 'center',
    fontFace: SANS, fontSize: 10, color: MUTED, charSpacing: 2,
  });

  s.addShape(pres.ShapeType.rightArrow, {
    x: 4.15, y: 1.49, w: 0.75, h: 0.26, fill: { color: AMBER },
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 5.2, y: 1.06, w: 2.9, h: 1.12, rectRadius: 0.12,
    fill: { color: '17203A' }, line: { color: AMBER, width: 1.25 },
  });
  s.addText('diffusion model', {
    x: 5.2, y: 1.06, w: 2.9, h: 1.12, isTextBox: true, align: 'center',
    valign: 'middle', fontFace: SANS, fontSize: 16, bold: true, color: WHITE,
  });

  s.addShape(pres.ShapeType.rightArrow, {
    x: 8.4, y: 1.49, w: 0.75, h: 0.26, fill: { color: AMBER },
  });

  s.addImage({ path: asset('generated_seed0.png'), x: 9.45, y: 0.79, w: 1.66, h: 1.66 });
  s.addText('ROOT IMAGE', {
    x: 9.45, y: 2.53, w: 1.66, h: 0.25, isTextBox: true, margin: 0, align: 'center',
    fontFace: SANS, fontSize: 10, color: MUTED, charSpacing: 2,
  });

  s.addText('When it gets something right,\nis it reading the DNA — or just drawing\na convincing maize root?', {
    x: 0.9, y: 3.5, w: 11.5, h: 2.1, isTextBox: true, margin: 0,
    fontFace: SERIF, fontSize: 34, color: WHITE, lineSpacingMultiple: 1.15,
  });
  s.addText('Let me show you how we found out.', {
    x: 0.9, y: 5.75, w: 11.5, h: 0.5, isTextBox: true, margin: 0,
    fontFace: SANS, fontSize: 18, color: AMBER,
  });
  s.addNotes('Which raises the question this talk is really about: when it gets something right, is it reading the DNA, or just drawing a convincing maize root? Let me show you how we found out.');
}

pres.writeFile({ fileName: OUT }).then(() => console.log('wrote ' + OUT));
