# Checks that a d2d dataset is ready to train, without training anything: run it on
# the cluster's login node before sbatch, or locally before copying.
#
#     python -m d2d.cluster_check <dataset>
#
# Reads exactly what the trainers read, through the launcher's path redirection:
# the genotype table and its cache, the marker map, the QTL table of the chosen
# encoding, the split, the prepared images and their metadata. Checks the
# launcher's in-memory patches still apply to the trainer sources. Builds no model
# and takes no optimiser step.
import os
import sys

import numpy as np
import pandas as pd

from .launch_trainer import PATCHES, TRAINERS, prepare
from .project import ROOT, Dataset

ok_all = True


def report(ok, what, detail=''):
    global ok_all
    ok_all &= bool(ok)
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}" + (f': {detail}' if detail else ''), flush=True)


def main(name):
    ds = Dataset(name)
    print(f'd2d cluster check: {name} in {ROOT}')
    report(ds.config_path.exists(), 'dataset config', str(ds.config_path))
    if not ds.config_path.exists():
        return
    cfg = ds.config()
    env = (cfg.get('training') or {}).get('env', {}) or (cfg.get('genetics') or {}).get('env', {})
    for k, v in env.items():
        os.environ.setdefault(k, str(v))
    prepare(ds)
    import widiv_paths as W

    for label, p in [('genotypes', W.WIDIV_GENOTYPES), ('marker map', W.WIDIV_SNP_MAP), ('split', W.WIDIV_SPLIT),
                     ('prepared images', W.WIDIV_SCALED_DIR), ('image metadata', W.WIDIV_SCALED_METADATA)]:
        report(p.exists(), label, str(p))

    # Prepared images: every listed file present, 512 px RGB.
    meta = pd.read_csv(W.WIDIV_SCALED_METADATA)
    missing = [f for f in meta.scaled_filename if not (W.WIDIV_SCALED_DIR / f).exists()]
    report(not missing, 'prepared image files', f'{len(meta) - len(missing)} of {len(meta)} present'
           + (f', e.g. missing {missing[:2]}' if missing else ''))
    from PIL import Image
    sample = meta.scaled_filename.sample(min(20, len(meta)), random_state=0)
    sizes = {Image.open(W.WIDIV_SCALED_DIR / f).size for f in sample if (W.WIDIV_SCALED_DIR / f).exists()}
    report(sizes == {(W.WIDIV_IMAGE_SIZE, W.WIDIV_IMAGE_SIZE)}, 'image size', str(sorted(sizes)))

    # Genotypes as the trainer loads them, and the encoding's QTL table.
    from widiv_snp_encoding import load_widiv_genotypes
    names, snps, dosage = load_widiv_genotypes(W.WIDIV_GENOTYPES, W.WIDIV_GENOTYPE_CACHE)
    report(len(names) > 0 and dosage.shape == (len(names), len(snps)), 'genotype table',
           f'{len(names)} lines x {len(snps)} markers, values {sorted(np.unique(dosage[:200]).tolist())[:5]}')
    q = os.environ.get('WIDIV_QTL_TABLE')
    if os.environ.get('WIDIV_SNP_ENCODING') in ('qtl_only', 'hybrid') or q:
        qp = (ROOT / q) if q and not os.path.isabs(q) else q
        if qp and os.path.exists(qp):
            cols = pd.read_csv(qp).iloc[:, 0].astype(str)
            report(cols.isin(set(snps)).all(), 'QTL table', f'{len(cols)} columns, '
                   f'{(~cols.isin(set(snps))).sum()} not in the genotype table')
        else:
            report(False, 'QTL table', f'{qp} not found')

    # Split over imaged, genotyped lines; the trainer takes its validation lines from training.
    split = pd.read_csv(W.WIDIV_SPLIT)
    imaged = set(meta.genotype.astype(str))
    known = imaged & set(map(str, names))
    report(len(known) == len(imaged), 'imaged lines have genotypes', f'{len(known)} of {len(imaged)}')
    col = W.WIDIV_SPLITS[0]
    n_test = int((split[col] == 'test').sum()) if col in split else -1
    report(col in split and 0 < n_test < len(split), 'split', f'{col}: {len(split) - n_test} train, {n_test} test')

    # The launcher's patches must still apply to the trainer sources.
    import re
    for which, path in TRAINERS.items():
        src = path.read_text(encoding='utf-8')
        n = [len(re.findall(p, src)) for p, _ in PATCHES[which]]
        report(all(k == 1 for k in n), f'{which} trainer patch', path.name)

    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none on this node'
        report(True, 'torch', f'{torch.__version__}, GPU {gpu}')
    except Exception as e:  # noqa: BLE001
        report(False, 'torch', str(e))
    print('READY to submit' if ok_all else 'NOT READY: fix the FAIL lines first')
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: python -m d2d.cluster_check <dataset>')
    main(sys.argv[1])
