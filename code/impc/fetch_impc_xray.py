# Download the IMPC whole-body dorso-ventral X-rays of one phenotyping centre, with
# everything needed to train and to score: per-image metadata, the centre's X-ray
# measurements per mouse, the IMPC's significant genotype-phenotype calls, and gene
# positions. Public data from the International Mouse Phenotyping Consortium
# (www.mousephenotype.org, CC BY 4.0), through its Solr API and the BioImage Archive
# mirror of the images.
#
#   python code/impc/fetch_impc_xray.py [--centre BCM] [--workers 8] [--metadata-only]
#
# Writes dataset/impc_<centre>/:
#   images/<observation_id>.jpg                the X-rays as published (JPEG)
#   metadata/impc_xray_images.csv              one row per image: mouse, gene, allele,
#                                              zygosity, sex, control/experimental, dates
#   metadata/impc_xray_measurements.csv        long table: mouse x X-ray parameter
#                                              (tibia length, rib and vertebra counts,
#                                              categorical calls such as skull shape)
#   metadata/impc_gene_positions.csv           gene, chromosome, start (GRCm39)
#   answer_key/impc_significant_calls.csv      the IMPC's own significant calls for this
#                                              centre's genes; for scoring only, never an
#                                              input to a model
# Re-running skips images already on disk, so an interrupted download resumes.
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import DATASET_DIR  # noqa: E402

SOLR = 'https://www.ebi.ac.uk/mi/impc/solr'
DV_PARAMETER = 'IMPC_XRY_034_001'          # "XRay Images Whole Body Dorso Ventral"
IMAGE_FIELDS = ['observation_id', 'specimen_id', 'external_sample_id', 'colony_id', 'litter_id',
                'gene_symbol', 'gene_accession_id', 'allele_symbol', 'zygosity', 'sex',
                'biological_sample_group', 'strain_name', 'genetic_background', 'date_of_birth',
                'date_of_experiment', 'age_in_days', 'age_in_weeks', 'metadata_group',
                'phenotyping_center', 'pipeline_stable_id', 'file_type', 'jpeg_url',
                'download_file_path']
MEASURE_FIELDS = ['specimen_id', 'external_sample_id', 'gene_symbol', 'zygosity', 'sex',
                  'biological_sample_group', 'parameter_stable_id', 'parameter_name',
                  'observation_type', 'data_point', 'category', 'date_of_experiment', 'metadata_group']
CALL_FIELDS = ['marker_symbol', 'marker_accession_id', 'allele_symbol', 'zygosity', 'sex',
               'procedure_name', 'procedure_stable_id', 'parameter_name', 'parameter_stable_id',
               'mp_term_id', 'mp_term_name', 'top_level_mp_term_name', 'p_value', 'effect_size',
               'phenotyping_center']


# Each Solr core's unique key, which cursor paging must sort on.
SORT_KEY = {'impc_images': 'id', 'experiment': 'id', 'genotype-phenotype': 'doc_id', 'gene': 'mgi_accession_id'}


def solr_all(core, query, fields, page=2000):
    # Every matching document, paged with a cursor so the result does not shift underneath.
    docs, cursor = [], '*'
    while True:
        params = {'q': query, 'fl': ','.join(fields), 'rows': page, 'wt': 'json', 'sort': f'{SORT_KEY[core]} asc',
                  'cursorMark': cursor}
        url = f'{SOLR}/{core}/select?' + urllib.parse.urlencode(params)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    data = json.load(r)
                break
            except Exception as e:                          # network hiccup: back off and retry
                if attempt == 4:
                    raise
                print(f'  retry {attempt + 1} ({e})', flush=True)
                time.sleep(5 * (attempt + 1))
        docs += data['response']['docs']
        nxt = data['nextCursorMark']
        print(f'  {core}: {len(docs):,} / {data["response"]["numFound"]:,}', flush=True)
        if nxt == cursor:
            return pd.DataFrame(docs)
        cursor = nxt


def flat(df):
    # List-valued Solr fields (e.g. top_level_mp_term_name) as one ';'-joined string.
    for c in df.columns:
        if df[c].map(lambda v: isinstance(v, list)).any():
            df[c] = df[c].map(lambda v: ';'.join(map(str, v)) if isinstance(v, list) else v)
    return df


def download(url, path, retries=4):
    tmp = path.with_suffix('.part')
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r, open(tmp, 'wb') as f:
                f.write(r.read())
            if tmp.stat().st_size < 5000:
                raise IOError(f'only {tmp.stat().st_size} bytes')
            tmp.replace(path)
            return path.stat().st_size
        except Exception as e:
            if attempt == retries - 1:
                tmp.unlink(missing_ok=True)
                return f'failed: {e}'
            time.sleep(3 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--centre', default='BCM')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--metadata-only', action='store_true')
    ap.add_argument('--images-only', action='store_true',
                    help='download images listed in an existing impc_xray_images.csv')
    args = ap.parse_args()
    root = DATASET_DIR / f"impc_{args.centre.lower().replace(' ', '_')}"
    (root / 'images').mkdir(parents=True, exist_ok=True)
    (root / 'metadata').mkdir(exist_ok=True)
    (root / 'answer_key').mkdir(exist_ok=True)
    centre = f'phenotyping_center:"{args.centre}"'
    if args.images_only:
        download_images(pd.read_csv(root / 'metadata' / 'impc_xray_images.csv'), root, args.workers)
        return

    print('Image records')
    img = flat(solr_all('impc_images', f'parameter_stable_id:{DV_PARAMETER} AND {centre}', IMAGE_FIELDS))
    img['gene_symbol'] = img['gene_symbol'].where(img.biological_sample_group == 'experimental', 'wildtype')
    img.to_csv(root / 'metadata' / 'impc_xray_images.csv', index=False)

    print('X-ray measurements')
    meas = flat(solr_all('experiment', f'procedure_group:IMPC_XRY AND {centre}', MEASURE_FIELDS))
    meas.to_csv(root / 'metadata' / 'impc_xray_measurements.csv', index=False)

    print('Significant calls (answer key)')
    genes = sorted(set(img.gene_symbol) - {'wildtype'})
    calls = flat(solr_all('genotype-phenotype', centre, CALL_FIELDS))
    calls = calls[calls.marker_symbol.isin(genes)]
    calls.to_csv(root / 'answer_key' / 'impc_significant_calls.csv', index=False)

    print('Gene positions')
    gpos = flat(solr_all('gene', 'marker_symbol:(' + ' OR '.join(f'"{g}"' for g in genes) + ')',
                         ['marker_symbol', 'chr_name', 'seq_region_start', 'seq_region_end'], page=1000)
                if len(genes) < 1500 else pd.DataFrame())
    gpos.to_csv(root / 'metadata' / 'impc_gene_positions.csv', index=False)

    print(f'{len(img):,} images of {img.specimen_id.nunique():,} mice, {len(genes)} genes, '
          f'{(img.biological_sample_group == "control").sum():,} control images; '
          f'{meas.specimen_id.nunique():,} mice with X-ray measurements; '
          f'{calls.marker_symbol.nunique()} genes with significant calls ({len(calls)} calls)')
    if not args.metadata_only:
        download_images(img, root, args.workers)


def download_images(img, root, workers):
    # The index lists each BCM image twice under the same observation_id (identical
    # pixels), so download each observation once.
    todo = [(r.jpeg_url, root / 'images' / f'{r.observation_id}.jpg')
            for r in img.drop_duplicates('observation_id').itertuples()
            if isinstance(r.jpeg_url, str) and not (root / 'images' / f'{r.observation_id}.jpg').exists()]
    print(f'Downloading {len(todo):,} images with {workers} workers '
          f'({img.observation_id.nunique() - len(todo):,} already on disk or without a JPEG)')
    done, failed, total = 0, [], 0
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(download, u, p): (u, p) for u, p in todo}
        for f in as_completed(futures):
            res = f.result()
            if isinstance(res, int):
                done += 1
                total += res
            else:
                failed.append((futures[f][0], res))
            if (done + len(failed)) % 500 == 0:
                print(f'  {done + len(failed):,} / {len(todo):,}  ({total / 1e9:.2f} GB, {len(failed)} failed)', flush=True)
    pd.DataFrame(failed, columns=['url', 'error']).to_csv(root / 'metadata' / 'download_failures.csv', index=False)
    print(f'Done: {done:,} downloaded ({total / 1e9:.2f} GB), {len(failed)} failed')


if __name__ == '__main__':
    main()
