# Intake: turn an arbitrary dataset folder into the layout the pipeline reads.
#
# Analysis walks the source folder and reports what is there: images (count,
# formats, folders, filename patterns), tables (columns, a few rows, whether they
# look like a 0/1/2 genotype matrix, a trait table or a file list), VCF and HapMap
# files. It then scores the ways image names could map to genotype IDs by how
# many images each would match.
#
# The decision says which file is which and how an image's line is found. Apply
# writes image_index.csv, genotypes_012.csv.gz + snp_map.csv, traits.csv and the
# train/test split. Images are not copied: the index points at the originals,
# and the image stage writes the training copies.
import gzip
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..imaging import IMAGE_EXTENSIONS
from ..project import write_json
from .base import NOTES, Stage, array, boolean, number, obj, string

# Covariates: columns of a per-line table that describe how or when a line was imaged
# rather than its genotype. Batch effects hide in them (the mouse X-rays changed with
# imaging month), so they are added to the genotype table as one-hot columns: the
# model can then separate them from the genotype, and the results can show how much
# accuracy they explain. Columns that are IDs (colony, litter) are never covariates.
COVARIATE_HINT = re.compile(r'sex|gender|batch|date|plate|site|year|season|rep(licate)?$|block|experiment|'
                            r'operator|location|env|trial|cohort|tray|camera|scanner|run|machine|setup', re.I)
IMAGING_DATE_HINT = re.compile(r'experiment|imag|scan|acqui|photo|captur|measur|phenotyp|assay|session', re.I)
ID_LIKE = re.compile(r'(^|[_\- ])(id|ids|key|uuid|hash)$|^id[_\- ]', re.I)
NUMERIC_COVARIATE = re.compile(r'^(age|dose|days?|weeks?|batch|plate|rep|block|year)([_\- ]|$)|[_\- ](age|dose)$', re.I)
MAX_LEVELS = 200
MIN_GROUP_FOR_TEST = 4          # repeated-genotype designs: smaller groups stay wholly in training
TABLE_EXTENSIONS = {'.csv', '.tsv', '.txt', '.parquet'}
SPLIT_RE = re.compile(r'[_\-. ]+')
ID_HINTS = ('line', 'genotype', 'taxa', 'taxon', 'sample', 'id', 'name', 'accession', 'strain', 'mouse')


def _stem(s):
    return Path(s).stem if isinstance(s, str) else ''


def _fname(s):
    return Path(s).name if isinstance(s, str) else ''


def _rel(path, base):
    return str(Path(path).relative_to(base)).replace('\\', '/')


def _read_table(path, nrows=None):
    path = Path(path)
    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
        return df.head(nrows) if nrows else df
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', errors='replace') as f:
        head = f.readline()
    sep = '\t' if head.count('\t') > head.count(',') else ','
    return pd.read_csv(path, sep=sep, nrows=nrows, low_memory=False)


def _is_text(col):
    return pd.api.types.is_string_dtype(col) or pd.api.types.is_object_dtype(col)


def _looks_like_genotypes(df):
    """Share of numeric cells in 0/1/2 (or -1/0/1) beyond the first column."""
    num = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    if num.shape[1] < 20:
        return 0.0, None
    vals = num.to_numpy().ravel()
    vals = vals[~np.isnan(vals)]
    if not len(vals):
        return 0.0, None
    s012 = np.isin(vals, (0, 1, 2)).mean()
    sm1 = np.isin(vals, (-1, 0, 1)).mean()
    if s012 >= sm1:
        return float(s012), 'matrix_012'
    return float(sm1), 'matrix_minus1_1'


def _id_column(df):
    for c in df.columns:
        if any(h in str(c).lower() for h in ID_HINTS) and _is_text(df[c]):
            return str(c)
    for c in df.columns:
        if _is_text(df[c]):
            return str(c)
    return str(df.columns[0])


class Intake(Stage):
    name = 'intake'
    effort = 'high'      # the one genuinely open-ended judgement in the pipeline

    def needs_agent(self, ds, report, default):
        m = [x for x in report['image_name_mappings'] if x.get('to_genotypes', True)]
        if not m or not default['genotypes']['path']:
            return True, 'no image-to-line mapping reaches the genotype IDs, or no genotype file'
        if m[0]['share'] < 0.95:
            return True, f"best mapping matches only {m[0]['share']:.0%} of images"
        # A rival reaches a different ID list; two patterns giving the same IDs are one mapping.
        rivals = [x for x in m[1:] if x['ids_from'] != m[0]['ids_from'] and x['share'] > 0.9 * m[0]['share']]
        if rivals:
            return True, 'more than one naming pattern matches most images'
        dates = [c for c in report.get('covariate_candidates', []) if c['kind'] == 'date' and not c['in_genotypes']]
        if dates and not any(c['kind'] == 'date_month' for c in default.get('covariates', [])):
            return True, 'the metadata has dates, but none is clearly the imaging date'
        return False, f"{m[0]['share']:.0%} of images map to genotyped lines ({m[0]['mapping']})"

    def digest(self, ds, report):
        return {
            'n_images': report['n_images'], 'image_extensions': report['image_extensions'],
            'top_folders': dict(list(report['image_folders'].items())[:5]),
            'image_examples': report['image_examples'][:8],
            'tables': [{k: t.get(k) for k in ('path', 'n_columns', 'id_column_guess', 'genotype_like',
                                             'genotype_format_guess', 'error')}
                       | {'columns': t.get('columns', [])[:12]} for t in report['tables'][:10]],
            'vcf_files': report['vcf_files'][:3], 'hapmap_files': report['hapmap_files'][:3],
            'image_name_mappings': report['image_name_mappings'][:5],
            'covariate_candidates': report.get('covariate_candidates', [])[:15],
        }

    def analyze(self, ds):
        cfg = ds.config()
        src = Path(cfg.get('source', ''))
        if not src.exists():
            raise SystemExit(f'source folder not found: {src}')
        images, tables, vcfs, hapmaps, other = [], [], [], [], Counter()
        for p in src.rglob('*'):
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            name = p.name.lower()
            if suf in IMAGE_EXTENSIONS:
                images.append(p)
            elif name.endswith(('.vcf', '.vcf.gz')):
                vcfs.append(p)
            elif name.endswith(('.hmp.txt', '.hmp.txt.gz')):
                hapmaps.append(p)
            elif suf in TABLE_EXTENSIONS or name.endswith(('.csv.gz', '.tsv.gz')):
                tables.append(p)
            else:
                other[suf or p.name] += 1
        if not images:
            raise SystemExit(f'no images (png, jpg, tif, bmp) found under {src}')

        folders = Counter(_rel(p.parent, src) for p in images)
        exts = Counter(p.suffix.lower() for p in images)
        stems = [p.stem for p in images]

        table_reports = []
        for t in sorted(tables, key=lambda p: -p.stat().st_size)[:25]:
            try:
                head = _read_table(t, nrows=60)
            except Exception as e:  # noqa: BLE001
                table_reports.append({'path': _rel(t, src), 'error': str(e)})
                continue
            geno_share, geno_fmt = _looks_like_genotypes(head)
            idc = _id_column(head)
            numeric = [str(c) for c in head.columns if pd.api.types.is_numeric_dtype(head[c])]
            table_reports.append({
                'path': _rel(t, src), 'size_mb': round(t.stat().st_size / 1e6, 2),
                'n_columns': int(head.shape[1]), 'columns': [str(c) for c in head.columns[:40]],
                'id_column_guess': idc, 'numeric_columns': numeric[:40],
                'genotype_like': round(geno_share, 3), 'genotype_format_guess': geno_fmt,
                'head': head.iloc[:4, :12].astype(str).values.tolist(),
            })

        # Every ID a table offers, to test image-name mappings against.
        id_sets, frames = {}, {}
        for tr in table_reports:
            if 'error' in tr:
                continue
            try:
                df = _read_table(src / tr['path'])
            except Exception:  # noqa: BLE001
                continue
            frames[tr['path']] = df
            if tr['genotype_like'] > 0.9 and tr['n_columns'] > 50:
                id_sets[tr['path'] + '::rows'] = set(df.iloc[:, 0].astype(str))
                id_sets[tr['path'] + '::columns'] = set(map(str, df.columns[1:]))
            else:
                for c in df.columns:
                    if _is_text(df[c]):
                        id_sets[f"{tr['path']}::{c}"] = set(df[c].astype(str))
        for v in vcfs[:3]:
            id_sets[_rel(v, src) + '::samples'] = set(_vcf_samples(v))

        mappings = []
        tokens = [SPLIT_RE.split(s) for s in stems]
        candidates = {'whole stem': ('^(?P<line>.+)$', stems)}
        for k in range(3):
            vals = [t[k] if len(t) > k else '' for t in tokens]
            pattern = '^' + ''.join(r'[^_\-. ]+[_\-. ]+' for _ in range(k)) + r'(?P<line>[^_\-. ]+)'
            candidates[f'token {k}'] = (pattern, vals)
        for label, (pattern, vals) in candidates.items():
            for key, ids in id_sets.items():
                hit = sum(v in ids for v in vals)
                if hit:
                    mappings.append({'mapping': label, 'regex': pattern, 'ids_from': key,
                                     'images_matched': hit, 'share': round(hit / len(vals), 3)})
        # The genotype IDs. A mapping only helps if it ends there: image names that match
        # an image table's file column still need that table's line column.
        geno_keys = [k for k in id_sets if k.endswith(('::rows', '::columns', '::samples'))]
        geno_ids = set().union(*(id_sets[k] for k in geno_keys)) if geno_keys else set()
        for m in mappings:
            m['to_genotypes'] = m['ids_from'] in geno_keys
        # Two hops: a table with one column matching the image names and another matching
        # the genotype IDs (an image index keyed by file, giving each file's line).
        stem_set = set(stems)
        direct = {m['ids_from'] for m in mappings if m['to_genotypes'] and m['share'] >= 0.95}
        if direct:      # the image names already are line IDs; no need for a lookup table
            frames_for_lookup = {}
        else:
            frames_for_lookup = frames
        for path, df in frames_for_lookup.items():
            text = [c for c in df.columns if _is_text(df[c])]
            fcols = [c for c in text if df[c].astype(str).map(_stem).isin(stem_set).any()]
            for fc in fcols:
                name_of = df[fc].astype(str).map(_stem)
                for lc in text:
                    if lc == fc:
                        continue
                    vals = df[lc].astype(str)
                    if not geno_ids or vals.isin(geno_ids).mean() < 0.5:
                        continue
                    lookup = dict(zip(name_of, vals))
                    hit = sum(lookup.get(s) in geno_ids for s in stems)
                    if hit:
                        key = max(geno_keys, key=lambda k: vals.isin(id_sets[k]).mean())
                        mappings.append({'mapping': 'table', 'regex': '', 'ids_from': key, 'via': f'{path}::{lc}',
                                         'table_path': path, 'filename_column': str(fc), 'line_column': str(lc),
                                         'to_genotypes': True, 'images_matched': hit,
                                         'share': round(hit / len(stems), 3)})
        mappings.sort(key=lambda m: (not m['to_genotypes'], -m['images_matched']))
        # The line IDs: the genotype ID list the best mapping reaches (rows or columns of
        # the genotype table, whichever holds the lines). A trait table, covariate table or
        # file list shares its IDs with them.
        line_ids = id_sets.get(mappings[0]['ids_from'], set()) if mappings else set()
        for tr in table_reports:
            ids = id_sets.get(f"{tr.get('path')}::{tr.get('id_column_guess')}", set())
            tr['id_overlap_with_lines'] = round(len(ids & line_ids) / len(ids), 3) if ids else 0.0
        geno_cols = set()
        for tr in table_reports:
            if tr.get('genotype_like', 0) > 0.9 and tr['path'] in frames:
                geno_cols |= {str(c).upper() for c in frames[tr['path']].columns}
        covariates = _covariate_candidates(frames, line_ids, geno_cols)

        return {
            'source': str(src),
            'n_images': len(images),
            'image_extensions': dict(exts),
            'image_folders': dict(folders.most_common(20)),
            'image_examples': [_rel(p, src) for p in images[:: max(1, len(images) // 25)][:25]],
            'tables': table_reports,
            'vcf_files': [_rel(v, src) for v in vcfs],
            'hapmap_files': [_rel(h, src) for h in hapmaps],
            'other_files': dict(other.most_common(15)),
            'image_name_mappings': mappings[:15],
            'covariate_candidates': covariates,
        }

    def schema(self, ds, report):
        return obj({
            'image_extensions': array(string(), 'Lower-case extensions to use, e.g. [".png"].'),
            'image_subfolder': string('Folder under the source holding the images; "" for all of it.'),
            'exclude_patterns': array(string(), 'Substrings of relative paths to skip (masks, thumbnails).'),
            'line_id': obj({
                'method': string(enum=['regex', 'table']),
                'regex': string('Python regex with a named group (?P<line>...), matched against the '
                                'image file name without extension.'),
                'table_path': string('For method "table": the table listing image files; "" otherwise.'),
                'table_filename_column': string(), 'table_line_column': string(),
            }),
            'genotypes': obj({
                'path': string('Relative path of the genotype file.'),
                'format': string(enum=['matrix_012', 'matrix_minus1_1', 'vcf', 'hapmap']),
                'orientation': string(enum=['lines_as_rows', 'markers_as_rows']),
                'id_column': string('Column holding line IDs (lines_as_rows) or marker IDs.'),
                'marker_map_path': string('"" if there is none or it is inside the VCF/HapMap.'),
                'marker_map_snp_column': string(), 'chr_column': string(), 'pos_column': string(),
            }),
            'traits': obj({
                'path': string('"" if the dataset has no trait table.'),
                'id_column': string(), 'level': string(enum=['none', 'line', 'image']),
                'columns': array(string(), 'The trait columns (measurements), not IDs or covariates.'),
            }),
            'split': obj({
                'method': string(enum=['random', 'table']),
                'test_fraction': number(),
                'table_path': string(), 'line_column': string(), 'split_column': string(),
                'test_value': string('Value in split_column that marks a test line.'),
                'ensure_coverage': boolean('When many lines share a genotype (knockouts, clones, '
                                           'repeated crosses): move test lines whose genotype never occurs '
                                           'in training into training, since nothing could predict them.'),
            }),
            'covariates': array(obj({
                'table_path': string(), 'id_column': string('Column holding line IDs.'),
                'column': string(), 'kind': string(enum=['date_month', 'category']),
            }), 'How or when lines were imaged (imaging date, sex, plate, site). Added to the genotype '
                'table as one-hot columns so batch effects are not mistaken for genetics. Never IDs, '
                'traits or the genotype itself.'),
            'pixel_size': obj({
                'um_per_px': number('Physical size of one source pixel in micrometres, 0 if unknown.'),
                'table_column': string('Image-table column with per-image pixel size, "" if none.'),
            }),
            'notes': NOTES,
        })

    def default_decision(self, ds, report):
        exts = sorted(report['image_extensions'], key=lambda e: -report['image_extensions'][e])
        maps_ = report['image_name_mappings']
        best = next((m for m in maps_ if m.get('to_genotypes', True)), maps_[0] if maps_ else None)
        tables = [t for t in report['tables'] if 'error' not in t]
        geno = max(tables, key=lambda t: (t['genotype_like'] > 0.9, t['n_columns']), default=None)
        if report['vcf_files']:
            g = {'path': report['vcf_files'][0], 'format': 'vcf', 'orientation': 'markers_as_rows',
                 'id_column': 'ID'}
        elif report['hapmap_files']:
            g = {'path': report['hapmap_files'][0], 'format': 'hapmap', 'orientation': 'markers_as_rows',
                 'id_column': 'rs#'}
        elif geno and geno['genotype_like'] > 0.9:
            rows_match = best and best['ids_from'] == geno['path'] + '::columns'
            g = {'path': geno['path'], 'format': geno['genotype_format_guess'] or 'matrix_012',
                 'orientation': 'markers_as_rows' if rows_match else 'lines_as_rows',
                 'id_column': geno['id_column_guess']}
        else:
            g = {'path': '', 'format': 'matrix_012', 'orientation': 'lines_as_rows', 'id_column': ''}
        g.update(marker_map_path='', marker_map_snp_column='', chr_column='', pos_column='')
        maps = [t for t in tables if {'chr', 'pos'} <= {c.lower() for c in t['columns']}
                or {'chrom', 'pos'} <= {c.lower() for c in t['columns']}]
        if maps and g['format'].startswith('matrix'):
            cols = {c.lower(): c for c in maps[0]['columns']}
            g.update(marker_map_path=maps[0]['path'],
                     marker_map_snp_column=cols.get('snp', cols.get('marker', maps[0]['columns'][0])),
                     chr_column=cols.get('chr', cols.get('chrom', '')), pos_column=cols.get('pos', ''))
        # A trait table: mostly numeric, not the genotype table, IDs that match lines.
        trait = {'path': '', 'id_column': '', 'level': 'none', 'columns': []}
        for t in tables:
            if t is geno or t['genotype_like'] > 0.9 or t in maps or len(t['numeric_columns']) < 1:
                continue
            idc = t['id_column_guess']
            if t.get('id_overlap_with_lines', 0) > 0.5 or (best and best['ids_from'] == f"{t['path']}::{idc}"):
                trait = {'path': t['path'], 'id_column': idc, 'level': 'line',
                         'columns': [c for c in t['numeric_columns'] if c != idc and not ID_LIKE.search(c)
                                     and not NUMERIC_COVARIATE.search(c)][:30]}
                break
        split = {'method': 'random', 'test_fraction': 0.15, 'table_path': '', 'line_column': '',
                 'split_column': '', 'test_value': 'test', 'ensure_coverage': True}
        for t in tables:
            for c in t['columns']:
                if 'split' in c.lower():
                    split = {'method': 'table', 'test_fraction': 0.15, 'table_path': t['path'],
                             'line_column': t['id_column_guess'], 'split_column': c, 'test_value': 'test',
                             'ensure_coverage': True}
        if best and best['mapping'] == 'table':
            line_id = {'method': 'table', 'regex': '', 'table_path': best['table_path'],
                       'table_filename_column': best['filename_column'], 'table_line_column': best['line_column']}
        else:
            line_id = {'method': 'regex', 'regex': best['regex'] if best else '^(?P<line>.+)$',
                       'table_path': '', 'table_filename_column': '', 'table_line_column': ''}
        # Covariates: every hinted category, and the imaging date (by month) when one date
        # column clearly is it. Several undistinguished dates are left to the agent.
        cands = [c for c in report.get('covariate_candidates', []) if not c['in_genotypes']]
        dates = [c for c in cands if c['kind'] == 'date']
        img_dates = [c for c in dates if IMAGING_DATE_HINT.search(c['column'])]
        chosen = [c for c in cands if c['kind'] == 'category' and c['hinted']]
        if len({c['column'] for c in img_dates}) == 1 or len({c['column'] for c in dates}) == 1:
            chosen.append((img_dates or dates)[0])
        seen = set()
        chosen = [c for c in chosen if not (c['column'] in seen or seen.add(c['column']))]
        covariates = [{'table_path': c['table_path'], 'id_column': c['id_column'], 'column': c['column'],
                       'kind': 'date_month' if c['kind'] == 'date' else 'category'} for c in chosen]
        return {
            'image_extensions': exts,
            'image_subfolder': '',
            'exclude_patterns': ['mask', 'thumb'],
            'line_id': line_id,
            'genotypes': g, 'traits': trait, 'split': split, 'covariates': covariates,
            'pixel_size': {'um_per_px': 0.0, 'table_column': ''},
            'notes': (f"Rule-based guess: line ID from the image name ({best['mapping']}, matches "
                      f"{best['share']:.0%} of images against {best['ids_from']})." if best else
                      'Rule-based guess: no image-name pattern matched any ID list; check the line mapping.'),
        }

    def prompt(self, ds, report):
        return ('Work out how this dataset is organised: which images to use, how each image '
                'maps to its line (genotype), which file holds the genotypes and in what format, '
                'whether there is a trait table, and how to split lines into train and test. '
                'Prefer the mapping that matches the most images and ends at the genotype IDs. '
                'Masks, thumbnails and calibration images are not training images. Choose the '
                'covariates: columns that say how or when a line was imaged (imaging date, sex, '
                'plate, site, camera), not IDs, traits or the genotype; when several dates exist, '
                'pick the one the images were taken on.')

    def apply(self, ds, d):
        src = Path(ds.config()['source'])
        base = src / d['image_subfolder'] if d['image_subfolder'] else src
        exts = {e.lower() if e.startswith('.') else '.' + e.lower() for e in d['image_extensions']}
        paths = sorted(p for p in base.rglob('*') if p.is_file() and p.suffix.lower() in exts
                       and not any(x.lower() in _rel(p, src).lower() for x in d['exclude_patterns'] if x))

        # Genotypes -> lines x markers, 0/1/2, -1 missing.
        lines, markers, dosage, snp_map = _load_genotypes(src, d['genotypes'])
        ds.genetic_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(dosage, index=pd.Index(lines, name='line'), columns=markers)
        groups = _genotype_groups(frame)         # before covariates: lines with identical genotypes
        cov_raw, cov_cols = _covariates(src, d.get('covariates', []), lines)
        if len(cov_cols.columns):
            frame = pd.concat([frame, cov_cols], axis=1)
            snp_map = pd.concat([snp_map, pd.DataFrame({'snp': cov_cols.columns, 'chr': 'covariate',
                                                        'pos': np.arange(len(cov_cols.columns))})],
                                ignore_index=True)
        ds.metadata_dir.mkdir(parents=True, exist_ok=True)
        if len(cov_raw.columns):
            cov_raw.to_csv(ds.metadata_dir / 'line_covariates.csv', index_label='line')
        elif (ds.metadata_dir / 'line_covariates.csv').exists():
            (ds.metadata_dir / 'line_covariates.csv').unlink()
        frame.to_csv(ds.genotypes_path, compression='gzip')
        snp_map.to_csv(ds.snp_map_path, index=False)
        line_set = set(lines)

        # Image -> line.
        if d['line_id']['method'] == 'table':
            t = _read_table(src / d['line_id']['table_path'])
            name_to_line = dict(zip(t[d['line_id']['table_filename_column']].astype(str).map(_fname),
                                    t[d['line_id']['table_line_column']].astype(str)))
            line_of = lambda p: name_to_line.get(p.name, name_to_line.get(p.stem))  # noqa: E731
        else:
            rx = re.compile(d['line_id']['regex'])
            def line_of(p):
                m = rx.search(p.stem)
                return m.group('line') if m else None
        px_col = d['pixel_size']['table_column']
        px_map = {}
        if px_col and d['line_id']['method'] == 'table':
            px_map = dict(zip(t[d['line_id']['table_filename_column']].astype(str).map(_fname),
                              pd.to_numeric(t[px_col], errors='coerce')))
        rows, unmatched = [], []
        for i, p in enumerate(paths):
            line = line_of(p)
            if line is None or line not in line_set:
                unmatched.append(_rel(p, src))
                continue
            rows.append({'image_id': f'{_slug(line)}_{i:06d}', 'line': line, 'relpath': _rel(p, src),
                         'source_path': str(p),
                         'pixel_size_um': px_map.get(p.name, d['pixel_size']['um_per_px'] or np.nan)})
        if not rows:
            raise SystemExit(f'no image matched a genotyped line ({len(unmatched)} unmatched); '
                             'check the line_id mapping and the genotype ID column')
        index = pd.DataFrame(rows)
        ds.metadata_dir.mkdir(parents=True, exist_ok=True)
        index.to_csv(ds.image_index_path, index=False)
        pd.Series(unmatched, name='relpath').to_csv(ds.metadata_dir / 'unmatched_images.csv', index=False)

        # Traits (line means or per image).
        n_traits = 0
        if d['traits']['level'] != 'none' and d['traits']['path']:
            t = _read_table(src / d['traits']['path'])
            cols = [c for c in d['traits']['columns'] if c in t.columns]
            out = t[[d['traits']['id_column']] + cols].copy()
            out[cols] = out[cols].apply(pd.to_numeric, errors='coerce')
            if d['traits']['level'] == 'image':
                key = out[d['traits']['id_column']].astype(str).map(_stem)
                by_stem = dict(zip(index.relpath.map(lambda r: Path(r).stem), index.image_id))
                out.insert(0, 'image_id', key.map(by_stem))
                out = out.dropna(subset=['image_id']).merge(index[['image_id', 'line']], on='image_id')
            else:
                out = out.rename(columns={d['traits']['id_column']: 'line'})
                out['line'] = out['line'].astype(str)
            out.to_csv(ds.traits_path, index=False)
            n_traits = len(cols)

        # Split, over lines that have both images and genotypes.
        imaged = sorted(index.line.unique())
        grp = groups.reindex(imaged)
        sizes = grp.map(grp.value_counts())
        repeated = bool((sizes >= 2).mean() > 0.5)       # most lines share their genotype with others
        sp = d['split']
        if sp['method'] == 'table' and sp['table_path']:
            t = _read_table(src / sp['table_path'])
            test = set(t.loc[t[sp['split_column']].astype(str) == sp['test_value'],
                             sp['line_column']].astype(str))
        elif repeated:
            # Stratified: each genotype group gives test_fraction of its lines; groups of
            # fewer than MIN_GROUP_FOR_TEST lines stay in training, so every test genotype
            # also occurs in training.
            rng = np.random.default_rng(0)
            test = set()
            for _, members in grp.groupby(grp):
                m = list(members.index)
                if len(m) >= MIN_GROUP_FOR_TEST:
                    rng.shuffle(m)
                    test |= set(m[:max(1, int(round(sp['test_fraction'] * len(m))))])
        else:
            rng = np.random.default_rng(0)
            shuffled = list(imaged)
            rng.shuffle(shuffled)
            test = set(shuffled[:int(round(sp['test_fraction'] * len(shuffled)))])
        test &= set(imaged)
        # Coverage: with repeated genotypes, a test line whose genotype never occurs in
        # training cannot be predicted from anything the model saw.
        uncovered = set()
        if repeated:
            train_groups = set(grp[[l for l in imaged if l not in test]])
            uncovered = {l for l in test if grp[l] not in train_groups}
            if uncovered and sp.get('ensure_coverage', True):
                test -= uncovered
        pd.DataFrame({'line': imaged, 'split_random': ['test' if l in test else 'train' for l in imaged]}
                     ).to_csv(ds.split_path, index=False)
        coverage = {'design': 'repeated genotypes' if repeated else 'one line per genotype',
                    'genotype_groups': int(grp.nunique()), 'test_lines': len(test),
                    'test_lines_without_training_genotype': len(uncovered),
                    'moved_to_training': len(uncovered) if sp.get('ensure_coverage', True) else 0}
        if uncovered:
            ds.log('intake', f"{len(uncovered)} test lines had a genotype never seen in training"
                   + (' and were moved to training' if sp.get('ensure_coverage', True) else ''), level='warning')

        ds.update_config(pipeline={'splits': ['split_random'], 'has_traits': n_traits > 0,
                                   'pixel_size_um': d['pixel_size']['um_per_px'] or None})
        write_json(ds.stage_dir('intake') / 'summary.json', {
            'images': len(index), 'lines_with_images': len(imaged), 'genotyped_lines': len(lines),
            'markers': len(markers), 'covariate_columns': list(cov_cols.columns)[:50],
            'n_covariate_columns': int(len(cov_cols.columns)), 'unmatched_images': len(unmatched),
            'traits': n_traits, 'split_coverage': coverage})
        return {'summary': f'{len(index)} images from {len(imaged)} lines, {len(markers)} markers'
                           + (f' + {len(cov_cols.columns)} covariate columns ({", ".join(cov_raw.columns)})'
                              if len(cov_cols.columns) else '')
                           + f', {n_traits} traits; {len(unmatched)} images unmatched'
                           + (f'; {len(uncovered)} test lines with an unseen genotype moved to training'
                              if coverage['moved_to_training'] else ''),
                'images': len(index), 'lines': len(imaged), 'markers': len(markers),
                'covariate_columns': int(len(cov_cols.columns)), 'unmatched_images': len(unmatched),
                'traits': n_traits, 'split_coverage': coverage}


def _parse_dates(s):
    s = s.dropna().astype(str)
    if not len(s) or s.str.fullmatch(r'-?\d+(\.\d+)?').mean() > 0.5:    # plain numbers are not dates
        return None
    d = pd.to_datetime(s, errors='coerce', utc=True, format='mixed')
    return d if d.notna().mean() > 0.9 else None


def _covariate_candidates(frames, line_ids, geno_cols):
    """Columns of per-line tables that could be covariates: a few levels each, not IDs,
    not numbers (except dates), keyed by a column of genotyped line IDs."""
    out = []
    for path, df in frames.items():
        idc = next((c for c in df.columns if _is_text(df[c]) and df[c].astype(str).isin(line_ids).mean() > 0.5),
                   None)
        if idc is None:
            continue
        for c in df.columns:
            if c == idc or ID_LIKE.search(str(c)) or 'split' in str(c).lower():
                continue
            col = df[c]
            dates = _parse_dates(col) if _is_text(col) else None
            if dates is not None:
                levels = dates.dt.strftime('%Y-%m').nunique()
                kind = 'date'
            elif _is_text(col):
                levels = col.nunique()
                kind = 'category'
                if not 2 <= levels <= MAX_LEVELS or levels > 0.5 * len(col.dropna()):
                    continue
            else:
                continue
            name = str(c)
            out.append({'table_path': path, 'id_column': str(idc), 'column': name, 'kind': kind,
                        'levels': int(levels), 'hinted': bool(COVARIATE_HINT.search(name)),
                        'examples': [str(v) for v in col.dropna().unique()[:3]],
                        'in_genotypes': any(g.startswith(name.upper() + '_') for g in geno_cols)
                                        or (kind == 'date' and any(g.startswith('BATCH_') for g in geno_cols))})
    out.sort(key=lambda c: (not c['hinted'], c['kind'] != 'date'))
    return out


def _covariates(src, specs, lines):
    """One value per line for each covariate (dates as YYYY-MM), and their one-hot columns:
    a two-level category becomes one column, anything else one column per level."""
    raw = pd.DataFrame(index=pd.Index(lines, name='line'))
    for s in specs:
        t = _read_table(src / s['table_path'])
        if s['column'] not in t or s['id_column'] not in t:
            continue
        v = t[[s['id_column'], s['column']]].dropna()
        v[s['id_column']] = v[s['id_column']].astype(str)
        if s['kind'] == 'date_month':
            v[s['column']] = pd.to_datetime(v[s['column']].astype(str), errors='coerce', utc=True,
                                            format='mixed').dt.strftime('%Y-%m')
        per_line = v.dropna().groupby(s['id_column'])[s['column']].agg(lambda x: x.astype(str).mode().iat[0])
        raw[_slug(s['column'])] = per_line.reindex(raw.index)
    cols = []
    for c in raw.columns:
        levels = sorted(raw[c].dropna().unique())
        if len(levels) < 2:
            continue
        keep = levels[1:] if len(levels) == 2 else levels
        prefix = 'BATCH' if c and any(h in c.lower() for h in ('date', 'month')) else c.upper()
        for lv in keep:
            cols.append(pd.Series((raw[c] == lv).astype(np.int8), name=f'{prefix}_{_slug(str(lv))}'))
    onehot = pd.concat(cols, axis=1) if cols else pd.DataFrame(index=raw.index)
    return raw, onehot


def _genotype_groups(frame):
    """Lines with identical genotype rows share a group id."""
    h = pd.util.hash_pandas_object(frame, index=False)
    return pd.Series(pd.factorize(h)[0], index=frame.index.astype(str))


def _slug(s):
    return re.sub(r'[^A-Za-z0-9]+', '-', str(s)).strip('-')[:40] or 'line'


def _vcf_samples(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', errors='replace') as f:
        for line in f:
            if line.startswith('#CHROM'):
                return line.rstrip('\n').split('\t')[9:]
    return []


def _load_genotypes(src, g):
    path = src / g['path']
    if not g['path'] or not path.exists():
        raise SystemExit(f'genotype file not found: {g["path"]!r}')
    if g['format'] == 'vcf':
        return _read_vcf(path)
    if g['format'] == 'hapmap':
        return _read_hapmap(path)
    df = _read_table(path)
    idc = g['id_column'] if g['id_column'] in df.columns else df.columns[0]
    df = df.set_index(idc)
    if g['orientation'] == 'markers_as_rows':
        df = df.T
    lines = [str(x) for x in df.index]
    markers = [str(c) for c in df.columns]
    x = df.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
    if g['format'] == 'matrix_minus1_1':
        x = x + 1
    x = np.where(np.isnan(x), -1, np.clip(np.round(x), 0, 2)).astype(np.int8)
    if g['marker_map_path']:
        m = _read_table(src / g['marker_map_path'])
        m = m.rename(columns={g['marker_map_snp_column']: 'snp', g['chr_column']: 'chr', g['pos_column']: 'pos'})
        m['snp'] = m['snp'].astype(str)
        snp_map = pd.DataFrame({'snp': markers}).merge(m[['snp', 'chr', 'pos']], on='snp', how='left')
        snp_map['chr'] = snp_map['chr'].fillna(0)
        snp_map['pos'] = snp_map['pos'].fillna(-1)
    else:
        snp_map = pd.DataFrame({'snp': markers, 'chr': 0, 'pos': np.arange(len(markers))})
    return lines, markers, x, snp_map


def _read_vcf(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    samples, rows, ids, chrs, poss = None, [], [], [], []
    with opener(path, 'rt', errors='replace') as f:
        for line in f:
            if line.startswith('##'):
                continue
            parts = line.rstrip('\n').split('\t')
            if line.startswith('#CHROM'):
                samples = parts[9:]
                continue
            fmt = parts[8].split(':')
            gi = fmt.index('GT') if 'GT' in fmt else 0
            dos = []
            for call in parts[9:]:
                gt = call.split(':')[gi].replace('|', '/')
                if '.' in gt:
                    dos.append(-1)
                else:
                    dos.append(sum(a != '0' for a in gt.split('/')))
            rows.append(dos)
            ids.append(parts[2] if parts[2] != '.' else f'{parts[0]}_{parts[1]}')
            chrs.append(parts[0])
            poss.append(int(parts[1]))
    x = np.array(rows, dtype=np.int8).T
    return samples, ids, np.clip(x, -1, 2), pd.DataFrame({'snp': ids, 'chr': chrs, 'pos': poss})


def _read_hapmap(path):
    df = pd.read_csv(path, sep='\t', low_memory=False)
    samples = list(df.columns[11:])
    calls = df.iloc[:, 11:].astype(str).to_numpy()
    x = np.full(calls.shape, -1, dtype=np.int8)
    for i, row in enumerate(calls):
        alleles = [a for c in row for a in c if a in 'ACGT']
        if not alleles:
            continue
        major = Counter(alleles).most_common(1)[0][0]
        for j, c in enumerate(row):
            c = c.strip()
            if len(c) == 1:
                c = c * 2
            if all(a in 'ACGT' for a in c) and len(c) == 2:
                x[i, j] = sum(a != major for a in c)
    return samples, list(df.iloc[:, 0].astype(str)), x.T, pd.DataFrame(
        {'snp': df.iloc[:, 0].astype(str), 'chr': df.iloc[:, 2], 'pos': df.iloc[:, 3]})
