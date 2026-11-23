# Writes the working train_widiv.py minus the matched-PCA / PCA-variance hunks,
# so the RIL dataset can be committed on its own.
import sys
src = open(sys.argv[1], encoding='utf-8', newline='').read()

def cut(old, new):
    global src
    assert src.count(old) == 1, old[:80]
    src = src.replace(old, new)

cut("""        # says why). pca_n_components fixes the number instead. WIDIV_PCA_VARIANCE
        # overrides it; 1.0 keeps every component the fitting lines support (732
        # on split_random), which on test lines decodes SNPs as well as any linear
        # encoding fitted on these lines can and predicts traits about as well as
        # GBLUP (results/widiv/snp_encoding_analysis/).
        pca_target_variance = float(os.environ.get('WIDIV_PCA_VARIANCE', DATASET_DEFAULTS['pca_variance']))
        pca_n_components = None
        # Shrink the training lines' PC scores to the spread unseen lines have
        # (widiv_snp_encoding.heldout_score_scale), so validation and test lines
        # are not pulled towards the centre of what the model trained on.
        # WIDIV_PCA_MATCH_HELDOUT=1 turns it on.
        pca_match_heldout = os.environ.get('WIDIV_PCA_MATCH_HELDOUT', '0') == '1'
""", """        # says why). pca_n_components fixes the number instead.
        pca_target_variance = DATASET_DEFAULTS['pca_variance']
        pca_n_components = None
""")
cut("""    if cfg.pca_match_heldout and (windows or cfg.orthogonalize_qtl):
        raise SystemExit("pca_match_heldout is only implemented for snp_encoding='pca' "
                         "without orthogonalize_qtl")
""", "")
cut("""                                + (f'_pca{round(cfg.pca_target_variance * 100)}'
                                   if not windows and cfg.pca_n_components is None
                                   and cfg.pca_target_variance != DATASET_DEFAULTS['pca_variance'] else '')
                                + ('_matched' if cfg.pca_match_heldout else '')
""", "")
cut("""        saved_proj = resume_ckpt['snp_projector']
        if not windows and bool(saved_proj.get('match_heldout', False)) != cfg.pca_match_heldout:
            raise SystemExit(f"{resume_path.name} was trained with pca_match_heldout="
                             f"{bool(saved_proj.get('match_heldout', False))}, not "
                             f"{cfg.pca_match_heldout}. Give this run its own run_name.")
""", "")
cut("""                                           target_variance=cfg.pca_target_variance,
                                           match_heldout=cfg.pca_match_heldout)
            projector.fit(dosage[fit_rows], snp_names, exclude_columns=exclude,
                          orthogonalize_columns=orthogonalize,
                          line_names=[g for g in sorted(train_lines) if g in row_of])
""", """                                           target_variance=cfg.pca_target_variance)
            projector.fit(dosage[fit_rows], snp_names, exclude_columns=exclude,
                          orthogonalize_columns=orthogonalize)
""")
cut("""    # kept as int8 - 90,790 values per line - and cast inside the encoder. With
    # pca_match_heldout the training lines' scores are shrunk (project_lines).
    projected = projector.project_lines(dosage, line_names)
""", """    # kept as int8 - 90,790 values per line - and cast inside the encoder.
    projected = projector.transform(dosage)
""")
open(sys.argv[2], 'w', encoding='utf-8', newline='').write(src)
