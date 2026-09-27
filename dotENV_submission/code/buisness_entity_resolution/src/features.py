"""
Person C — Similarity / Feature Engineering (~13GB RAM, e.g. Colab free tier)

Changes from the 4GB-constrained version:
  - Per-entity derived features (interned token tuples, name length, first
    token) are computed ONCE per entity in the lookup table, not once per
    candidate pair. Since blocking typically produces many candidate pairs
    per entity, this eliminates a large amount of redundant str.split() work.
    Tokens are interned (sys.intern) and stored as tuples rather than
    frozensets, and the actual set() used for jaccard is built transiently
    per pair rather than stored permanently for millions of entities — at
    4M+ referenced entities, persistent frozensets are enough to exhaust a
    ~13GB instance on their own.
  - TF-IDF vectors are computed ONCE for the whole (filtered) lookup table
    and reused via array indexing, instead of re-running the vectorizer on
    the same entity's name string every time it appears in another pair.
  - name_length_diff and name_first_token_match are now fully vectorized
    (no Python loop) since the underlying per-entity values are precomputed.
  - SequenceMatcher-based ratios use rapidfuzz when available (falls back to
    difflib automatically), which is substantially faster at this row count.
  - Larger default chunk sizes, since memory is no longer the binding
    constraint.

Everything else (S3 caching, needed-entity-id filtering, ground truth
merge-based labeling, incremental output writing) is unchanged in spirit
from the memory-constrained version, since those were good practices
independent of available RAM.

Usage (train — with labels):
  python features.py \
      --s1 normalized_source1.tsv --s2 normalized_source2.tsv --s3 normalized_source3.tsv \
      --candidates candidate_pairs_long.tsv --out features_train.tsv \
      --ground-truth train_ground_truth.tsv --local-cache-dir /home/ec2-user/SageMaker/er_cache

Usage (test — no labels):
  python features.py \
      --s1 test_normalized_source1.tsv --s2 ... --s3 ... \
      --candidates candidate_pairs_long_test.tsv --out features_test.tsv \
      --local-cache-dir /home/ec2-user/SageMaker/er_cache
"""

import argparse
import gc
import os
import sys
import time
from difflib import SequenceMatcher
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

LOOKUP_COLS = ['entity_id', 'business_name_clean', 'business_address_clean', 'country_clean']

# rapidfuzz is much faster than difflib.SequenceMatcher at this volume.
# Fall back gracefully if it isn't installed so the script still runs.
try:
    from rapidfuzz.fuzz import ratio as _rf_ratio
    _HAVE_RAPIDFUZZ = True
except ImportError:
    _HAVE_RAPIDFUZZ = False


def fuzzy_ratio(a, b):
    """Returns a 0-1 similarity ratio, equivalent in spirit to
    difflib.SequenceMatcher(None, a, b).ratio(), but via rapidfuzz (C-level)
    when available."""
    if not a and not b:
        return 1.0
    if _HAVE_RAPIDFUZZ:
        return _rf_ratio(a, b) / 100.0
    return SequenceMatcher(None, a, b).quick_ratio()


def log_memory(label=""):
    try:
        import psutil
        rss_gb = psutil.Process().memory_info().rss / 1e9
        total_gb = psutil.virtual_memory().total / 1e9
        print(f"    [memory] {label}: {rss_gb:.2f} GB used / {total_gb:.2f} GB total", flush=True)
    except ImportError:
        pass


# ---------------------------------------------------------------------
# S3 helpers (same pattern as blocking.py)
# ---------------------------------------------------------------------
def _download_from_s3(s3_uri, local_dir):
    import boto3
    parsed = urlparse(s3_uri)
    bucket, key = parsed.netloc, parsed.path.lstrip('/')
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, os.path.basename(key))
    if os.path.exists(local_path):
        print(f"    (cache hit, skipping download) {local_path}")
        return local_path
    t0 = time.time()
    s3 = boto3.client('s3')
    head = s3.head_object(Bucket=bucket, Key=key)
    size_mb = head['ContentLength'] / (1024 * 1024)
    print(f"    downloading s3://{bucket}/{key} ({size_mb:.1f} MB)...")
    s3.download_file(bucket, key, local_path)
    print(f"    downloaded in {time.time() - t0:.1f}s -> {local_path}")
    return local_path


def resolve_local_path(path, cache_dir):
    if cache_dir and path.startswith('s3://'):
        return _download_from_s3(path, cache_dir)
    return path


# ---------------------------------------------------------------------
# Jaccard over PRECOMPUTED, INTERNED token tuples.
#
# We deliberately do NOT store a frozenset per entity permanently — with
# millions of entities, millions of persistent hash-table objects is what
# was blowing past Colab's ~13GB ceiling. Instead each entity stores a
# lightweight tuple of interned strings (built once, in build_entity_lookup),
# and we build a real set() only here, transiently, for the two rows being
# compared right now. Interning means repeated words (LLC, Street, Road,
# country names, etc.) share ONE string object across all 4M+ entities
# instead of being re-allocated per entity, which is where most of the
# memory actually went.
# ---------------------------------------------------------------------
def set_jaccard(tokens_a, tokens_b):
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    a, b = set(tokens_a), set(tokens_b)
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------
# Find which entity IDs are actually referenced by the candidate pairs file
# BEFORE loading any source data — this is usually a small fraction of the
# full entity pool, so filtering here avoids building a lookup (and a TF-IDF
# matrix) far larger than what this run actually needs.
# ---------------------------------------------------------------------
def find_needed_entity_ids(candidates_path, chunksize=500_000):
    needed = set()
    for chunk in pd.read_csv(candidates_path, sep='\t', dtype=str,
                              usecols=['source1_entity_id', 'candidate_entity_id'], chunksize=chunksize):
        needed.update(chunk['source1_entity_id'].dropna())
        needed.update(chunk['candidate_entity_id'].dropna())
    return needed


def build_entity_lookup(paths, needed_ids, chunksize=500_000):
    frames = []
    for path in paths:
        t0 = time.time()
        n_kept = 0
        for chunk in pd.read_csv(path, sep='\t', dtype=str, usecols=LOOKUP_COLS, chunksize=chunksize):
            chunk = chunk.fillna('')
            filtered = chunk[chunk['entity_id'].isin(needed_ids)]
            if len(filtered) > 0:
                frames.append(filtered)
                n_kept += len(filtered)
        print(f"    {os.path.basename(path)}: kept {n_kept} needed rows in {time.time() - t0:.1f}s", flush=True)
    lookup = pd.concat(frames, ignore_index=True).set_index('entity_id')
    del frames
    gc.collect()

    # --- Precompute per-entity derived features ONCE here, instead of once
    # per candidate pair. This is the main win at scale: an entity that
    # appears in, say, 50 candidate pairs previously had its name/address
    # split into tokens 50 separate times.
    #
    # sys.intern() is the key memory-safety change here: without it, every
    # entity's tokens allocate their own fresh string objects even when the
    # word (e.g. "LLC", "STREET", "PRIVATE", a country name) is identical to
    # one already seen in another entity. At 4M+ entities, that duplication
    # is what exhausts a ~13GB Colab instance. Interning makes repeated
    # words share one object; tuples (not frozensets) keep the per-entity
    # container itself cheap too. See set_jaccard() for how these get used.
    print("    precomputing per-entity tokens (interned) / lengths / first tokens...", flush=True)
    lookup['name_tokens'] = lookup['business_name_clean'].apply(
        lambda s: tuple(sys.intern(t) for t in s.split())
    )
    log_memory("    after name_tokens")
    lookup['address_tokens'] = lookup['business_address_clean'].apply(
        lambda s: tuple(sys.intern(t) for t in s.split())
    )
    log_memory("    after address_tokens")
    lookup['name_len'] = lookup['business_name_clean'].str.len()
    lookup['first_tok'] = lookup['business_name_clean'].apply(lambda s: s.split()[0] if s else '')
    log_memory("    after lengths/first_tok")
    gc.collect()

    return lookup


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--s1', required=True)
    p.add_argument('--s2', required=True)
    p.add_argument('--s3', required=True)
    p.add_argument('--candidates', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--ground-truth', default=None)
    p.add_argument('--local-cache-dir', default=None)
    p.add_argument('--chunksize', type=int, default=500_000, help="Candidate pairs processed per chunk")
    args = p.parse_args()

    if not _HAVE_RAPIDFUZZ:
        print("NOTE: rapidfuzz not installed — falling back to difflib (slower). "
              "Install with `pip install rapidfuzz` for a large speedup on this workload.")

    s1_path = resolve_local_path(args.s1, args.local_cache_dir)
    s2_path = resolve_local_path(args.s2, args.local_cache_dir)
    s3_path = resolve_local_path(args.s3, args.local_cache_dir)
    cand_path = resolve_local_path(args.candidates, args.local_cache_dir)
    gt_path = resolve_local_path(args.ground_truth, args.local_cache_dir) if args.ground_truth else None

    print("Scanning candidate pairs for needed entity IDs...")
    t0 = time.time()
    needed_ids = find_needed_entity_ids(cand_path)
    print(f"  found {len(needed_ids)} unique entity IDs referenced in candidates ({time.time() - t0:.1f}s)")
    log_memory("after scanning needed IDs")

    print("Building entity lookup (entity_id -> name/address/country + precomputed features), filtered to needed IDs only...")
    t0 = time.time()
    lookup = build_entity_lookup([s1_path, s2_path, s3_path], needed_ids)
    print(f"  lookup built: {len(lookup)} entities in {time.time() - t0:.1f}s")
    log_memory("after building lookup")

    # Fit TF-IDF once, then ALSO transform the whole lookup once. Every
    # candidate pair then just indexes into this precomputed matrix instead
    # of re-vectorizing the same entity's name string again.
    #
    # IMPORTANT: fit() and transform() are split, and transform() runs in
    # BATCHES, rather than one fit_transform() call on the whole corpus.
    # sklearn's vectorizer builds transient Python-level (doc, n-gram) index
    # lists proportional to num_docs * avg_ngrams_per_doc BEFORE compressing
    # to a sparse matrix -- for millions of documents that transient
    # structure (not the final matrix) is what exhausts a ~13GB instance.
    # Batching bounds that transient size to one batch at a time; the final
    # concatenated matrix is the same size either way. dtype=float32 also
    # roughly halves the final matrix's storage vs the sklearn default of
    # float64.
    print("Fitting TF-IDF vectorizer (vocabulary pass, on a sample to bound memory)...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2,
                                  max_features=500_000, dtype=np.float32)
    # sklearn's fit() is internally implemented as fit_transform() with the
    # matrix discarded -- it is NOT cheaper than fit_transform on the full
    # input. Fitting on the full multi-million-row corpus reproduces the
    # same transient-memory blowup transform() had before batching, just
    # with no batching available for fit(). Character n-gram vocabularies
    # are highly repetitive, so a large random sample yields essentially the
    # same vocabulary and near-identical IDF weights while bounding fit()'s
    # internal memory to the sample size rather than the full corpus.
    fit_sample_size = min(1_000_000, len(lookup))
    fit_sample = lookup['business_name_clean'].sample(n=fit_sample_size, random_state=42)
    vectorizer.fit(fit_sample)
    del fit_sample
    gc.collect()
    print(f"  vocabulary fit on {fit_sample_size} sampled entities (of {len(lookup)} total) "
          f"in {time.time() - t0:.1f}s, {len(vectorizer.vocabulary_)} n-grams kept")
    log_memory("after fitting vectorizer vocabulary (sampled)")

    print("Transforming all needed entities into TF-IDF vectors (batched)...")
    t0 = time.time()
    transform_batch = 250_000
    names_arr = lookup['business_name_clean'].to_numpy()
    name_batches = []
    for start in range(0, len(names_arr), transform_batch):
        batch = names_arr[start:start + transform_batch]
        name_batches.append(vectorizer.transform(batch))
        if (start // transform_batch) % 4 == 0:
            log_memory(f"    after vectorizing {start + len(batch)}/{len(names_arr)} entities")
    name_vectors = sparse.vstack(name_batches, format='csr')
    del name_batches, names_arr
    gc.collect()
    entity_pos = pd.Series(np.arange(len(lookup)), index=lookup.index)
    print(f"  vectorized {name_vectors.shape[0]} entities in {time.time() - t0:.1f}s")
    log_memory("after transforming vectorizer (batched)")

    # Ground truth: build a lightweight pair-membership DataFrame for merging
    # (a merge-based join, not a giant Python set of tuples).
    gt_pairs_df = None
    if gt_path:
        gt = pd.read_csv(gt_path, sep='\t', dtype=str).fillna('')
        exploded = []
        skipped = 0
        for row in gt.itertuples(index=False):
            if not row.matched_entity_ids:
                continue
            if row.source1_entity_id not in needed_ids:
                skipped += 1
                continue
            for m in row.matched_entity_ids.split(','):
                m = m.strip()
                if m in needed_ids:
                    exploded.append((row.source1_entity_id, m))
        gt_pairs_df = pd.DataFrame(exploded, columns=['source1_entity_id', 'candidate_entity_id'])
        gt_pairs_df['label'] = 1
        del exploded, gt
        gc.collect()
        print(f"  ground truth: {len(gt_pairs_df)} true pairs kept (skipped {skipped} rows for entities not in this run's candidates)")
        log_memory("after building ground truth")

    del needed_ids
    gc.collect()

    print(f"\nProcessing candidate pairs in chunks of {args.chunksize}...")
    first_write = True
    total_rows = 0
    overall_start = time.time()

    for chunk_i, pairs_chunk in enumerate(
        pd.read_csv(cand_path, sep='\t', dtype=str, usecols=['source1_entity_id', 'candidate_entity_id'],
                    chunksize=args.chunksize)
    ):
        t0 = time.time()
        pairs_chunk = pairs_chunk.fillna('')

        merged = pairs_chunk.merge(
            lookup.add_prefix('s1_'), left_on='source1_entity_id', right_index=True, how='left'
        ).merge(
            lookup.add_prefix('cand_'), left_on='candidate_entity_id', right_index=True, how='left'
        )

        # --- TF-IDF cosine via precomputed vectors, indexed by position ---
        pos1 = pairs_chunk['source1_entity_id'].map(entity_pos)
        pos2 = pairs_chunk['candidate_entity_id'].map(entity_pos)
        valid_pos = pos1.notna() & pos2.notna()
        name_tfidf_cosine = np.zeros(len(pairs_chunk), dtype=float)
        if valid_pos.any():
            v1 = name_vectors[pos1[valid_pos].astype(int).to_numpy()]
            v2 = name_vectors[pos2[valid_pos].astype(int).to_numpy()]
            name_tfidf_cosine[valid_pos.to_numpy()] = v1.multiply(v2).sum(axis=1).A1

        # --- Fully vectorized features (no Python loop needed) ---
        merged['name_length_diff'] = (merged['s1_name_len'] - merged['cand_name_len']).abs().fillna(0).astype(int)
        s1_first = merged['s1_first_tok'].fillna('')
        cand_first = merged['cand_first_tok'].fillna('')
        merged['name_first_token_match'] = ((s1_first == cand_first) & (s1_first != '')).astype(int)
        merged['country_match'] = (merged['s1_country_clean'] == merged['cand_country_clean']).astype(int)

        # --- Remaining features genuinely need a per-pair Python-level call
        # (set intersection for jaccard on precomputed sets; rapidfuzz/
        # difflib ratio for fuzzy string similarity). Everything expensive
        # about "knowing" each entity's tokens/length/etc. was already done
        # once in build_entity_lookup, so this loop is now much cheaper. ---
        name_jac, name_sr, addr_jac, addr_sr = [], [], [], []
        for row in merged.itertuples(index=False):
            name_jac.append(set_jaccard(row.s1_name_tokens, row.cand_name_tokens))
            addr_jac.append(set_jaccard(row.s1_address_tokens, row.cand_address_tokens))
            name_sr.append(fuzzy_ratio(row.s1_business_name_clean, row.cand_business_name_clean))
            addr_sr.append(fuzzy_ratio(row.s1_business_address_clean, row.cand_business_address_clean))

        merged['name_token_jaccard'] = name_jac
        merged['name_seq_ratio'] = name_sr
        merged['address_token_jaccard'] = addr_jac
        merged['address_seq_ratio'] = addr_sr
        merged['name_tfidf_cosine'] = name_tfidf_cosine

        out_cols = ['source1_entity_id', 'candidate_entity_id', 'name_token_jaccard', 'name_seq_ratio',
                    'name_tfidf_cosine', 'address_token_jaccard', 'address_seq_ratio', 'country_match',
                    'name_length_diff', 'name_first_token_match']
        result = merged[out_cols].copy()

        if gt_pairs_df is not None:
            result = result.merge(gt_pairs_df, on=['source1_entity_id', 'candidate_entity_id'], how='left')
            result['label'] = result['label'].fillna(0).astype(int)

        result.to_csv(args.out, sep='\t', index=False, mode='w' if first_write else 'a', header=first_write)
        first_write = False
        total_rows += len(result)

        print(f"  chunk {chunk_i + 1}: {len(result)} rows in {time.time() - t0:.1f}s (total so far: {total_rows})")
        log_memory(f"after chunk {chunk_i + 1}")

        del pairs_chunk, merged, result
        gc.collect()

    print(f"\nDone. {total_rows} feature rows written to {args.out} in {time.time() - overall_start:.1f}s")
    if gt_pairs_df is not None:
        print("Note: label distribution should be checked separately — "
              "run pd.read_csv(args.out, sep='\\t')['label'].value_counts()")


if __name__ == "__main__":
    main()
