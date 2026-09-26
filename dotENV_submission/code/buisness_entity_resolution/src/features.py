"""
Person C — Similarity / Feature Engineering (memory-constrained version)

Rewritten for small instances (e.g. ml.t3.medium, 4GB RAM) and the real pair
volume this project produces (tens of millions of candidate pairs at full
scale). Key changes from the original version:
  - Candidate pairs are processed in CHUNKS, never loaded fully into memory
  - Entity lookups use a single pandas DataFrame (merge-based), not a Python
    dict-of-dicts, which is far more memory-efficient at millions of rows
  - Ground truth labels are attached via a merge, not a giant Python set
  - Output is written incrementally (append per chunk), not built in memory
    and written once at the end

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
import time
from difflib import SequenceMatcher
from urllib.parse import urlparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LOOKUP_COLS = ['entity_id', 'business_name_clean', 'business_address_clean', 'country_clean']


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
# Similarity functions (row-wise, applied per chunk — bounded cost)
# ---------------------------------------------------------------------
def token_jaccard(a, b):
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def seq_ratio(a, b):
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------
# Build a single entity lookup DataFrame (indexed by entity_id) —
# proven cheap in the blocking-stage audit (~12.5M entity_id+country rows
# cost well under 1GB); adding name+address roughly doubles that, still
# comfortably within a 4GB budget alongside chunked candidate processing.
# ---------------------------------------------------------------------
def build_entity_lookup(paths, chunksize=300_000):
    frames = []
    for path in paths:
        for chunk in pd.read_csv(path, sep='\t', dtype=str, usecols=LOOKUP_COLS, chunksize=chunksize):
            frames.append(chunk.fillna(''))
    lookup = pd.concat(frames, ignore_index=True).set_index('entity_id')
    del frames
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
    p.add_argument('--chunksize', type=int, default=200_000, help="Candidate pairs processed per chunk")
    args = p.parse_args()

    s1_path = resolve_local_path(args.s1, args.local_cache_dir)
    s2_path = resolve_local_path(args.s2, args.local_cache_dir)
    s3_path = resolve_local_path(args.s3, args.local_cache_dir)
    cand_path = resolve_local_path(args.candidates, args.local_cache_dir)
    gt_path = resolve_local_path(args.ground_truth, args.local_cache_dir) if args.ground_truth else None

    print("Building entity lookup (entity_id -> name/address/country)...")
    t0 = time.time()
    lookup = build_entity_lookup([s1_path, s2_path, s3_path])
    print(f"  lookup built: {len(lookup)} entities in {time.time() - t0:.1f}s")
    log_memory("after building lookup")

    # Fit ONE global TF-IDF vectorizer up front (needs the name column only,
    # cheap — already proven cheap in blocking.py's audit-style reads).
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
    vectorizer.fit(lookup['business_name_clean'])
    log_memory("after fitting vectorizer")

    # Ground truth: build a lightweight pair-membership DataFrame for merging
    # (NOT a giant Python set of tuples — a merge-based join is much cheaper
    # at millions of rows).
    gt_pairs_df = None
    if gt_path:
        gt = pd.read_csv(gt_path, sep='\t', dtype=str).fillna('')
        exploded = []
        for row in gt.itertuples(index=False):
            if not row.matched_entity_ids:
                continue
            for m in row.matched_entity_ids.split(','):
                exploded.append((row.source1_entity_id, m.strip()))
        gt_pairs_df = pd.DataFrame(exploded, columns=['source1_entity_id', 'candidate_entity_id'])
        gt_pairs_df['label'] = 1
        del exploded, gt
        gc.collect()
        print(f"  ground truth: {len(gt_pairs_df)} true pairs")
        log_memory("after building ground truth")

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

        # Vectorized attach of s1 and candidate fields via merge against the lookup
        merged = pairs_chunk.merge(
            lookup.add_prefix('s1_'), left_on='source1_entity_id', right_index=True, how='left'
        ).merge(
            lookup.add_prefix('cand_'), left_on='candidate_entity_id', right_index=True, how='left'
        )

        # TF-IDF cosine similarity, vectorized per chunk
        v1 = vectorizer.transform(merged['s1_business_name_clean'])
        v2 = vectorizer.transform(merged['cand_business_name_clean'])
        # row-wise cosine of paired vectors (not full cross product): multiply elementwise, sum
        name_tfidf_cosine = v1.multiply(v2).sum(axis=1).A1  # TF-IDF rows are L2-normalized -> this IS cosine sim

        merged['name_token_jaccard'] = merged.apply(
            lambda r: token_jaccard(r['s1_business_name_clean'], r['cand_business_name_clean']), axis=1)
        merged['name_seq_ratio'] = merged.apply(
            lambda r: seq_ratio(r['s1_business_name_clean'], r['cand_business_name_clean']), axis=1)
        merged['address_token_jaccard'] = merged.apply(
            lambda r: token_jaccard(r['s1_business_address_clean'], r['cand_business_address_clean']), axis=1)
        merged['address_seq_ratio'] = merged.apply(
            lambda r: seq_ratio(r['s1_business_address_clean'], r['cand_business_address_clean']), axis=1)
        merged['name_tfidf_cosine'] = name_tfidf_cosine
        merged['country_match'] = (merged['s1_country_clean'] == merged['cand_country_clean']).astype(int)

        out_cols = ['source1_entity_id', 'candidate_entity_id', 'name_token_jaccard', 'name_seq_ratio',
                    'name_tfidf_cosine', 'address_token_jaccard', 'address_seq_ratio', 'country_match']
        result = merged[out_cols].copy()

        if gt_pairs_df is not None:
            result = result.merge(gt_pairs_df, on=['source1_entity_id', 'candidate_entity_id'], how='left')
            result['label'] = result['label'].fillna(0).astype(int)

        result.to_csv(args.out, sep='\t', index=False, mode='w' if first_write else 'a', header=first_write)
        first_write = False
        total_rows += len(result)

        print(f"  chunk {chunk_i + 1}: {len(result)} rows in {time.time() - t0:.1f}s (total so far: {total_rows})")
        log_memory(f"after chunk {chunk_i + 1}")

        del pairs_chunk, merged, v1, v2, result
        gc.collect()

    print(f"\nDone. {total_rows} feature rows written to {args.out} in {time.time() - overall_start:.1f}s")
    if gt_pairs_df is not None:
        print("Note: label distribution should be checked separately — "
              "run pd.read_csv(args.out, sep='\\t')['label'].value_counts()")


if __name__ == "__main__":
    main()