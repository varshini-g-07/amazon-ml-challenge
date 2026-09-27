"""
Person B — Blocking / Candidate Generation (memory-constrained version)

Rewritten for small instances (e.g. ml.t3.medium, 4GB RAM). The key change
from the earlier version: instead of loading s1/s2/s3 fully into memory and
then filtering by country, this script PARTITIONS each file to disk by
country first (streaming, low memory), then processes ONE country at a time,
fully releasing memory before moving to the next. Peak memory is bounded by
your largest single country's data, not by the whole dataset.

Produces:
  candidate_pairs_long.tsv  (internal handoff to Person C: one row per pair)
  candidate_pairs.tsv       (final submission format: one row per Source 1 entity)

Usage (audit only — cheap, run this first):
  python blocking.py --s1 ... --s2 ... --s3 ... \
      --out-long candidate_pairs_long.tsv --out-wide candidate_pairs.tsv \
      --ground-truth train_ground_truth.tsv --audit-only \
      --local-cache-dir /tmp/er_cache

Usage (dry run on a small sample first):
  python blocking.py --s1 ... --s2 ... --s3 ... \
      --out-long candidate_pairs_long.tsv --out-wide candidate_pairs.tsv \
      --dry-run 2000 --local-cache-dir /tmp/er_cache

Usage (full run):
  python blocking.py --s1 ... --s2 ... --s3 ... \
      --out-long candidate_pairs_long.tsv --out-wide candidate_pairs.tsv \
      --local-cache-dir /tmp/er_cache --partition-dir /tmp/er_partitions
"""

import argparse
import gc
import os
import shutil
import time
from urllib.parse import urlparse

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

BLOCKING_COLS = ['entity_id', 'business_name_clean', 'country_clean']


def log_memory(label=""):
    try:
        import psutil
        rss_gb = psutil.Process().memory_info().rss / 1e9
        total_gb = psutil.virtual_memory().total / 1e9
        print(f"    [memory] {label}: {rss_gb:.2f} GB used / {total_gb:.2f} GB total", flush=True)
    except ImportError:
        pass  # psutil not installed — skip silently rather than fail the run


# ---------------------------------------------------------------------
# S3 download helper
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
# Country label audit (chunked — never loads full name/address columns)
# ---------------------------------------------------------------------
def audit_country_labels(s1_path, s23_paths, ground_truth_path=None, chunksize=200_000):
    print("\n=== Country label audit ===")

    def chunked_value_counts(path, label):
        counts = {}
        n_empty = 0
        for chunk in pd.read_csv(path, sep='\t', dtype=str, usecols=['country_clean'], chunksize=chunksize):
            chunk = chunk.fillna('')
            vc = chunk['country_clean'].value_counts()
            for k, v in vc.items():
                counts[k] = counts.get(k, 0) + v
            n_empty += (chunk['country_clean'].str.len() == 0).sum()
        print(f"\n{label} country_clean value counts:")
        for k, v in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {k!r}: {v}")
        print(f"{label} empty/missing country_clean: {n_empty}")

    chunked_value_counts(s1_path, "s1")
    for i, p in enumerate(s23_paths, start=2):
        chunked_value_counts(p, f"s{i}")

    if not ground_truth_path:
        print("\nNo --ground-truth given, skipping true-match mismatch check.")
        return

    # Build entity_id -> country_clean lookup, chunked, columns restricted
    country_lookup = {}
    for path in [s1_path] + s23_paths:
        for chunk in pd.read_csv(path, sep='\t', dtype=str, usecols=['entity_id', 'country_clean'], chunksize=chunksize):
            chunk = chunk.fillna('')
            country_lookup.update(dict(zip(chunk['entity_id'], chunk['country_clean'])))

    gt = pd.read_csv(ground_truth_path, sep='\t', dtype=str).fillna('')
    mismatches, total_pairs, examples = 0, 0, []
    for row in gt.itertuples(index=False):
        if not row.matched_entity_ids:
            continue
        s1_country = country_lookup.get(row.source1_entity_id)
        for match_id in row.matched_entity_ids.split(','):
            match_id = match_id.strip()
            match_country = country_lookup.get(match_id)
            total_pairs += 1
            if s1_country != match_country:
                mismatches += 1
                if len(examples) < 10:
                    examples.append((row.source1_entity_id, s1_country, match_id, match_country))

    pct = 100 * mismatches / total_pairs if total_pairs else 0.0
    print(f"\nTrue-match pairs with mismatched country_clean: {mismatches}/{total_pairs} ({pct:.2f}%)")
    if examples:
        print("Examples (s1_id, s1_country, match_id, match_country):")
        for ex in examples:
            print(f"  {ex}")
    if pct > 2.0:
        print(f"\nWARNING: {pct:.2f}% of true matches would be lost under strict country filtering.")
    else:
        print(f"\nCountry filtering looks safe (<2% of true matches lost).")

    del country_lookup
    gc.collect()


# ---------------------------------------------------------------------
# Partition each file by country, streaming (low memory)
# ---------------------------------------------------------------------
def partition_by_country(input_path, output_dir, prefix, chunksize=100_000, sample_n=None):
    """Streams input_path in chunks, appending each row to
    {output_dir}/{prefix}__{country}.tsv. Returns the set of countries seen."""
    os.makedirs(output_dir, exist_ok=True)
    countries_seen = set()
    headers_written = set()
    rows_written = 0

    for chunk in pd.read_csv(input_path, sep='\t', dtype=str, usecols=BLOCKING_COLS, chunksize=chunksize):
        chunk = chunk.fillna('')
        if sample_n is not None:
            remaining = sample_n - rows_written
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        for country, group in chunk.groupby('country_clean'):
            out_path = os.path.join(output_dir, f"{prefix}__{country or 'UNKNOWN'}.tsv")
            write_header = out_path not in headers_written and not os.path.exists(out_path)
            group.to_csv(out_path, sep='\t', index=False, mode='a', header=write_header)
            headers_written.add(out_path)
            countries_seen.add(country)

        rows_written += len(chunk)
        if sample_n is not None and rows_written >= sample_n:
            break

    print(f"  Partitioned {prefix}: {rows_written} rows across {len(countries_seen)} countries")
    return countries_seen


# ---------------------------------------------------------------------
# Per-country candidate generation
# ---------------------------------------------------------------------
def _tokens_for(name, min_token_len=2):
    return {t for t in name.split() if len(t) >= min_token_len}


def build_token_index(names, min_token_len=2, max_df_ratio=0.05, max_postings_per_token=5000):
    """
    Maps token -> list of row indices, EXCLUDING tokens that appear in more
    than max_df_ratio of documents (e.g. 'ltd', 'inc', 'street' — too common
    to be a useful blocking signal, and would blow up candidate lists).
    This is what replaces brute-force n1*n23 comparison: two records are only
    ever compared if they share at least one distinctive token.

    Two-pass design: tokens are recomputed per document in each pass (cheap
    string splitting) rather than stored for all documents at once — storing
    a Python set() per document for millions of documents simultaneously is
    what blows up memory at full country scale (e.g. ~6M docs for 'us').
    """
    n = len(names)
    t0 = time.time()

    # Pass 1: document frequency only — no per-document data retained
    doc_freq = defaultdict(int)
    for name in names:
        for t in _tokens_for(name, min_token_len):
            doc_freq[t] += 1
    print(f"    token index pass 1/2 done ({n} docs, {len(doc_freq)} unique tokens) in {time.time() - t0:.1f}s", flush=True)

    max_df = max(1, int(max_df_ratio * n))
    common_tokens = {t for t, c in doc_freq.items() if c > max_df}
    del doc_freq

    # Pass 2: build the index, skipping common tokens, capped per token
    t1 = time.time()
    index = defaultdict(list)
    for i, name in enumerate(names):
        for t in _tokens_for(name, min_token_len) - common_tokens:
            lst = index[t]
            if len(lst) < max_postings_per_token:
                lst.append(i)
    print(f"    token index pass 2/2 done in {time.time() - t1:.1f}s", flush=True)

    return index, common_tokens


def process_country(country, s1_dir, s23_dir, k, sim_floor, max_candidates_per_doc=300, batch_size=5000,
                     out_path=None, flush_every=100_000):
    """
    Two-stage blocking, kept intact:
      Stage 1 — token-overlap filtering (bounds candidates to a small subset
                per S1 record; this is what keeps runtime/memory feasible at
                full country scale, e.g. ~6.19M candidates for 'us')
      Stage 2 — hashed cosine scoring, computed per S1 record against ONLY
                that record's token-overlap candidates. Nothing whole-country
                sized is ever built — each record hashes just itself plus its
                own <=max_candidates_per_doc candidates on demand, so peak
                memory per iteration is bounded regardless of country size.

    Results are flushed to out_path every flush_every matches (instead of
    accumulated in memory for the whole country) — with up to k matches per
    record, an unflushed `records` list can itself grow into multiple GB
    over a large country's full run.

    Top-k selection uses np.argpartition instead of a full Python sort.
    """
    s1_path = os.path.join(s1_dir, f"s1__{country or 'UNKNOWN'}.tsv")
    s23_path_2 = os.path.join(s23_dir, f"s2__{country or 'UNKNOWN'}.tsv")
    s23_path_3 = os.path.join(s23_dir, f"s3__{country or 'UNKNOWN'}.tsv")

    if not os.path.exists(s1_path):
        return pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'blocking_score'])

    s1_group = pd.read_csv(s1_path, sep='\t', dtype=str).fillna('')

    frames = []
    for p in [s23_path_2, s23_path_3]:
        if os.path.exists(p):
            frames.append(pd.read_csv(p, sep='\t', dtype=str).fillna(''))
    if not frames:
        del s1_group
        return pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'blocking_score'])

    s23_group = pd.concat(frames, ignore_index=True)
    del frames

    s23_names = s23_group['business_name_clean'].tolist()
    s1_names = s1_group['business_name_clean'].tolist()
    s1_ids = s1_group['entity_id'].values
    s23_ids = s23_group['entity_id'].values

    # Free the DataFrames now — everything needed is extracted into the
    # lists/arrays above. Holding these alongside the token index and the
    # TF-IDF matrices was adding to peak memory for no benefit.
    del s1_group, s23_group
    gc.collect()

    # Stage 1: token-overlap blocking — this is what avoids brute-force
    # n1*n23 comparison. Only records sharing a distinctive token are ever
    # compared in Stage 2.
    index, common_tokens = build_token_index(s23_names)
    log_memory("after token index")

    # HashingVectorizer instead of TfidfVectorizer: no fitting phase, fixed
    # output width regardless of corpus size. norm='l2' bakes in
    # normalization so a plain dot product IS the cosine similarity.
    #
    # On instances with real headroom (12GB+, e.g. Colab), hash the WHOLE
    # country ONCE in a fast vectorized batch call, then just SLICE into it
    # per record — instead of a separate tiny hasher.transform() call per
    # S1 record. The per-record version was necessary to survive a 4GB
    # instance, but it pays real per-call Python/tokenization overhead
    # hundreds of thousands to millions of times over, which is far too slow
    # once memory isn't the binding constraint. Estimated final X23 size for
    # a full country (~1-3GB) comfortably fits in 12GB alongside everything
    # else, so there's no need to pay that speed cost anymore.
    log_memory("before hashing vectorizer")
    t_hash = time.time()
    hasher = HashingVectorizer(analyzer='char_wb', ngram_range=(2, 4), n_features=2 ** 17,
                                alternate_sign=False, norm='l2', dtype=np.float32)
    X23 = hasher.transform(s23_names)
    X1 = hasher.transform(s1_names)
    print(f"    batch hashing done (X23={X23.shape}, X1={X1.shape}) in {time.time() - t_hash:.1f}s", flush=True)
    log_memory("after batch hashing")

    # Flush accumulated matches to disk periodically instead of holding the
    # entire country's results in memory until the end — with up to k=20
    # matches per record, a large country's `records` list can itself grow
    # into multiple GB by the time the country finishes, which is what was
    # driving the memory growth observed during a live run.
    records = []
    total_written = 0

    def flush(force=False):
        nonlocal records, total_written
        if not records or (not force and len(records) < flush_every):
            return
        chunk_df = pd.DataFrame(records, columns=['source1_entity_id', 'candidate_entity_id', 'blocking_score'])
        if out_path:
            # Header goes on only if the file doesn't exist yet or is empty —
            # checked against the file itself (not a local flag) since this
            # function is called fresh per country, and later countries must
            # APPEND without repeating the header.
            need_header = (not os.path.exists(out_path)) or os.path.getsize(out_path) == 0
            chunk_df.to_csv(out_path, sep='\t', index=False, mode='a', header=need_header)
        total_written += len(chunk_df)
        records = []
        return chunk_df

    all_flushed_frames = [] if not out_path else None  # only used if out_path is None (backward compatible)

    t_score = time.time()
    for i, name in enumerate(s1_names):
        tokens = {t for t in name.split() if len(t) >= 2} - common_tokens
        candidate_idx = set()
        for t in tokens:
            candidate_idx.update(index.get(t, []))
            if len(candidate_idx) >= max_candidates_per_doc:
                break

        if not candidate_idx:
            continue  # no shared distinctive token — no candidates from this pass

        candidate_idx = np.array(list(candidate_idx)[:max_candidates_per_doc])

        # Cheap array slicing into the already-computed X1/X23 — no new
        # hashing work happens here at all, this is just sparse row lookup
        # and a small matrix multiply.
        x1_vec = X1[i]
        cand_vecs = X23[candidate_idx]

        sims = (x1_vec @ cand_vecs.T).toarray().ravel()

        mask = sims >= sim_floor
        if np.any(mask):
            valid_idx = candidate_idx[mask]
            valid_sims = sims[mask]
            if len(valid_sims) > k:
                top_k = np.argpartition(valid_sims, -k)[-k:]
                top_k = top_k[np.argsort(-valid_sims[top_k])]
            else:
                top_k = np.argsort(-valid_sims)
            s1_id = s1_ids[i]
            for j in top_k:
                records.append((s1_id, s23_ids[valid_idx[j]], float(valid_sims[j])))

        if len(records) >= flush_every:
            flushed = flush()
            if flushed is not None and all_flushed_frames is not None:
                all_flushed_frames.append(flushed)
            total_written += flush_every if out_path else 0

        if i % 200_000 == 0 and i > 0:
            print(f"    scoring progress: {i}/{len(s1_names)} records", flush=True)
            log_memory(f"during scoring (record {i})")

    # Final flush — whatever's left in the buffer when the loop ends
    flushed = flush(force=True)
    if flushed is not None and all_flushed_frames is not None:
        all_flushed_frames.append(flushed)

    print(f"    scoring loop done ({len(s1_names)} records) in {time.time() - t_score:.1f}s", flush=True)

    del hasher, X1, X23, index, common_tokens
    gc.collect()

    if out_path:
        return total_written  # int count — results already written incrementally to out_path
    return pd.concat(all_flushed_frames, ignore_index=True) if all_flushed_frames else \
        pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'blocking_score'])


def to_wide_format(long_df, all_s1_ids, id_col='candidate_entity_id'):
    grouped = long_df.groupby('source1_entity_id')[id_col].apply(lambda ids: ','.join(ids))
    wide = pd.DataFrame({'source1_entity_id': all_s1_ids})
    wide = wide.merge(grouped.rename('candidate_entity_ids'), on='source1_entity_id', how='left')
    wide['candidate_entity_ids'] = wide['candidate_entity_ids'].fillna('')
    return wide


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--s1', required=True)
    p.add_argument('--s2', required=True)
    p.add_argument('--s3', required=True)
    p.add_argument('--out-long', required=True)
    p.add_argument('--out-wide', required=True)
    p.add_argument('--k', type=int, default=20)
    p.add_argument('--sim-floor', type=float, default=0.3)
    p.add_argument('--ground-truth', default=None)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--dry-run', type=int, default=None, metavar='N')
    p.add_argument('--local-cache-dir', default=None)
    p.add_argument('--partition-dir', default='/tmp/er_partitions')
    p.add_argument('--chunksize', type=int, default=100_000)
    args = p.parse_args()

    s1_path = resolve_local_path(args.s1, args.local_cache_dir)
    s2_path = resolve_local_path(args.s2, args.local_cache_dir)
    s3_path = resolve_local_path(args.s3, args.local_cache_dir)
    gt_path = resolve_local_path(args.ground_truth, args.local_cache_dir) if args.ground_truth else None
    log_memory("after download")

    audit_country_labels(s1_path, [s2_path, s3_path], ground_truth_path=gt_path)
    log_memory("after audit")
    if args.audit_only:
        return

    # In dry-run mode, cap s2/s3 too — otherwise per-country processing still
    # runs against the full multi-million-row candidate pool, defeating the
    # purpose of a quick timing/memory check.
    s23_sample_n = (args.dry_run * 10) if args.dry_run else None  # generous but bounded

    # Always start from a clean partition directory — the partitioning step
    # appends to files, so stale partitions from a previous (e.g. failed or
    # differently-scoped) run would otherwise silently mix into this one.
    if os.path.exists(args.partition_dir):
        print(f"Clearing stale partitions in {args.partition_dir} ...")
        shutil.rmtree(args.partition_dir)

    print(f"\nPartitioning files by country into {args.partition_dir} ...")
    partition_by_country(s1_path, args.partition_dir, 's1', chunksize=args.chunksize, sample_n=args.dry_run)
    partition_by_country(s2_path, args.partition_dir, 's2', chunksize=args.chunksize, sample_n=s23_sample_n)
    partition_by_country(s3_path, args.partition_dir, 's3', chunksize=args.chunksize, sample_n=s23_sample_n)
    log_memory("after partitioning")

    countries = sorted(set(
        f.split('__', 1)[1].rsplit('.tsv', 1)[0]
        for f in os.listdir(args.partition_dir) if f.startswith('s1__')
    ))
    print(f"\nProcessing {len(countries)} country groups: {countries}")

    # Start from a clean output file — process_country now writes to it
    # incrementally (per country, flushed periodically within a country too)
    # rather than returning one big DataFrame that main() writes once.
    local_out_long = args.out_long
    if local_out_long.startswith('s3://'):
        local_out_long = os.path.join(args.partition_dir, '_candidate_pairs_long_local.tsv')
    if os.path.exists(local_out_long):
        os.remove(local_out_long)

    total_pairs = 0
    overall_start = time.time()

    for country in countries:
        t0 = time.time()
        n_written = process_country(country, args.partition_dir, args.partition_dir, args.k, args.sim_floor,
                                     out_path=local_out_long)
        total_pairs += n_written
        print(f"  [{country}] {n_written} pairs in {time.time() - t0:.1f}s")
        log_memory(f"after country {country}")

    # If the destination is S3, upload the locally-built file now
    if args.out_long.startswith('s3://'):
        import boto3
        parsed = urlparse(args.out_long)
        boto3.client('s3').upload_file(local_out_long, parsed.netloc, parsed.path.lstrip('/'))
        print(f"Uploaded {local_out_long} -> {args.out_long}")

    print(f"\nTotal candidate generation time: {time.time() - overall_start:.1f}s, {total_pairs} pairs written")

    # Build wide-format output from the long file we just wrote (streamed, not fully loaded)
    all_s1_ids = pd.read_csv(s1_path, sep='\t', dtype=str, usecols=['entity_id'])['entity_id']
    if args.dry_run:
        all_s1_ids = all_s1_ids.head(args.dry_run)

    long_df = pd.read_csv(local_out_long, sep='\t', dtype=str)
    wide_df = to_wide_format(long_df, all_s1_ids.values)
    wide_df.to_csv(args.out_wide, sep='\t', index=False)
    print(f"Wrote {len(wide_df)} Source 1 rows -> {args.out_wide}")

    n_with_candidates = (wide_df['candidate_entity_ids'] != '').sum()
    print(f"Source 1 entities with >=1 candidate: {n_with_candidates}/{len(wide_df)}")


if __name__ == "__main__":
    main()
