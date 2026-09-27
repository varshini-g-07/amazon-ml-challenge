"""
Person D — Model / Evaluation

Trains a classifier on labeled features (from Person C), tunes a decision
threshold against the F_0.5 metric on a held-out validation split, then
applies the trained model + threshold to test features to produce the
final matching_results.tsv (and copies through candidate_pairs.tsv).

Usage (train + tune):
  python model.py train \
      --features features_train.tsv --model-out model.txt \
      --threshold-out threshold.txt

Usage (predict on test):
  python model.py predict \
      --features features_test.tsv --candidates candidate_pairs_test.tsv \
      --model model.txt --threshold threshold.txt \
      --out-matches matching_results.tsv --out-candidates candidate_pairs.tsv
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb

FEATURE_COLS = [
    'name_token_jaccard', 'name_seq_ratio', 'name_tfidf_cosine',
    'address_token_jaccard', 'address_seq_ratio', 'country_match',
    'name_length_diff', 'name_first_token_match',
]


def macro_f_beta(preds_df, beta=0.5):
    """Kept for backward compatibility / small-scale use — see the grouped
    version below for the threshold sweep, which avoids re-grouping per
    threshold."""
    scores = []
    for s1_id, group in preds_df.groupby('source1_entity_id'):
        true_ids = set(group.loc[group['label'] == 1, 'candidate_entity_id'])
        pred_ids = set(group.loc[group['predicted'] == 1, 'candidate_entity_id'])

        if not true_ids:
            scores.append(1.0 if not pred_ids else 0.0)
            continue

        if not pred_ids:
            scores.append(0.0)
            continue

        tp = len(true_ids & pred_ids)
        precision = tp / len(pred_ids)
        recall = tp / len(true_ids)
        f = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0
        scores.append(f)
    return float(np.mean(scores))


def build_grouped_eval_cache(val_df):
    """
    Groups the validation set by source1_entity_id ONCE, pre-computing each
    entity's true-match set and its (candidate_id, prob) pairs. The threshold
    sweep then reuses this cache directly instead of calling pandas
    .groupby() from scratch for every threshold — at full dataset scale
    (potentially millions of validation rows), rebuilding that groupby
    structure ~45 times (once per threshold tested) is pure waste, since the
    grouping itself never changes, only which rows cross a probability cutoff.
    """
    cache = []
    for s1_id, group in val_df.groupby('source1_entity_id', sort=False):
        true_ids = frozenset(group.loc[group['label'] == 1, 'candidate_entity_id'])
        cand_ids = group['candidate_entity_id'].to_numpy()
        probs = group['prob'].to_numpy()
        cache.append((true_ids, cand_ids, probs))
    return cache


def macro_f_beta_from_cache(cache, threshold, beta=0.5):
    """Sweeps a threshold over the pre-grouped cache — no groupby call here."""
    scores = []
    for true_ids, cand_ids, probs in cache:
        pred_mask = probs >= threshold
        if not true_ids:
            scores.append(1.0 if not pred_mask.any() else 0.0)
            continue
        if not pred_mask.any():
            scores.append(0.0)
            continue
        pred_ids = set(cand_ids[pred_mask])
        tp = len(true_ids & pred_ids)
        precision = tp / len(pred_ids)
        recall = tp / len(true_ids)
        f = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0
        scores.append(f)
    return float(np.mean(scores))


def _read_csv_with_progress(path, dtype_map, chunksize=2_000_000, label="features"):
    """
    Reads a large TSV in chunks and prints progress periodically, instead of
    one silent pd.read_csv() call that produces zero output for minutes.
    Colab (and similar hosted notebooks) can drop/interrupt a cell that goes
    silent for an extended stretch — regular prints keep the connection
    visibly alive and give real progress instead of a blank wait.
    """
    import time as _time
    t0 = _time.time()
    chunks = []
    total = 0
    for i, chunk in enumerate(pd.read_csv(path, sep='\t', dtype=dtype_map, chunksize=chunksize)):
        chunks.append(chunk)
        total += len(chunk)
        print(f"  reading {label}: {total} rows so far ({_time.time() - t0:.1f}s elapsed)", flush=True)
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"  loaded {len(df)} rows total in {_time.time() - t0:.1f}s")
    return df


def train(args):
    import gc
    # float32 for the numeric feature columns halves memory vs pandas'
    # float64 default — free win, no precision loss that matters for
    # similarity scores in [0, 1].
    dtype_map = {c: 'float32' for c in FEATURE_COLS}
    df = _read_csv_with_progress(args.features, dtype_map, label="features_train")

    # Split by source1_entity_id so no entity's pairs leak across train/val
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df['source1_entity_id']))
    train_df, val_df = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
    del df  # the full DataFrame is never needed again once train/val are split out —
    gc.collect()  # holding all three simultaneously at 43M+ rows is unnecessary peak memory

    X_train, y_train = train_df[FEATURE_COLS], train_df['label']
    X_val, y_val = val_df[FEATURE_COLS], val_df['label']

    print(f"  starting training on {len(X_train)} rows...", flush=True)
    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        class_weight='balanced', random_state=42, verbose=1
    )
    model.fit(X_train, y_train)
    print("  training done", flush=True)
    model.booster_.save_model(args.model_out)
    print(f"Model saved -> {args.model_out}")

    val_probs = model.predict_proba(X_val)[:, 1]
    val_df = val_df.copy()
    val_df['prob'] = val_probs

    print("  building grouped validation cache (one pass)...")
    t_group = time.time()
    eval_cache = build_grouped_eval_cache(val_df)
    print(f"  cache built ({len(eval_cache)} entities) in {time.time() - t_group:.1f}s")

    # Sweep thresholds against the pre-grouped cache — no re-grouping per threshold
    best_thresh, best_score = 0.5, -1
    for thresh in np.arange(0.1, 0.99, 0.02):
        score = macro_f_beta_from_cache(eval_cache, thresh, beta=0.5)
        print(f"  threshold={thresh:.2f} -> macro F0.5={score:.4f}")
        if score > best_score:
            best_thresh, best_score = thresh, score

    print(f"Best threshold: {best_thresh:.2f} (macro F0.5={best_score:.4f})")
    with open(args.threshold_out, 'w') as f:
        json.dump({'threshold': float(best_thresh), 'val_f0_5': float(best_score)}, f)
    print(f"Threshold saved -> {args.threshold_out}")


def predict(args):
    dtype_map = {c: 'float32' for c in FEATURE_COLS}
    df = _read_csv_with_progress(args.features, dtype_map, label="features_test")
    print("  reading candidates file...", flush=True)
    candidates_df = pd.read_csv(args.candidates, sep='\t', dtype=str).fillna('')
    print(f"  candidates: {len(candidates_df)} rows")

    model = lgb.Booster(model_file=args.model)
    with open(args.threshold) as f:
        threshold = json.load(f)['threshold']

    probs = model.predict(df[FEATURE_COLS])
    df = df.copy()
    df['predicted'] = (probs >= threshold).astype(int)

    matched = df[df['predicted'] == 1]
    grouped = matched.groupby('source1_entity_id')['candidate_entity_id'].apply(
        lambda ids: ','.join(ids)
    )

    # every Source 1 entity from candidate_pairs.tsv must appear, even with no match
    all_s1_ids = candidates_df['source1_entity_id'].unique()
    result = pd.DataFrame({'source1_entity_id': all_s1_ids})
    result = result.merge(grouped.rename('matched_entity_ids'), on='source1_entity_id', how='left')
    result['matched_entity_ids'] = result['matched_entity_ids'].fillna('')

    result.to_csv(args.out_matches, sep='\t', index=False)
    print(f"Wrote {len(result)} rows -> {args.out_matches}")

    candidates_df.to_csv(args.out_candidates, sep='\t', index=False)
    print(f"Wrote candidate_pairs.tsv -> {args.out_candidates}")

    n_matched = (result['matched_entity_ids'] != '').sum()
    print(f"Entities with >=1 match: {n_matched}/{len(result)}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)

    train_p = sub.add_parser('train')
    train_p.add_argument('--features', required=True)
    train_p.add_argument('--model-out', required=True)
    train_p.add_argument('--threshold-out', required=True)

    pred_p = sub.add_parser('predict')
    pred_p.add_argument('--features', required=True)
    pred_p.add_argument('--candidates', required=True)
    pred_p.add_argument('--model', required=True)
    pred_p.add_argument('--threshold', required=True)
    pred_p.add_argument('--out-matches', required=True)
    pred_p.add_argument('--out-candidates', required=True)

    args = p.parse_args()
    if args.cmd == 'train':
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
