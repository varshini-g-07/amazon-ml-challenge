# Business Entity Resolution — Pipeline README

This folder contains a self-contained, runnable pipeline that takes the raw
Source 1 / Source 2 / Source 3 TSVs and produces the two required submission
files: `matching_results.tsv` and `candidate_pairs.tsv`.

## Pipeline overview

```
raw source files
      │
      ▼
  normalize.py        (clean + transliterate names/addresses, per file)
      │
      ▼
  blocking.py          (candidate generation — cuts the search space)
      │
      ▼
  features.py           (similarity features per candidate pair)
      │
      ▼
  model.py            (train a classifier + tune a threshold; or predict)
      │
      ▼
matching_results.tsv + candidate_pairs.tsv
```

Four scripts, one responsibility each:

| Script | Purpose |
|---|---|
| `src/normalize.py` | Cleans one raw source file: accent-stripping, transliteration of non-Latin text, abbreviation expansion (e.g. "Rd" → "road", "Ltd" → "limited"), punctuation stripping. Run once per source file (6 total: train ×3, test ×3). |
| `src/blocking.py` | Candidate generation. Partitions by country, then blocks per Source 1 entity using a hashed token index + cosine similarity, keeping the top-`k` candidates above `--sim-floor`. Produces both the long-format pairs file (fed to `features.py`) and the wide, one-row-per-entity format required for submission. |
| `src/features.py` | Computes similarity features (name/address token Jaccard, fuzzy ratio, TF-IDF cosine, country match, etc.) for every candidate pair. Run once for train (with `--ground-truth`, producing labels) and once for test (without). |
| `src/model.py` | `train`: fits a LightGBM classifier on labeled training features, then sweeps a decision threshold to maximize macro F₀.₅ on a held-out validation split (grouped by `source1_entity_id` so no entity leaks across the split). `predict`: applies the trained model + threshold to test features and writes the two final submission files. |

## Setup

```bash
pip install -r requirements.txt
```

If you're running in a hosted notebook (Colab / SageMaker) and pulling data
from S3, all four scripts accept `s3://...` paths directly wherever they take
a file argument — they cache-download to `--local-cache-dir` automatically.
If your data is already local, just pass local paths and omit
`--local-cache-dir`.

## Reproducing end-to-end

The commands below assume this layout (adjust paths to match your own; the
scripts don't require this exact structure):

```
dataset/
  train/
    train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
  test/
    test_source1.tsv   test_source2.tsv   test_source3.tsv
normalized/
intermediate/
output/
```

### 1. Normalize every source file

Run once per file (six invocations total — three train, three test):

```bash
python src/normalize.py dataset/train/train_source1.tsv normalized/normalized_train1.tsv --workers 4
python src/normalize.py dataset/train/train_source2.tsv normalized/normalized_train2.tsv --workers 4
python src/normalize.py dataset/train/train_source3.tsv normalized/normalized_train3.tsv --workers 4

python src/normalize.py dataset/test/test_source1.tsv normalized/normalized_test1.tsv --workers 4
python src/normalize.py dataset/test/test_source2.tsv normalized/normalized_test2.tsv --workers 4
python src/normalize.py dataset/test/test_source3.tsv normalized/normalized_test3.tsv --workers 4
```

Each output file adds `business_name_clean`, `business_address_clean`,
`country_clean`, and `was_transliterated` columns alongside the originals.

### 2. Blocking — generate candidate pairs

**Train:**
```bash
python src/blocking.py \
    --s1 normalized/normalized_train1.tsv \
    --s2 normalized/normalized_train2.tsv \
    --s3 normalized/normalized_train3.tsv \
    --out-long intermediate/candidate_pairs_long_train.tsv \
    --out-wide intermediate/candidate_pairs_train.tsv \
    --ground-truth dataset/train/train_ground_truth.tsv \
    --partition-dir /tmp/er_partitions_train
```

**Test:**
```bash
python src/blocking.py \
    --s1 normalized/normalized_test1.tsv \
    --s2 normalized/normalized_test2.tsv \
    --s3 normalized/normalized_test3.tsv \
    --out-long intermediate/candidate_pairs_long_test.tsv \
    --out-wide intermediate/candidate_pairs_test.tsv \
    --partition-dir /tmp/er_partitions_test
```

Notes:
- `--ground-truth` is only used to print a country-mismatch audit (it does
  not affect candidate generation itself) — omit it for the test run since
  no ground truth exists for test data.
- `--k` (default 20) and `--sim-floor` (default 0.3) control candidate-set
  size vs. recall ceiling; the values above are the defaults used to produce
  the submitted results. See `Documentation_template.md` for the tuning
  rationale.
- Run with `--audit-only` first if you want to sanity-check country label
  distributions before committing to a full partitioning run.

### 3. Feature engineering

**Train** (produces labeled features):
```bash
python src/features.py \
    --s1 normalized/normalized_train1.tsv \
    --s2 normalized/normalized_train2.tsv \
    --s3 normalized/normalized_train3.tsv \
    --candidates intermediate/candidate_pairs_long_train.tsv \
    --out intermediate/features_train.tsv \
    --ground-truth dataset/train/train_ground_truth.tsv
```

**Test** (no labels):
```bash
python src/features.py \
    --s1 normalized/normalized_test1.tsv \
    --s2 normalized/normalized_test2.tsv \
    --s3 normalized/normalized_test3.tsv \
    --candidates intermediate/candidate_pairs_long_test.tsv \
    --out intermediate/features_test.tsv
```

Both use the **long-format** candidate pairs file (one row per pair), not
the wide submission-format file.

### 4. Train the model

```bash
python src/model.py train \
    --features intermediate/features_train.tsv \
    --model-out intermediate/model.txt \
    --threshold-out intermediate/threshold.txt
```

This splits train/validation by `source1_entity_id` (80/20), trains a
LightGBM classifier, sweeps decision thresholds from 0.1 to 0.99 against
macro F₀.₅ on the validation split, and saves both the model and the chosen
threshold.

### 5. Predict on the test set

```bash
python src/model.py predict \
    --features intermediate/features_test.tsv \
    --candidates intermediate/candidate_pairs_test.tsv \
    --model intermediate/model.txt \
    --threshold intermediate/threshold.txt \
    --out-matches output/matching_results.tsv \
    --out-candidates output/candidate_pairs.tsv
```

`--candidates` here is the **wide-format** file from the test blocking run
(`candidate_pairs_test.tsv`) — it's used both to ensure every Source 1 test
entity appears in the output (including zero-match singletons) and is
copied through unchanged as the submission's `candidate_pairs.tsv`.

### 6. Validate before submitting

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

`utils/validate_submission.py` is provided by the challenge organizers
(stdlib only) — see the problem statement for details. It should print
`PASS` before you submit.

## Memory / performance notes

`blocking.py` and `features.py` were both rewritten to run within a single
memory-constrained instance (originally 4GB, validated up to ~13GB) at full
dataset scale — see the docstring at the top of each file for the specific
techniques used (country-partitioned streaming, filtering entity lookups to
only IDs actually referenced by candidates, batched TF-IDF fitting/transform,
interned token tuples instead of persistent per-entity sets, etc.). If you
run on a smaller or larger instance than that, the `--chunksize` /
`--partition-dir` / batch-size constants in each script are the first things
to tune.

## Methodology

See `Documentation_template.md` (in the submission zip root) for the full
write-up: blocking strategy and tuning, feature choices, model architecture,
and threshold selection.
