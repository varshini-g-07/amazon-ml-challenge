"""
Scalable normalization for entity_id / business_name / business_address / country data.
Handles millions of rows with chunking and parallel transliteration.

Install once:
    pip install unidecode --break-system-packages

Usage:
    python normalize.py input_file.tsv output_file.tsv --workers 4 --chunksize 200000
"""

import re
import time
import logging
import argparse
import unicodedata
from multiprocessing import Pool

import pandas as pd
from unidecode import unidecode


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# 1. Abbreviation maps & pre-compiled regexes
# ---------------------------------------------------------------------
AMPERSAND_RE = re.compile(r'\s*&\s*')

GENERAL_ABBREV = {
    'inc': 'incorporated', 'corp': 'corporation', 'co': 'company',
    'llc': 'limited liability company', 'ltd': 'limited',
    'llp': 'limited liability partnership', 'pvt': 'private',
    'pc': 'professional corporation', 'sarl': 'sarl', 'sasu': 'sasu', 'sas': 'sas',
}

# NOTE: deliberately no 'r': 'rue' here — a bare "R" token is too overloaded
# across US/India/France addresses (block letters, locality names like "R Nagar",
# unit letters) to safely blanket-expand to French "rue" for every country.
ADDRESS_ABBREV = {
    'st': 'street', 'rd': 'road', 'ave': 'avenue', 'blvd': 'boulevard',
    'apt': 'apartment', 'ste': 'suite', 'dr': 'drive', 'ln': 'lane',
    'hwy': 'highway', 'nr': 'near', 'soc': 'society',
    'socty': 'society', 'ind': 'industrial', 'coop': 'cooperative',
}

NON_WORD_RE = re.compile(r'[^\w\s]')
MULTI_SPACE_RE = re.compile(r'\s+')
# Inserts spaces between concatenated Letter-Number boundaries (e.g. "18Shakti" -> "18 Shakti")
LETTER_NUM_BOUNDARIES = re.compile(r'(?<=\d)(?=[a-zA-Z])|(?<=[a-zA-Z])(?=\d)')

# Unicode range check for non-Latin scripts (used to decide which rows need unidecode)
NON_LATIN_RE = re.compile(r'[^\u0000-\u024F\u2000-\u206F\s]')

# Flags any remaining non-ASCII character (used to scope the expensive accent-stripping loop)
NON_ASCII_RE = re.compile(r'[^\x00-\x7F]')

# Collapses runs of 3+ identical characters down to 2 — tames unidecode's
# stuttering phonetic output on Indic scripts (e.g. "iisttrrnnn" -> "iisttrnn").
REPEATED_CHAR_RE = re.compile(r'(.)\1{2,}')

# Collapses dotted single-letter acronyms ("P.V.T.", "S.A.S") into a bare token
# ("PVT", "SAS") BEFORE punctuation stripping, so they still match the
# abbreviation dictionaries below instead of being shattered into single letters.
DOTTED_ACRONYM_RE = re.compile(r'\b(?:[A-Za-z]\.){2,}[A-Za-z]?\b')

# abbrev regex: optional trailing "s" (plural) and optional trailing "." (dotted form)
GENERAL_ABBREV_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(k) for k in GENERAL_ABBREV) + r')s?\.?\b'
)
ADDRESS_ABBREV_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(k) for k in ADDRESS_ABBREV) + r')s?\.?\b'
)


# ---------------------------------------------------------------------
# 2. Vectorized cleaning functions
# ---------------------------------------------------------------------
def strip_accents_and_casefold(series):
    """
    Decomposes Unicode characters (NFD) to separate base characters from accents,
    strips combining diacritics, and casefolds to lowercase.
    Only runs the expensive per-character loop on rows that actually contain
    non-ASCII characters — pure-ASCII rows skip it entirely.
    """
    s = series.fillna("").astype(str)
    non_ascii_mask = s.str.contains(NON_ASCII_RE, regex=True, na=False)
    if non_ascii_mask.any():
        s.loc[non_ascii_mask] = s.loc[non_ascii_mask].apply(
            lambda x: ''.join(
                c for c in unicodedata.normalize('NFD', x)
                if unicodedata.category(c) != 'Mn'
            )
        )
    return s.str.casefold().str.strip()


def collapse_repeated_chars(series):
    return series.str.replace(REPEATED_CHAR_RE, r'\1\1', regex=True)


def collapse_dotted_acronyms(series):
    return series.str.replace(
        DOTTED_ACRONYM_RE, lambda m: m.group(0).replace('.', ''), regex=True
    )


def vectorized_basic_clean(series):
    s = series.str.replace(LETTER_NUM_BOUNDARIES, ' ', regex=True)
    s = s.str.replace(NON_WORD_RE, ' ', regex=True)
    s = s.str.replace(MULTI_SPACE_RE, ' ', regex=True).str.strip()
    return s


def vectorized_expand_abbrev(series, pattern, mapping):
    def resolve(m):
        token = m.group(0).rstrip('.')
        if token in mapping:
            return mapping[token]
        if token.endswith('s') and token[:-1] in mapping:
            return mapping[token[:-1]] + 's'  # e.g. "apts" -> "apartments", not "apartment"
        return token
    return series.str.replace(pattern, resolve, regex=True)


# ---------------------------------------------------------------------
# 3. Transliteration
# ---------------------------------------------------------------------
def safe_transliterate(text):
    try:
        return unidecode(text) if text else text
    except Exception as e:
        logger.warning(f"Transliteration failed on a row: {e}")
        return text


def transliterate_batch(texts):
    return [safe_transliterate(t) for t in texts]


def parallel_transliterate(series, pool, batch_size=5000):
    """Uses a long-lived pool passed in by the caller (see normalize_file)."""
    texts = series.tolist()
    if not texts:
        return series

    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    results = pool.map(transliterate_batch, batches)
    flat = [item for batch in results for item in batch]
    return pd.Series(flat, index=series.index)


# ---------------------------------------------------------------------
# 4. Field normalization pipeline
# ---------------------------------------------------------------------
def normalize_field(series, abbrev_pattern, abbrev_map, pool, field_label=""):
    raw = series.fillna("").astype(str)
    series = raw.copy()

    # 1. Transliterate non-Latin rows if present
    needs_translit_mask = series.str.contains(NON_LATIN_RE, regex=True, na=False)
    n_non_latin = needs_translit_mask.sum()

    if n_non_latin > 0:
        logger.info(f"  [{field_label}] Transliterating {n_non_latin} non-Latin rows out of {len(series)}...")
        transliterated = parallel_transliterate(series[needs_translit_mask], pool)
        series.loc[needs_translit_mask] = transliterated
        # Tame unidecode's stuttering phonetic output on the rows we just transliterated
        series.loc[needs_translit_mask] = collapse_repeated_chars(series.loc[needs_translit_mask])

    # 2. Accent stripping & casefolding (ASCII-masked internally for speed)
    series = strip_accents_and_casefold(series)

    # 3. Collapse dotted acronyms (P.V.T. -> PVT) before punctuation gets stripped
    series = collapse_dotted_acronyms(series)

    # 4. Ampersand expansion
    series = series.str.replace(AMPERSAND_RE, ' and ', regex=True)

    # 5. Abbreviation expansion (handles plural + dotted forms)
    series = vectorized_expand_abbrev(series, abbrev_pattern, abbrev_map)

    # 6. Boundary separation & final non-word punctuation stripping
    cleaned = vectorized_basic_clean(series)

    # 7. Fall back to the stripped raw value if cleaning produced an empty string
    #    (e.g. names that were entirely punctuation, like "***" or "---") so that
    #    such rows don't all collapse into one indistinguishable empty string.
    empty_mask = cleaned.str.len() == 0
    n_empty = empty_mask.sum()
    if n_empty > 0:
        logger.warning(f"  [{field_label}] {n_empty} rows became empty after cleaning; falling back to raw value")
        cleaned.loc[empty_mask] = raw.loc[empty_mask].str.strip()

    return cleaned, needs_translit_mask


def normalize_country(series):
    return series.fillna("").astype(str).str.strip().str.casefold()


# ---------------------------------------------------------------------
# 5. File processing driver
# ---------------------------------------------------------------------
def normalize_file(input_path, output_path, chunksize, n_workers):
    start = time.time()
    total_rows = 0
    first_chunk = True

    logger.info(f"Starting normalization: {input_path} -> {output_path}")
    logger.info(f"chunksize={chunksize}, workers={n_workers}")

    # One pool for the whole file (not per-chunk, not per-field) — avoids the
    # overhead of repeatedly spawning/tearing down worker processes, and
    # try/finally guarantees it's closed even if something raises mid-loop.
    pool = Pool(n_workers)
    try:
        for i, chunk in enumerate(pd.read_csv(input_path, sep='\t', dtype=str, chunksize=chunksize)):
            chunk_start = time.time()

            required_cols = {'entity_id', 'business_name', 'business_address', 'country'}
            missing = required_cols - set(chunk.columns)
            if missing:
                raise ValueError(f"Missing expected columns: {missing}")

            chunk['business_name_clean'], name_translit_mask = normalize_field(
                chunk['business_name'], GENERAL_ABBREV_RE, GENERAL_ABBREV, pool,
                field_label="business_name"
            )
            chunk['business_address_clean'], addr_translit_mask = normalize_field(
                chunk['business_address'], ADDRESS_ABBREV_RE, ADDRESS_ABBREV, pool,
                field_label="business_address"
            )
            chunk['country_clean'] = normalize_country(chunk['country'])

            # Lets the matching model learn to trust name/address similarity less
            # for rows that went through lossy phonetic transliteration.
            chunk['was_transliterated'] = (name_translit_mask | addr_translit_mask).values

            chunk.to_csv(
                output_path, sep='\t',
                mode='w' if first_chunk else 'a',
                header=first_chunk, index=False
            )
            first_chunk = False
            total_rows += len(chunk)

            logger.info(
                f"Chunk {i+1}: {len(chunk)} rows in {time.time() - chunk_start:.1f}s "
                f"(total so far: {total_rows})"
            )
    finally:
        pool.close()
        pool.join()
        logger.info("Pool closed cleanly")

    logger.info(f"Done. {total_rows} rows written to {output_path} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    normalize_file(args.input_path, args.output_path, args.chunksize, args.workers)