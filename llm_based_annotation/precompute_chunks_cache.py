"""
Precompute and cache sentence chunks for all HTML files across val, test, and train splits.
Run this once to avoid reloading the spaCy model during annotation.

Usage:
    python precompute_chunks_cache.py
    python precompute_chunks_cache.py --splits val test  # only specific splits
    python precompute_chunks_cache.py --force             # recompute even if cache exists
"""

import os
import json
import argparse
import spacy
from pathlib import Path

from utils_extraction import (
    extract_body, tokenize, clean_tokens, decode,
    flatten_token_chunks, merge_sentences_with_heuristics_tokens,
    merge_tokens_general,
)

project_root = r"C:\Users\zakga\OneDrive\Documents\code\LeREaD_annotation_process"

CACHE_DIR = os.path.join(project_root, "chunk_cache", "sentence")
MIN_TOKENS = 500
CITATION_THRESHOLD = 25
SPLITS = ["val", "test", "train"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_cache_path(split: str, filename: str) -> str:
    return os.path.join(CACHE_DIR, split, f"{filename}.json")


def cache_exists(split: str, filename: str) -> bool:
    return os.path.isfile(get_cache_path(split, filename))


def save_cache(split: str, filename: str, token_chunks: list[list[str]]) -> None:
    path = get_cache_path(split, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(token_chunks, f)


def load_cache(split: str, filename: str) -> list[list[str]]:
    with open(get_cache_path(split, filename), "r", encoding="utf-8") as f:
        return json.load(f)


def get_all_html_files_in(folder_path: str) -> dict[str, str]:
    """Returns {filename_without_extension: html_content}."""
    files = {}
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if not os.path.isfile(full_path):
            continue
        name, ext = os.path.splitext(entry)
        if ext.lower() not in {".html", ".htm"}:
            continue
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                files[name] = f.read()
        except UnicodeDecodeError:
            with open(full_path, "r", encoding="latin-1") as f:
                files[name] = f.read()
    return files


# ---------------------------------------------------------------------------
# Core chunking logic (extracted so main.py can import it too)
# ---------------------------------------------------------------------------

def compute_sentence_chunks(html_content: str, nlp, min_tokens: int = MIN_TOKENS) -> list[list[str]]:
    """Compute sentence-based token chunks for a single HTML document."""
    body_content = extract_body(html_content)
    tokens = tokenize(body_content)
    normalized_cleaned_tokens = clean_tokens(
        html_tokens=tokens, normalize=True,
        keep_manual_label=True, keep_bookmarks=True,
    )

    doc = nlp(decode(normalized_cleaned_tokens))
    initial_sentences = [sent.text for sent in doc.sents]
    initial_sentences_token = [tokenize(sent) for sent in initial_sentences]

    from utils_extraction import flatten_token_chunks
    flat_initial_sentences = flatten_token_chunks(initial_sentences_token, separator="<sep>")

    is_sep_tag = lambda token: token == "<sep>"

    corrected_initial_sentences = merge_tokens_general(
        original_tokens=normalized_cleaned_tokens,
        derived_tokens=flat_initial_sentences,
        is_protected_func=is_sep_tag,
        log=False,
    )

    flat_token_sentence_chunks = merge_sentences_with_heuristics_tokens(
        corrected_initial_sentences,
        citation_threshold=CITATION_THRESHOLD,
        min_tokens=min_tokens,
    )

    # Split on <sep> into individual chunks
    token_chunks = []
    current_chunk = []
    for token in flat_token_sentence_chunks:
        if token != "<sep>":
            current_chunk.append(token)
        else:
            token_chunks.append(current_chunk)
            current_chunk = []
    token_chunks.append(current_chunk)

    assert flatten_token_chunks(token_chunks) == normalized_cleaned_tokens, \
        "Chunk reassembly mismatch — please check the chunking logic."

    return token_chunks


# ---------------------------------------------------------------------------
# Precomputation entry point
# ---------------------------------------------------------------------------

def precompute(splits: list[str], force: bool = False) -> None:
    # Collect all files that still need processing before loading the model
    pending: list[tuple[str, str, str]] = []  # (split, filename, html_content)

    for split in splits:
        source_dir = os.path.join(project_root, "data", "final", "Original", split)
        if not os.path.isdir(source_dir):
            print(f"⚠  Split directory not found, skipping: {source_dir}")
            continue

        files = get_all_html_files_in(source_dir)
        print(f"[{split}] Found {len(files)} HTML files in {source_dir}")

        for filename, html_content in files.items():
            if not force and cache_exists(split, filename):
                print(f"  ✓ [{split}] {filename} — cache hit, skipping")
                continue
            pending.append((split, filename, html_content))

    if not pending:
        print("\n✅ All files already cached. Nothing to do.")
        return

    # Load the heavy model only when there is actual work to do
    print(f"\n🔄 Loading spaCy model (needed for {len(pending)} file(s))…")
    nlp = spacy.load("en_core_web_trf")
    print("✅ Model loaded.\n")

    ok = err = 0
    for split, filename, html_content in pending:
        print(f"  → [{split}] {filename}", end=" … ", flush=True)
        try:
            token_chunks = compute_sentence_chunks(html_content, nlp, min_tokens=MIN_TOKENS)
            save_cache(split, filename, token_chunks)
            print(f"✓  ({len(token_chunks)} chunks)")
            ok += 1
        except Exception as exc:
            print(f"✗  ERROR: {exc}")
            err += 1

    print(f"\n{'='*50}")
    print(f"Done — {ok} cached, {err} failed, splits: {splits}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Precompute sentence chunk cache.")
    parser.add_argument(
        "--splits", nargs="+", default=SPLITS,
        choices=SPLITS, help="Splits to process (default: all)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute and overwrite existing cache files"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    precompute(splits=args.splits, force=args.force)