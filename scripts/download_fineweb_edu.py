#!/usr/bin/env python3
"""Download and tokenize FineWeb-Edu dataset.

Downloads ~5B tokens from FineWeb-Edu and tokenizes with Rust BPE.
Output: fineweb_edu_train.bin (binary uint16 tokens)
"""

import os
import sys
import numpy as np
from datasets import load_dataset
import time

# Add src to path for tokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
from rust_tokenizer import get_rust_tokenizer

# Config
TARGET_TOKENS = 5_000_000_000  # 5B tokens
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "fineweb_edu_train.bin")
MERGES_PATH = os.path.join(os.path.dirname(__file__), "../configs/rust_bpe_merges.txt")
BUFFER_SIZE = 100_000_000  # Flush every 100M tokens

def main():
    print("--- FineWeb-Edu Download & Tokenization ---")
    print(f"Target: {TARGET_TOKENS:,} tokens")
    print(f"Output: {OUTPUT_PATH}")

    # Load tokenizer
    tokenizer = get_rust_tokenizer()
    if os.path.exists(MERGES_PATH):
        tokenizer.load(MERGES_PATH)
        print(f"Loaded tokenizer from {MERGES_PATH}")
    else:
        print(f"ERROR: Merges file not found at {MERGES_PATH}")
        return

    # Stream FineWeb-Edu (sample-10BT subset for faster download)
    print("Loading FineWeb-Edu dataset (streaming)...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",  # 10B token sample (smaller download)
        split="train",
        streaming=True
    )

    # Tokenize and write
    total_tokens = 0
    docs_processed = 0
    token_buffer = []
    start_time = time.time()

    # Open file for writing
    with open(OUTPUT_PATH, 'wb') as f:
        for doc in dataset:
            text = doc['text']
            tokens = tokenizer.encode(text)
            token_buffer.extend(tokens)
            docs_processed += 1

            # Flush buffer periodically
            if len(token_buffer) >= BUFFER_SIZE:
                arr = np.array(token_buffer, dtype=np.uint16)
                arr.tofile(f)
                total_tokens += len(token_buffer)
                token_buffer = []

                elapsed = time.time() - start_time
                rate = total_tokens / elapsed
                eta = (TARGET_TOKENS - total_tokens) / rate if rate > 0 else 0
                print(f"Progress: {total_tokens/1e9:.2f}B / {TARGET_TOKENS/1e9:.1f}B tokens | "
                      f"{docs_processed:,} docs | {rate/1e6:.2f}M tok/s | ETA: {eta/60:.1f} min")

            # Check if we've hit target
            if total_tokens + len(token_buffer) >= TARGET_TOKENS:
                # Write remaining and trim to exact target
                remaining = TARGET_TOKENS - total_tokens
                arr = np.array(token_buffer[:remaining], dtype=np.uint16)
                arr.tofile(f)
                total_tokens += remaining
                break

    # Final stats
    elapsed = time.time() - start_time
    file_size = os.path.getsize(OUTPUT_PATH) / (1024**3)
    print(f"\n=== Complete ===")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Documents: {docs_processed:,}")
    print(f"File size: {file_size:.2f} GB")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Output: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
