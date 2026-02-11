"""
Retrain Rust BPE tokenizer on domain-weighted corpus for ZK-ML.

Old tokenizer: 10K TinyStories samples, vocab 4096
New tokenizer: 200K domain-weighted samples, vocab 16384

Domain weights match the ZK-ML use cases:
  40% FineWeb-Edu      (general English, educational quality)
  15% Solidity/DISL    (smart contracts - #1 ZK-ML use case)
  10% Python/Rust code (programming syntax)
  10% OpenWebMath      (mathematical notation, LaTeX)
  10% SEC filings      (financial text, regulatory compliance)
  10% SQL/structured   (text-to-SQL, JSON parsing)
   5% Cosmopedia v2    (synthetic textbooks, structured reasoning)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from datasets import load_dataset
from rust_tokenizer import get_rust_tokenizer

HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Total samples to collect (more = better tokenizer, but slower training)
TOTAL_SAMPLES = 200_000

# Domain weights for tokenizer training corpus
SOURCES = [
    {
        "name": "fineweb_edu",
        "path": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "text_field": "text",
        "weight": 0.40,
    },
    {
        "name": "solidity_disl",
        "path": "ASSERT-KTH/DISL",
        "config": "decomposed",
        "text_field": "source_code",
        "weight": 0.15,
    },
    {
        "name": "tiny_codes",
        "path": "nampdn-ai/tiny-codes",
        "config": None,
        "text_field": "response",
        "weight": 0.10,
    },
    {
        "name": "openwebmath",
        "path": "open-web-math/open-web-math",
        "config": None,
        "text_field": "text",
        "weight": 0.10,
    },
    {
        "name": "sec_filings",
        "path": "PleIAs/SEC",
        "config": None,
        "text_field": "text",
        "weight": 0.10,
    },
    {
        "name": "text_to_sql",
        "path": "gretelai/synthetic_text_to_sql",
        "config": None,
        "text_field": "sql",
        "weight": 0.10,
    },
    {
        "name": "cosmopedia",
        "path": "HuggingFaceTB/cosmopedia-v2",
        "config": None,
        "text_field": "text",
        "weight": 0.05,
    },
]


def collect_corpus(corpus_path):
    """Collect domain-weighted corpus from HuggingFace datasets."""
    if os.path.exists(corpus_path):
        line_count = sum(1 for _ in open(corpus_path))
        if line_count >= TOTAL_SAMPLES * 0.9:
            print(f"Corpus already exists ({line_count:,} lines). Skipping download.")
            return
        else:
            print(f"Corpus exists but only {line_count:,} lines. Re-downloading...")

    print(f"Collecting {TOTAL_SAMPLES:,} domain-weighted samples...")

    with open(corpus_path, "w") as f:
        for source in SOURCES:
            n_samples = int(TOTAL_SAMPLES * source["weight"])
            name = source["name"]
            print(f"\n  [{name}] Collecting {n_samples:,} samples ({source['weight']*100:.0f}%)...")

            try:
                kwargs = {
                    "path": source["path"],
                    "split": "train",
                    "streaming": True,
                }
                if source["config"]:
                    kwargs["name"] = source["config"]
                if HF_TOKEN:
                    kwargs["token"] = HF_TOKEN

                ds = load_dataset(**kwargs)
                count = 0
                for item in ds:
                    text = item.get(source["text_field"], "")
                    if not text or len(text) < 20:
                        continue

                    # For SQL dataset, combine multiple fields
                    if name == "text_to_sql":
                        sql_context = item.get("sql_context", "")
                        sql_prompt = item.get("sql_prompt", "")
                        sql_query = item.get("sql", "")
                        text = f"{sql_context}\n-- {sql_prompt}\n{sql_query}"

                    # Truncate very long documents (tokenizer doesn't need full docs)
                    if len(text) > 4000:
                        text = text[:4000]

                    # Clean newlines for line-based corpus
                    text = text.replace("\n", " \\n ")
                    f.write(text + "\n")
                    count += 1

                    if count >= n_samples:
                        break

                print(f"    Collected {count:,}/{n_samples:,} samples")

            except Exception as e:
                print(f"    WARNING: Failed to load {name}: {e}")
                # Fill with FineWeb-Edu as fallback
                print(f"    Falling back to FineWeb-Edu for {n_samples:,} samples...")
                try:
                    ds = load_dataset(
                        "HuggingFaceFW/fineweb-edu",
                        name="sample-10BT",
                        split="train",
                        streaming=True,
                    )
                    count = 0
                    for item in ds:
                        text = item.get("text", "")
                        if text and len(text) >= 20:
                            if len(text) > 4000:
                                text = text[:4000]
                            text = text.replace("\n", " \\n ")
                            f.write(text + "\n")
                            count += 1
                            if count >= n_samples:
                                break
                    print(f"    Fallback: collected {count:,} samples")
                except Exception as e2:
                    print(f"    FALLBACK ALSO FAILED: {e2}")

    final_lines = sum(1 for _ in open(corpus_path))
    print(f"\nCorpus ready: {final_lines:,} lines saved to {corpus_path}")


def train_tokenizer(corpus_path, vocab_size, output_path):
    """Train Rust BPE tokenizer on the corpus."""
    print(f"\nTraining Rust BPE Tokenizer (vocab_size={vocab_size})...")
    print(f"  This will learn {vocab_size - 256} merges from the corpus.")

    t = get_rust_tokenizer()
    t.train(corpus_path, vocab_size)
    t.save(output_path)

    print(f"  Merges saved to: {output_path}")

    # Verify
    t2 = get_rust_tokenizer()
    t2.load(output_path)

    test_strings = [
        "function transfer(address to, uint256 amount) public returns (bool)",
        "SELECT COUNT(*) FROM users WHERE active = 1 GROUP BY department",
        "The quick brown fox jumps over the lazy dog.",
        "\\int_{0}^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}",
        "Revenue for Q3 2025 increased 15% year-over-year to $4.2 billion.",
        "def fibonacci(n: int) -> int:\\n    if n <= 1: return n\\n    return fibonacci(n-1) + fibonacci(n-2)",
        "pragma solidity ^0.8.0; contract Token { mapping(address => uint256) balances; }",
    ]

    print(f"\n  Verification (vocab={vocab_size}):")
    for s in test_strings:
        ids = t2.encode(s)
        decoded = t2.decode(ids)
        ratio = len(s.encode("utf-8")) / len(ids) if ids else 0
        print(f"    [{len(ids):3d} tokens, {ratio:.1f} bytes/tok] {s[:60]}...")


def main():
    global TOTAL_SAMPLES
    import argparse

    parser = argparse.ArgumentParser(description="Train Rust BPE tokenizer")
    parser.add_argument("--vocab-size", type=int, default=16384, help="Vocabulary size")
    parser.add_argument(
        "--corpus",
        default=os.path.join(os.path.dirname(__file__), "domain_corpus.txt"),
        help="Corpus path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output merges path (default: configs/rust_bpe_merges_{vocab_size}.txt)",
    )
    parser.add_argument("--samples", type=int, default=TOTAL_SAMPLES, help="Total samples")
    args = parser.parse_args()

    TOTAL_SAMPLES = args.samples

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(__file__),
            f"../configs/rust_bpe_merges_{args.vocab_size}.txt",
        )

    # Step 1: Collect corpus
    collect_corpus(args.corpus)

    # Step 2: Train tokenizer
    train_tokenizer(args.corpus, args.vocab_size, args.output)

    print(f"\nDone! Use --vocab-size to change vocab size.")
    print(f"  Merges: {args.output}")
    print(f"  Update config: \"vocab_size\": {args.vocab_size}")


if __name__ == "__main__":
    main()
