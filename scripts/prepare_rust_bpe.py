from datasets import load_dataset
from rust_tokenizer import get_rust_tokenizer
import os

def prepare():
    corpus_path = "tinystories_corpus.txt"
    vocab_size = 4096
    
    print("Loading TinyStories data...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    # Dump 10k samples to text file for Rust training
    if not os.path.exists(corpus_path):
        print(f"Dumping corpus to {corpus_path}...")
        with open(corpus_path, "w") as f:
            for i, item in enumerate(dataset):
                f.write(item['text'] + "\n")
                if i >= 10000:
                    break
    
    # Train
    print("Training Rust BPE Tokenizer...")
    t = get_rust_tokenizer()
    t.train(corpus_path, vocab_size)
    
    # Save
    t.save("rust_bpe_merges.txt")
    print("Success: rust_bpe_merges.txt created.")
    
    # Test
    test_str = "Once upon a time, Lily was a happy girl."
    ids = t.encode(test_str)
    print(f"Test Encode: '{test_str}' -> {ids}")

if __name__ == "__main__":
    prepare()
