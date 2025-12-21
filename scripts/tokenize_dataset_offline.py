
import numpy as np
from datasets import load_dataset
import os
import sys
import time

# Add src to path for rust_tokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
from rust_tokenizer import get_rust_tokenizer

def tokenize_offline():
    print("--- Offline Tokenization: TinyStories ---")
    
    # 1. Setup Tokenizer
    tokenizer = get_rust_tokenizer()
    merges_path = os.path.join(os.path.dirname(__file__), "../configs/rust_bpe_merges.txt")
    if os.path.exists(merges_path):
        tokenizer.load(merges_path)
        print(f"Loaded merges from {merges_path}")
    else:
        print("Error: Merges file not found.")
        return

    # 2. Load Dataset
    print("Loading TinyStories...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    # 3. Process
    output_bin = "tinystories_train.bin"
    print(f"Writing to {output_bin}...")
    
    # We will write uint16 (vocab 4096 fits in 16 bits)
    all_tokens = []
    count = 0
    total_tokens = 0
    start_time = time.time()
    
    # Open file for appending binary
    with open(output_bin, "wb") as f:
        for item in dataset:
            text = item['text']
            # Prepend BOS (optional, but standard)
            # ids = [tokenizer.bos_id] + tokenizer.encode(text)
            ids = tokenizer.encode(text)
            
            # Add a document separator token if needed (e.g. 0)
            ids.append(0) 
            
            arr = np.array(ids, dtype=np.uint16)
            f.write(arr.tobytes())
            
            count += 1
            total_tokens += len(ids)
            
            if count % 1000 == 0:
                elapsed = time.time() - start_time
                tps = total_tokens / elapsed
                print(f"Processed {count} docs | {total_tokens:,} tokens | {tps:.0f} tok/sec")
            
            if count >= 200000: # Increased for production training chunk
                break
                
    print(f"\nSuccess! Total Tokens: {total_tokens:,}")
    print(f"File size: {os.path.getsize(output_bin) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    tokenize_offline()
