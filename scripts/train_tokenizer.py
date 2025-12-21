
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from datasets import load_dataset
import os

def train_tokenizer():
    print("--- Training Micro-BPE Tokenizer (Vocab 4096) ---")
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=4096, 
        special_tokens=["<|endoftext|>", "<|padding|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    def batch_iterator(batch_size=1000):
        for i, item in enumerate(dataset):
            yield item['text']
            if i > 5000: 
                break
                
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    save_path = os.path.join(os.path.dirname(__file__), "../configs/micro_bpe.json")
    tokenizer.save(save_path)
    print(f"Saved tokenizer to {save_path}")

if __name__ == "__main__":
    train_tokenizer()
