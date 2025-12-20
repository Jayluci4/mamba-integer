
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from surgery_dyadic import perform_dyadic_surgery
import math

def calculate_perplexity(model, tokenizer, text, device):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    return math.exp(loss.item())

def run_test():
    print("--- TinyLlama Dyadic Surgery Inference Test ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    
    test_prompt = "The capital of France is"
    test_text = "The capital of France is Paris, a major European city and a global center for art, fashion, gastronomy and culture."
    
    # 1. Baseline
    print("\n[Baseline] Generating...")
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    print(f"Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")
    
    ppl_base = calculate_perplexity(model, tokenizer, test_text, device)
    print(f"Baseline Perplexity: {ppl_base:.4f}")
    
    # 2. Surgery
    print("\n[Surgery] Applying Dyadic-Cayley Transform...")
    perform_dyadic_surgery(model)
    model = model.to(device) # Ensure everything moved correctly
    
    # 3. Post-Surgery
    print("\n[Dyadic] Generating...")
    try:
        out_dyadic = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        print(f"Output: {tokenizer.decode(out_dyadic[0], skip_special_tokens=True)}")
    except Exception as e:
        print(f"Generation Failed: {e}")
        
    ppl_dyadic = calculate_perplexity(model, tokenizer, test_text, device)
    print(f"Dyadic Perplexity: {ppl_dyadic:.4f}")
    
    # Verdict
    diff = ppl_dyadic - ppl_base
    ratio = ppl_dyadic / ppl_base
    print(f"\nDegradation: +{diff:.4f} (Ratio: {ratio:.2f}x)")
    
    if ratio < 1.5:
        print("Verdict: SCENARIO A (Coherent / Slight degradation). Holy Grail territory.")
    elif ratio < 5.0:
        print("Verdict: SCENARIO B (Needs Healing). Recoverable.")
    else:
        print("Verdict: SCENARIO C (Broken). Needs major retraining.")

if __name__ == "__main__":
    run_test()
