import ctypes
import os

class RustTokenizer:
    def __init__(self, lib_path, merges_path=None):
        self.lib = ctypes.CDLL(lib_path)
        
        # Define argtypes
        self.lib.tokenizer_new.restype = ctypes.c_void_p
        self.lib.tokenizer_free.argtypes = [ctypes.c_void_p]
        self.lib.tokenizer_train.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        self.lib.tokenizer_train.restype = ctypes.c_int
        self.lib.tokenizer_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.tokenizer_encode.restype = ctypes.POINTER(ctypes.c_uint)
        self.lib.tokenizer_free_ids.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_size_t]
        self.lib.tokenizer_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.tokenizer_save.restype = ctypes.c_int
        self.lib.tokenizer_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.tokenizer_load.restype = ctypes.c_int
        
        self.tokenizer_ptr = self.lib.tokenizer_new()
        
        # Vocab for decoding
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        self.next_id = 256
        
        if merges_path and os.path.exists(merges_path):
            self.load(merges_path)
            
    def __del__(self):
        if hasattr(self, 'lib') and self.tokenizer_ptr:
            self.lib.tokenizer_free(self.tokenizer_ptr)
            
    def train(self, input_path, vocab_size):
        res = self.lib.tokenizer_train(self.tokenizer_ptr, input_path.encode('utf-8'), vocab_size)
        if res != 0:
            raise Exception("Tokenizer training failed")
            
    def load(self, path):
        # Load into C++ engine
        res = self.lib.tokenizer_load(self.tokenizer_ptr, path.encode('utf-8'))
        if res != 0:
            raise Exception("Tokenizer load failed")
            
        # Load into Python for decoding
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        
        merges = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2: continue
                pair_str, id_str = parts
                p1, p2 = map(int, pair_str.split(','))
                idx = int(id_str)
                merges.append((idx, p1, p2))
        
        # Sort by ID to ensure we build bottom-up
        merges.sort(key=lambda x: x[0])
        
        for idx, p1, p2 in merges:
            b1 = self.id_to_bytes.get(p1, b"")
            b2 = self.id_to_bytes.get(p2, b"")
            self.id_to_bytes[idx] = b1 + b2

    def encode(self, text):
        out_len = ctypes.c_size_t(0)
        ids_ptr = self.lib.tokenizer_encode(self.tokenizer_ptr, text.encode('utf-8'), ctypes.byref(out_len))
        if not ids_ptr:
            return []
        ids = [ids_ptr[i] for i in range(out_len.value)]
        self.lib.tokenizer_free_ids(ids_ptr, out_len)
        return ids
        
    def save(self, output_path):
        res = self.lib.tokenizer_save(self.tokenizer_ptr, output_path.encode('utf-8'))
        if res != 0:
            raise Exception("Tokenizer save failed")

    def decode(self, ids):
        out_bytes = b""
        for idx in ids:
            out_bytes += self.id_to_bytes.get(idx, b"")
        return out_bytes.decode('utf-8', errors='replace')

def get_rust_tokenizer(merges_path=None):
    # Adjust path to find lib relative to this file or current dir
    # Try current dir first, then ../cuda_kernels/
    base = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(base, "../cuda_kernels/librustbpe.so")
    
    if not os.path.exists(lib_path):
        # Fallback to experiment path if running from weird location
        lib_path = "/home/jayantlohia16/experiment/mamba-integer/src/cuda_kernels/librustbpe.so"
        
    return RustTokenizer(lib_path, merges_path)