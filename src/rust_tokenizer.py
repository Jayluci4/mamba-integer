
import ctypes
import os

class RustTokenizer:
    def __init__(self, lib_path):
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
        
    def __del__(self):
        if hasattr(self, 'lib') and self.tokenizer_ptr:
            self.lib.tokenizer_free(self.tokenizer_ptr)
            
    def train(self, input_path, vocab_size):
        res = self.lib.tokenizer_train(self.tokenizer_ptr, input_path.encode('utf-8'), vocab_size)
        if res != 0:
            raise Exception("Tokenizer training failed")
            
    def load(self, path):
        res = self.lib.tokenizer_load(self.tokenizer_ptr, path.encode('utf-8'))
        if res != 0:
            raise Exception("Tokenizer load failed")

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
        # We don't have a Rust decode yet in this minimal version.
        # For training, encode is enough. For debug, we can use a dummy.
        return f"[BPE IDs: {ids}]"

def get_rust_tokenizer():
    lib_path = "/home/jayantlohia16/experiment/nanochat/rustbpe/target/release/librustbpe.so"
    return RustTokenizer(lib_path)
