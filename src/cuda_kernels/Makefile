NVCC = nvcc
# Targeting A100 (80). NVCC 11.5 supports up to 86/80.
NVCC_FLAGS = -O3 --compiler-options '-fPIC' -shared -gencode arch=compute_80,code=sm_80

all: libdyadic_mamba.so libbitshift_norm.so libsquareplus.so

libdyadic_mamba.so: dyadic_mamba_kernel.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

libbitshift_norm.so: bitshift_norm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

libsquareplus.so: squareplus_kernel.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f *.so
