NVCC = nvcc
# Targeting L4 (89), A100 (80), H100 (90)
NVCC_FLAGS = -O3 --compiler-options '-fPIC' -shared -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90

all: libdyadic_mamba.so libbitshift_norm.so libsquareplus.so

libdyadic_mamba.so: dyadic_mamba_kernel.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

libbitshift_norm.so: bitshift_norm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

libsquareplus.so: squareplus_kernel.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f *.so
