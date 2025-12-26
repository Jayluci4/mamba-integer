NVCC = nvcc
# Targeting A100 (80) and Ada/Hopper via forward-compat (86).
# NVCC 11.5 supports up to sm_86. L4 (sm_89) runs sm_86 code via JIT.
NVCC_FLAGS = -O3 --compiler-options '-fPIC' -shared \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_86,code=compute_86 \
    --use_fast_math

all: libdyadic_mamba.so libbitshift_norm.so libsquareplus.so libwarp_scan.so

libdyadic_mamba.so: dyadic_mamba_kernel.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

libbitshift_norm.so: bitshift_norm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

libsquareplus.so: squareplus_kernel.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

libwarp_scan.so: warp_scan_kernel.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f *.so
