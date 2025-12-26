NVCC = nvcc
NVCC_FLAGS = -O3 --compiler-options '-fPIC' -shared \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_80,code=compute_80 \
    --use_fast_math

all: libdyadic_mamba.so libbitshift_norm.so libsquareplus.so

libdyadic_mamba.so: dyadic_mamba_kernel.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

libbitshift_norm.so: bitshift_norm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

libsquareplus.so: squareplus_kernel.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Optional: warp scan kernel if exists
ifneq (,$(wildcard warp_scan_kernel.cu))
all: libwarp_scan.so
libwarp_scan.so: warp_scan_kernel.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@
endif

clean:
	rm -f *.so
