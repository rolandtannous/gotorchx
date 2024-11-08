all : libcgotorch.so

srcs := $(wildcard *.cc)
hdrs := $(wildcard *.h)
objs := $(srcs:%.cc=%.o)

%.o: %.cc
	$(CXX) -std=c++17 \
	-I .. -I $(LIBTORCH_DIR)/include \
	-I $(LIBTORCH_DIR)/include/torch/csrc/api/include \
	-fPIC  \
	-D_GLIBCXX_USE_CXX11_ABI=$(GLIBCXX_USE_CXX11_ABI) \
	$(CUDA_FLAGS) \
	-O -c $< -o $@

libcgotorch.so: $(objs) ${LIBTORCH_DIR}
	${CXX} 	-L libtorch/lib \
	$(objs) \
	-shared \
	-o $@ ${INSTALL_NAME} \
	-Wl,-rpath,libtorch/lib \
	-Wl,-${LOAD} libtorch/lib/libc10.${LIB_SUFFIX} \
	-lc10 -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda

clean:
	rm -rf *.so *.o
