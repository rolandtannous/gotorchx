all: mnist

mnist: main.cpp
	clang++ -std=c++14 \
	-I ${GOPATH}/src/github.com/rolandtannous/gotorchx/cgotorch/libtorch/include \
	-I ${GOPATH}/src/github.com/rolandtannous/gotorchx/cgotorch/libtorch/include/torch/csrc/api/include \
	-L ${GOPATH}/src/github.com/rolandtannous/gotorchx/cgotorch/libtorch/lib \
	main.cpp \
	-O -o mnist \
	-Wl,-rpath,${GOPATH}/src/github.com/rolandtannous/gotorchx/cgotorch/libtorch/lib \
	-pthread \
	-lc10 -ltorch -ltorch_cpu \
	-D_GLIBCXX_USE_CXX11_ABI=0

clean:
	rm mnist
