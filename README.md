# GoTorchX


GoTorchX implements PyTorch high-level APIs, including modules and functionals, in idiomatic Go. This enables deep learning programming in Go and Go+. We are following the original design and philosophy of the gotorch package while exploring architectural changes that could improve Go's performance with PyTorch. This project is the revival and continued development of the original [GoTorch project](https://github.com/wangkuiyi/gotorch).

## Attribution

This project is based on gotorch by Yi Wang.
Copyright 2020 Yi Wang.
Modified and extended by Roland Tannous and [GravityQ](https://gravityq.ai) in 2024.

Please refer to the LICENSE and NOTICE documents for detailed copyright and licensing information.

## Recent Developments

Since reviving the project in 2024, we've made significant improvements:

    - Upgraded to libtorch 2.5.1 with dynamic CPU/GPU package selection and CUDA compatibility for versions 11.8, 12.1, 12.4, and 12.5
    - Implemented advanced attention mechanisms including MultiHeadAttention and Flash Attention with comprehensive CUDA support
    - Added fundamental mathematical operations including Power (Pow), Square root (Sqrt), Natural logarithm (Log), and Exponential (Exp)
    - Enhanced tensor shape manipulation with Unsqueeze, Reshape, and Cat (Concatenate) operations
    - Added named tensor dimensions for more semantic clarity in tensor operations
    - Implemented comprehensive reduction operations including Max, Min, and Prod with various modes

#### Current Status & Roadmap

    Core Infrastructure

    ✅ Basic tensor operations and management
    ✅ CPU/GPU device support
    ✅ Memory management and GC integration
    ✅ Error handling system for C++/Go interop
    ✅ Basic type system aligned with libtorch

    Tensor Operations Current coverage:

    ✅ Basic arithmetic operations (add, subtract, multiply, etc.)
    ✅ Matrix operations (mm, matmul)
    ✅ Shape manipulation (reshape, view)
    ✅ Basic indexing and slicing
    ❓ Advanced indexing capabilities (need verification)
    ❓ Complex number support (need verification)
    ❌ Sparse tensor operations (appears missing)

    Neural Network Components Modules implemented:

    ✅ Linear layers
    ✅ Convolution layers
    ✅ BatchNorm
    ✅ Basic containers (Sequential)
    ❌ RNN/LSTM/GRU layers (appears missing)
    ❌ Transformer layers (being added in recent PR)
    ❌ Embedding layers (appears missing)

    Optimizers

    ✅ Basic optimizers (SGD)
    ❓ Advanced optimizers status unclear (Adam, RMSprop, etc.)
    ❌ Learning rate schedulers (appears missing)

    Loss Functions

    ❓ Status of common loss functions unclear
    ❓ Custom loss function support unclear

    Data Loading & Processing

    ✅ Basic image loading
    ✅ Common transforms
    ❌ Dataset abstractions (appears limited)
    ❌ DataLoader parallelization (appears missing)

    Serialization & Model Management

    ✅ Basic model saving/loading
    ❓ Checkpoint management (need verification)
    ❓ State dict handling (need verification)

## Easy Switch (Adopted from original package README)

Writing deep learning systems in Go is as efficiently as in Python. The DCGAN training programs in GoTorchX and PyTorch call similar APIs, have similar program structure, and have a similar number of lines. Go+ has a syntax similar to Python. The Go+ compiler translates Go+ programs into Go source programs. It is a joy to write Go+ programs that calls Go packages like GoTorchX.

We have a plan of a translator that migrates existing PyTorch models in Python into GoTorchX.

## Benefits (Adopted from original package README)

    Higher runtime efficiency. Go programs run as efficiently as C++.
    Training and prediction in the same language. No longer training in Python and online prediction in C++. All in Go/Go+. No TensorFlow graphs or PyTorch tracing.
    Same data processing code for training and prediction. No need to Wrap OpenCV functions into TensorFlow operators in C++ for prediction and Python for training.
    Supports many machine learning paradigms., including adversarial, reinforcement, and imitation learning -- those we cannot split into training and prediction.
    Same program for edge and cloud. GoTorchX programs compile and run on phones and self-driving cars as they do on servers and desktops.

## Technical Considerations and Bottlenecks
Updated Assessment of Gotorch Challenges in 2024

Examining the key issues mentioned in the original abandonment of gotorch, here's how the situation has evolved:
**1. CGO Performance**

Original Issue: The project was initially abandoned partly due to cgo performance concerns, with the original author noting that Go's team wasn't prioritizing cgo performance improvements.

2024 Status: Largely Resolved

    cgo performance has improved approximately 17x since 2015
    Current overhead is only ~40ns per call (single-core) or ~4ns (multi-core)
    Scales effectively up to 16 cores, enabling 250 million ops/second
    The Go team has indeed improved cgo performance significantly despite earlier indications

This performance level is now adequate for most ML workloads and no longer represents a significant bottleneck for projects like gotorch.

**2. GPU Memory Management**

Original Issue: Coordination between Go's garbage collector and GPU memory release was identified as a critical challenge, particularly for training workloads.

2024 Status: Partially Improved

    Go's GC has added memory limits (GOMEMLIMIT) for better resource management
    GC latency has improved with shorter pauses and better concurrency
    The fundamental challenge remains: Go's GC still isn't designed to coordinate with external resource management systems like CUDA
    No native mechanism exists for GPU memory coordination with Go's GC cycles

This remains the most significant technical challenge for a project like gotorch, particularly for training workloads where timely GPU memory release is critical.

**3. GC Pauses and Inference Performance**

Original Issue: Concerns were raised about Go's garbage collector causing irregular pauses that could impact online inference performance.

2024 Status: Significantly Improved

    Go's GC has made substantial strides in latency predictability
    More sophisticated GC pacing and cycle management
    Better tools for profiling and tuning GC behavior
    Explicit memory limits help prevent unexpected OOM conditions

While the GC still causes some pauses, they're shorter and more predictable than in 2020, making Go more viable for latency-sensitive inference workloads.

In summary, the cgo performance issues have been largely resolved, GC predictability for inference has significantly improved, but coordinating GPU memory with Go's GC remains a substantial challenge that would need to be addressed for any revival of the gotorch project.

We are actively investigating a potential re-architecture of the package to at least partially release it from over-reliance on cgo, which would address some of these remaining concerns.

## The Tech Stack (Adopted from original package README)

GoTorchX works with the following open-source communities to form Go+Torch.

    the Go+ community,
    the PyTorch community, and
    the TensorFlow XLA ecosystem.

The following figure reveals the stack of technologies.

text

Go+ applications   # users write DL applications in Go+,
     │             # whose syntax is as concise as Python
 [Go+ compiler]
     ↓
Go source code -→ GoTorchX -→ libtorch -→ pytorch/xla -→ XLA ops
     │
 [Go compiler]
     ↓
executable binary  # x86_64, ARM, CUDA, TPU
                   # Linux, macOS, Android, iOS


## Documentation ((Adopted from original package README))

- [Build and Test](CONTRIBUTING.md)
- [Design Docs](./doc/design.md) [(汉语)](./doc/design_cn.md)
- How To
  - [Wrap PyTorch Native Functions](./doc/wrap_native_functions.md) [(汉语)](./doc/wrap_native_functions_cn.md)
  - [Port Functionals and Modules](doc/develop_functionals_and_modules.md) [(汉语)](./doc/develop_functionals_and_modules_cn.md)
  - [Shuffle Images in Tarballs](doc/shuffle_tarball.md) [(汉语)](./doc/shuffle_tarball_cn.md)
- Examples
  - [MNIST Training and Prediction on Raspberry Pi](./example/mnist)
    - [Demo on Youtube](https://www.youtube.com/watch?v=izpeb_FugII&t=5s)
  - [Adversarial Learning](./example/dcgan)
  - [Train ResNet50 Using ImageNet](./example/resnet)
