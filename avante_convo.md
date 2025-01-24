wrap native functions


native global functions are at
`libtorch/include/ATen/Functions.h`


With Cgo, Go programs can refer to C symbols with their names prefixed by `C.`.


We put all C wrappers of native functions in the subdirectory `cgotorch`.  In
`cgotorch/cgotorch.h`, we can see the wrapper of `at::Tensor` and `at::mm`.

The implementation of the C wrapper `MM` is in C++ and in `cgotorch/torch.cc`.

In package `gotorch`, we define the Go wrappers of native functions, which
operates the Go type `Tensor` defined in `tensor.go`.

The implementation of the C wrapper `MM` is in C++ and in `cgotorch/torch.cc`.

```cpp
const char *MM(Tensor a, Tensor b, Tensor *result) {
  try {
    at::Tensor c = at::mm(*a, *b);
    *result = new at::Tensor(c);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}
```

Given the Go type `Tensor`, the Go wrapper `MM` is as follows.

```go
package gotorch

func MM(a, b Tensor) Tensor {
    var t C.Tensor
    MustNil(unsafe.Pointer(C.MM(C.Tensor(*a.T), C.Tensor(*b.T), &t)))
    SetTensorFinalizer((*unsafe.Pointer)(&t))
    return Tensor{(*unsafe.Pointer)(&t)}
}
```

