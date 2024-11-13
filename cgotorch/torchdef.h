/* Copyright 2020, GoTorch Authors */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <vector>
extern "C" {
typedef at::Tensor *Tensor;
typedef torch::optim::Optimizer *Optimizer;
typedef torch::data::datasets::MNIST *MNIST;
typedef torch::data::transforms::Normalize<> *Normalize;
typedef torch::Device *Device;
typedef std::vector<char> *ByteBuffer;  // NOLINT
typedef at::Dimname *Dimname;
#else
typedef void *Tensor;
typedef void *Optimizer;
typedef void *MNIST;
typedef void *Normalize;
typedef void *Device;
typedef void *ByteBuffer;
typedef void *Dimname;
#endif
typedef void *CUDAStream;

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

const char *exception_str(const char *e);
#ifdef __cplusplus
}
#endif
