/* Copyright 2020, GoTorch Authors */

#pragma once

#include "cgotorch/torchdef.h"
#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////////////////////////////////////
// Tensor construction and operations, torch
////////////////////////////////////////////////////////////////////////////////

// torch.randn
const char *RandN(int64_t *size, int64_t length, int64_t require_grad,
                  Tensor *result);
// torch.rand
const char *Rand(int64_t *size, int64_t length, int64_t require_grad,
                 Tensor *result);
// torch.empty
const char *Empty(int64_t *size, int64_t length, int64_t require_grad,
                  Tensor *result);
// torch.ones
const char *Ones(int64_t *size, int64_t length, int64_t require_grad,
                 Tensor *result);
// torch.eye
const char *Eye(int64_t n, int64_t m, int64_t require_grad, Tensor *result);
// torch.full, only for float32
const char *Full(int64_t *size, int64_t length, float value,
                 int64_t require_grad, Tensor *result);
// torch.arange
const char *Arange(float start, float end, float step, int64_t require_grad,
                   Tensor *result);
// torch.linspace
const char *Linspace(float start, float end, int64_t steps,
                     int64_t require_grad, Tensor *result);
// torch.logspace
const char *Logspace(float start, float end, int64_t steps, double base,
                     int64_t require_grad, Tensor *result);

const char *Equal(Tensor a, Tensor b, int64_t *result);

const char *MM(Tensor a, Tensor b, Tensor *result);
const char *Sum(Tensor a, Tensor *result);
const char *SumByDim(Tensor a, int64_t dim, int8_t keepDim, Tensor *result);
const char *Relu(Tensor a, Tensor *result);
const char *LeakyRelu(Tensor a, double negative_slope, Tensor *result);
const char *Tanh(Tensor a, Tensor *result);
const char *Sigmoid(Tensor a, Tensor *result);
const char *Add(Tensor a, Tensor other, float alpha, Tensor *result);
const char *Add_(Tensor a, Tensor other, float alpha, Tensor *result);
const char *Sub(Tensor a, Tensor other, float alpha, Tensor *result);
const char *Sub_(Tensor a, Tensor other, float alpha, Tensor *result);
const char *Mul(Tensor a, Tensor other, Tensor *result);
const char *Mul_(Tensor a, Tensor other, Tensor *result);
const char *Div(Tensor a, Tensor other, Tensor *result);
const char *Div_(Tensor a, Tensor other, Tensor *result);
const char *Permute(Tensor a, int64_t *dims, int64_t dims_size, Tensor *result);
const char *AllClose(Tensor a, Tensor b, int64_t *result);
const char *Flatten(Tensor a, int64_t startDim, int64_t endDim, Tensor *result);
const char *TopK(Tensor a, int64_t k, int64_t dim, int8_t largest,
                 int8_t sorted, Tensor *values, Tensor *indices);
const char *Transpose(Tensor a, int64_t dim0, int64_t dim1, Tensor *result);
const char *ExpandAs(Tensor a, Tensor other, Tensor *result);
const char *Eq(Tensor a, Tensor other, Tensor *result);
const char *IndexSelect(Tensor a, int64_t dim, Tensor index, Tensor *result);
const char *View(Tensor a, Tensor *result, int64_t *size, int64_t size_len);
const char *LogSoftmax(Tensor a, int64_t dim, Tensor *result);
const char *Squeeze(Tensor a, Tensor *result);
const char *SqueezeWithDim(Tensor a, int64_t dim, Tensor *result);
const char *Argmin(Tensor a, int64_t *dim, int8_t keepdim, Tensor *result);
const char *Argmax(Tensor a, int64_t *dim, int8_t keepdim, Tensor *result);

const char *Mean(Tensor a, Tensor *result);
const char *Stack(Tensor *tensors, int64_t tensors_size, int64_t dim,
                  Tensor *result);

// Add to the existing declarations
const char *Pow(Tensor input, double exponent, Tensor *result);
const char *Pow_(Tensor input, double exponent, Tensor *result);  // In-place version
const char *PowOut(Tensor input, double exponent, Tensor out, Tensor *result);  // Out version

const char *Abs(Tensor input, Tensor *result);
const char *Abs_(Tensor input, Tensor *result);  // In-place version
const char *AbsOut(Tensor input, Tensor out, Tensor *result);  // Out version
                                                               //
const char *Sqrt(Tensor input, Tensor *result);
const char *Sqrt_(Tensor input, Tensor *result);
const char *SqrtOut(Tensor input, Tensor out, Tensor *result);

const char *Log(Tensor input, Tensor *result);
const char *Log_(Tensor input, Tensor *result);
const char *LogOut(Tensor input, Tensor out, Tensor *result);

const char *Exp(Tensor input, Tensor *result);
const char *Exp_(Tensor input, Tensor *result);
const char *ExpOut(Tensor input, Tensor out, Tensor *result);

// Shape operations
const char *Unsqueeze(Tensor input, int64_t dim, Tensor *result);
const char *Unsqueeze_(Tensor input, int64_t dim, Tensor *result);  // In-place version
                                                                    //
// Add to the existing declarations
const char *Reshape(Tensor input, int64_t *shape, int64_t shape_len, Tensor *result);

// Add to the existing declarations
const char *Cat(Tensor *tensors, int64_t tensors_size, int64_t dim, Tensor *result);
const char *CatOut(Tensor *tensors, int64_t tensors_size, int64_t dim, Tensor out, Tensor *result);


// Named Dimensions Operations
const char *NewDimname(const char* name, bool is_wildcard, Dimname *result);
const char *FreeDimname(Dimname dimname);
const char *SetTensorDimnames(Tensor tensor, Dimname* names, int64_t num_names);
const char *GetTensorDimnames(Tensor tensor, Dimname** names, int64_t* num_names);
const char *DimnameToString(Dimname dimname, const char** result);

// Add to the existing declarations
// Change from double to float
#ifdef __cplusplus
}
#endif
