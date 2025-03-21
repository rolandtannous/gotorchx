// Copyright 2020, GoTorch Authors
#include "cgotorch/torch.h"

#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

const char *exception_str(const char *e) {
  auto len = strlen(e);
  auto r = new char[len + 1];
  snprintf(r, len + 1, "%s", e);
  return r;
}

////////////////////////////////////////////////////////////////////////////////
// Tensor construction and operations
////////////////////////////////////////////////////////////////////////////////

const char *RandN(int64_t *size, int64_t length, int64_t requires_grad,
                  Tensor *result) {
  try {
    at::Tensor t =
        torch::randn(torch::IntArrayRef(size, length),
                     at::TensorOptions().requires_grad(requires_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Rand(int64_t *size, int64_t length, int64_t requires_grad,
                 Tensor *result) {
  try {
    at::Tensor t =
        torch::rand(torch::IntArrayRef(size, length),
                    at::TensorOptions().requires_grad(requires_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Stack(Tensor *tensors, int64_t tensors_size, int64_t dim,
                  Tensor *result) {
  try {
    std::vector<torch::Tensor> data;
    while (data.size() < tensors_size) data.push_back(**tensors++);
    auto out = at::stack(data, dim);
    *result = new at::Tensor(out);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Empty(int64_t *size, int64_t length, int64_t requires_grad,
                  Tensor *result) {
  try {
    at::Tensor t =
        torch::empty(torch::IntArrayRef(size, length),
                     at::TensorOptions().requires_grad(requires_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

// torch.ones
const char *Ones(int64_t *size, int64_t length, int64_t requires_grad,
                 Tensor *result) {
  try {
    at::Tensor t =
        torch::ones(torch::IntArrayRef(size, length),
                    at::TensorOptions().requires_grad(requires_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

// torch.eye
const char *Eye(int64_t n, int64_t m, int64_t requires_grad, Tensor *result) {
  try {
    at::Tensor t =
        torch::eye(n, m, at::TensorOptions().requires_grad(requires_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

// torch.full, only for float32
const char *Full(int64_t *size, int64_t length, float value,
                 int64_t requires_grad, Tensor *result) {
  try {
    at::Tensor t =
        torch::full(torch::IntArrayRef(size, length), value,
                    torch::TensorOptions().requires_grad(requires_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

// torch.arange
const char *Arange(float start, float end, float step, int64_t requires_grad,
                   Tensor *result) {
  try {
    at::Tensor t = torch::arange(
        start, end, step, torch::TensorOptions().requires_grad(requires_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

// torch.linspace
const char *Linspace(float start, float end, int64_t steps,
                     int64_t requires_grad, Tensor *result) {
  try {
    at::Tensor t = torch::linspace(
        start, end, steps, torch::TensorOptions().requires_grad(requires_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

// torch.logspace
const char *Logspace(float start, float end, int64_t steps, double base,
                     int64_t requires_grad, Tensor *result) {
  try {
    at::Tensor t =
        torch::logspace(start, end, steps, base,
                        torch::TensorOptions().requires_grad(requires_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Equal(Tensor a, Tensor b, int64_t *result) {
  try {
    *result = at::equal(*a, *b) ? 1 : 0;
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *MM(Tensor a, Tensor b, Tensor *result) {
  try {
    at::Tensor c = at::mm(*a, *b);
    *result = new at::Tensor(c);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Sum(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->sum());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *SumByDim(Tensor a, int64_t dim, int8_t keepDim, Tensor *result) {
  try {
    *result = new at::Tensor(a->sum(dim, keepDim));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Relu(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->relu());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *LeakyRelu(Tensor a, double negative_slope, Tensor *result) {
  try {
    *result = new at::Tensor(at::leaky_relu(*a, negative_slope));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Tanh(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->tanh());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Sigmoid(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->sigmoid());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Add(Tensor a, Tensor other, float alpha, Tensor *result) {
  try {
    *result = new at::Tensor(torch::add(*a, *other, alpha));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Add_(Tensor a, Tensor other, float alpha, Tensor *result) {
  try {
    *result = new at::Tensor(a->add_(*other, alpha));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Sub(Tensor a, Tensor other, float alpha, Tensor *result) {
  try {
    *result = new at::Tensor(torch::sub(*a, *other, alpha));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Sub_(Tensor a, Tensor other, float alpha, Tensor *result) {
  try {
    *result = new at::Tensor(a->sub_(*other, alpha));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Mul(Tensor a, Tensor other, Tensor *result) {
  try {
    *result = new at::Tensor(torch::mul(*a, *other));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Mul_(Tensor a, Tensor other, Tensor *result) {
  try {
    *result = new at::Tensor(a->mul_(*other));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Div(Tensor a, Tensor other, Tensor *result) {
  try {
    *result = new at::Tensor(torch::div(*a, *other));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Div_(Tensor a, Tensor other, Tensor *result) {
  try {
    *result = new at::Tensor(a->div_(*other));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Permute(Tensor a, int64_t *dims, int64_t dims_size,
                    Tensor *result) {
  try {
    c10::ArrayRef<int64_t> d(dims, dims_size);
    *result = new at::Tensor(a->permute(d));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *AllClose(Tensor a, Tensor b, int64_t *result) {
  try {
    *result = at::allclose(*a, *b) ? 1 : 0;
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Flatten(Tensor a, int64_t startDim, int64_t endDim,
                    Tensor *result) {
  try {
    *result = new at::Tensor(torch::flatten(*a, startDim, endDim));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *TopK(Tensor a, int64_t k, int64_t dim, int8_t largest,
                 int8_t sorted, Tensor *values, Tensor *indices) {
  try {
    auto outputs = torch::topk(*a, k, dim, largest, sorted);
    *values = new at::Tensor(std::get<0>(outputs));
    *indices = new at::Tensor(std::get<1>(outputs));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Transpose(Tensor a, int64_t dim0, int64_t dim1, Tensor *result) {
  try {
    *result = new at::Tensor(torch::transpose(*a, dim0, dim1));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *ExpandAs(Tensor a, Tensor other, Tensor *result) {
  try {
    *result = new at::Tensor(a->expand_as(*other));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Eq(Tensor a, Tensor other, Tensor *result) {
  try {
    *result = new at::Tensor(torch::eq(*a, *other));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *IndexSelect(Tensor a, int64_t dim, Tensor index, Tensor *result) {
  try {
    *result = new at::Tensor(torch::index_select(*a, dim, *index));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *View(Tensor a, Tensor *result, int64_t *size, int64_t size_len) {
  try {
    *result = new at::Tensor(a->view(torch::IntArrayRef(size, size_len)));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *LogSoftmax(Tensor a, int64_t dim, Tensor *result) {
  try {
    *result = new at::Tensor(a->log_softmax(dim));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Squeeze(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->squeeze());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *SqueezeWithDim(Tensor a, int64_t dim, Tensor *result) {
  try {
    *result = new at::Tensor(a->squeeze(dim));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

// We use the pointer int64_t* to represent an optional int64_t parameter -- the
// value nullptr indicate not-specified.  Please be aware that we need only one
// "pointerized" parameter because C++ doesn't allow named parameters and the
// rest optional parameters don't need to be pointerized.
const char *Argmin(Tensor a, int64_t *dim, int8_t keepdim, Tensor *result) {
  try {
    if (dim == nullptr) {
      *result = new at::Tensor(a->argmin());
    } else {
      *result = new at::Tensor(a->argmin(*dim, static_cast<bool>(keepdim)));
    }
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Argmax(Tensor a, int64_t *dim, int8_t keepdim, Tensor *result) {
  try {
    if (dim == nullptr) {
      *result = new at::Tensor(a->argmax());
    } else {
      *result = new at::Tensor(a->argmax(*dim, static_cast<bool>(keepdim)));
    }
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}



const char *Pow(Tensor input, double exponent, Tensor *result) {
    try {
        at::Tensor out = at::pow(*input, exponent);
        *result = new at::Tensor(out);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *Pow_(Tensor input, double exponent, Tensor *result) {
    try {
        input->pow_(exponent);  // Using the tensor's in-place method directly
        *result = new at::Tensor(*input);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *PowOut(Tensor input, double exponent, Tensor out, Tensor *result) {
    try {
        at::pow_out(*out, *input, exponent);
        *result = new at::Tensor(*out);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *Abs(Tensor input, Tensor *result) {
    try {
        // Using at:: since this is a fundamental operation
        at::Tensor out = at::abs(*input);
        *result = new at::Tensor(out);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *Abs_(Tensor input, Tensor *result) {
    try {
        at::Tensor out = at::abs_(*input);
        *result = new at::Tensor(out);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *AbsOut(Tensor input, Tensor out, Tensor *result) {
    try {
        at::abs_out(*out, *input);
        *result = new at::Tensor(*out);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *Sqrt(Tensor input, Tensor *result) {
    try {
        at::Tensor out = at::sqrt(*input);
        *result = new at::Tensor(out);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *Sqrt_(Tensor input, Tensor *result) {
    try {
        input->sqrt_();
        *result = new at::Tensor(*input);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *SqrtOut(Tensor input, Tensor out, Tensor *result) {
    try {
        at::sqrt_out(*out, *input);
        *result = new at::Tensor(*out);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *Log(Tensor input, Tensor *result) {
    try {
        at::Tensor out = at::log(*input);
        *result = new at::Tensor(out);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *Log_(Tensor input, Tensor *result) {
    try {
        input->log_();
        *result = new at::Tensor(*input);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *LogOut(Tensor input, Tensor out, Tensor *result) {
    try {
        at::log_out(*out, *input);
        *result = new at::Tensor(*out);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *Exp(Tensor input, Tensor *result) {
    try {
        at::Tensor out = at::exp(*input);
        *result = new at::Tensor(out);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *Exp_(Tensor input, Tensor *result) {
    try {
        input->exp_();
        *result = new at::Tensor(*input);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *ExpOut(Tensor input, Tensor out, Tensor *result) {
    try {
        at::exp_out(*out, *input);
        *result = new at::Tensor(*out);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}


// Shape operations

const char *Unsqueeze(Tensor input, int64_t dim, Tensor *result) {
    try {
        at::Tensor out = at::unsqueeze(*input, dim);
        *result = new at::Tensor(out);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *Unsqueeze_(Tensor input, int64_t dim, Tensor *result) {
    try {
        input->unsqueeze_(dim);
        *result = new at::Tensor(*input);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *Reshape(Tensor input, int64_t *shape, int64_t shape_len, Tensor *result) {
    try {
        at::Tensor out = at::reshape(*input, torch::IntArrayRef(shape, shape_len));
        *result = new at::Tensor(out);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *Cat(Tensor *tensors, int64_t tensors_size, int64_t dim, Tensor *result) {
    try {
        std::vector<at::Tensor> tensor_list;
        for (int64_t i = 0; i < tensors_size; i++) {
            tensor_list.push_back(*tensors[i]);
        }
        at::Tensor output = at::cat(tensor_list, dim);
        *result = new at::Tensor(output);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *CatOut(Tensor *tensors, int64_t tensors_size, int64_t dim, Tensor out, Tensor *result) {
    try {
        std::vector<at::Tensor> tensor_list;
        for (int64_t i = 0; i < tensors_size; i++) {
            tensor_list.push_back(*tensors[i]);
        }
        at::cat_out(*out, tensor_list, dim);
        *result = new at::Tensor(*out);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *CatWithNames(Tensor *tensors, int64_t tensors_size, Dimname dim, Tensor *result) {
    try {
        std::vector<at::Tensor> tensor_list;
        for (int64_t i = 0; i < tensors_size; i++) {
            tensor_list.push_back(*tensors[i]);
        }
        at::Tensor output = at::cat(tensor_list, *dim);
        *result = new at::Tensor(output);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *CatWithNamesOut(Tensor *tensors, int64_t tensors_size, Dimname dim, Tensor out, Tensor *result) {
    try {
        std::vector<at::Tensor> tensor_list;
        for (int64_t i = 0; i < tensors_size; i++) {
            tensor_list.push_back(*tensors[i]);
        }
        at::cat_out(*out, tensor_list, *dim);
        *result = new at::Tensor(*out);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *NewDimname(const char* name, bool is_wildcard, Dimname *result) {
    try {
        if (is_wildcard) {
            *result = new at::Dimname(at::Dimname::wildcard());
        } else {
            if (!at::Dimname::isValidName(name)) {
                return exception_str("Invalid dimension name");
            }
            *result = new at::Dimname(at::Dimname::fromSymbol(at::Symbol::dimname(name)));
        }
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *FreeDimname(Dimname dimname) {
    try {
        delete dimname;
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *SetTensorDimnames(Tensor tensor, Dimname* names, int64_t num_names) {
    try {
        std::vector<at::Dimname> dimnames;
        for (int64_t i = 0; i < num_names; i++) {
            dimnames.push_back(*names[i]);
        }
        // Use the proper function from namedtensor.h
        at::internal_set_names_inplace(*tensor, std::move(dimnames), /*validate_names=*/true);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *GetTensorDimnames(Tensor tensor, Dimname** names, int64_t* num_names) {
    try {
        auto dimnames = tensor->names();  // This should work as it's a getter
        *num_names = dimnames.size();
        *names = new at::Dimname*[*num_names];
        for (int64_t i = 0; i < *num_names; i++) {
            (*names)[i] = new at::Dimname(dimnames[i]);
        }
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *DimnameToString(Dimname dimname, const char** result) {
    try {
        std::string str;
        if (dimname->isWildcard()) {
            str = "*";
        } else {
            str = dimname->symbol().toUnqualString();
        }
        *result = strdup(str.c_str());
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}


// Reduction operations

// Add to cgotorch/torch.cc

const char *Max(Tensor input, Tensor *result) {
    try {
        auto output = at::max(*input);
        *result = new at::Tensor(output);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *MaxDim(Tensor input, int64_t dim, int8_t keepdim,
                  Tensor *values, Tensor *indices) {
    try {
        auto results = at::max(*input, dim, (bool)keepdim);
        *values = new at::Tensor(std::get<0>(results));
        *indices = new at::Tensor(std::get<1>(results));
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *Min(Tensor input, Tensor *result) {
    try {
        auto output = at::min(*input);
        *result = new at::Tensor(output);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *MinDim(Tensor input, int64_t dim, int8_t keepdim,
                  Tensor *values, Tensor *indices) {
    try {
        auto results = at::min(*input, dim, (bool)keepdim);
        *values = new at::Tensor(std::get<0>(results));
        *indices = new at::Tensor(std::get<1>(results));
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *Prod(Tensor input, Tensor *result) {
    try {
        auto output = at::prod(*input);
        *result = new at::Tensor(output);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *ProdDim(Tensor input, int64_t dim, int8_t keepdim, Tensor *result) {
    try {
        auto output = at::prod(*input, dim, (bool)keepdim);
        *result = new at::Tensor(output);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *MaxElementwise(Tensor a, Tensor other, Tensor *result) {
    try {
        auto output = at::max(*a, *other);
        *result = new at::Tensor(output);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *MinElementwise(Tensor a, Tensor other, Tensor *result) {
    try {
        auto output = at::min(*a, *other);
        *result = new at::Tensor(output);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *MaxElementwiseOut(Tensor a, Tensor other, Tensor out, Tensor *result) {
    try {
        auto output = at::max_out(*out, *a, *other);
        *result = new at::Tensor(output);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *MinElementwiseOut(Tensor a, Tensor other, Tensor out, Tensor *result) {
    try {
        auto output = at::min_out(*out, *a, *other);
        *result = new at::Tensor(output);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}


const char *MaxDimOut(Tensor input, int64_t dim, int8_t keepdim,
                     Tensor values, Tensor indices,
                     Tensor *result_values, Tensor *result_indices) {
    try {
        auto results = at::max_out(*values, *indices, *input, dim, (bool)keepdim);
        *result_values = new at::Tensor(std::get<0>(results));
        *result_indices = new at::Tensor(std::get<1>(results));
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *MinDimOut(Tensor input, int64_t dim, int8_t keepdim,
                     Tensor values, Tensor indices,
                     Tensor *result_values, Tensor *result_indices) {
    try {
        auto results = at::min_out(*values, *indices, *input, dim, (bool)keepdim);
        *result_values = new at::Tensor(std::get<0>(results));
        *result_indices = new at::Tensor(std::get<1>(results));
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *ProdOut(Tensor input, Tensor out, Tensor *result) {
    try {
        auto output = at::prod_out(*out, *input);
        *result = new at::Tensor(output);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *ProdDimOut(Tensor input, int64_t dim, int8_t keepdim, Tensor out, Tensor *result) {
    try {
        auto output = at::prod_out(*out, *input, dim, (bool)keepdim);
        *result = new at::Tensor(output);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}
