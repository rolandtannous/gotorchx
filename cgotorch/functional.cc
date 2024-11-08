// Copyright 2020, GoTorch Authors
#include "cgotorch/functional.h"

#include <string>
#include <unordered_map>
#include <vector>

const char *BatchNorm(Tensor input, Tensor weight, Tensor bias,
                      Tensor running_mean, Tensor running_var, int8_t training,
                      double momentum, double eps, Tensor *result) {
  try {
    auto output = torch::nn::functional::batch_norm(
        *input, (running_mean ? *running_mean : at::Tensor()),
        (running_var ? *running_var : at::Tensor()),
        torch::nn::functional::BatchNormFuncOptions()
            .weight(weight ? *weight : at::Tensor())
            .bias(bias ? *bias : at::Tensor())
            .training(training)
            .momentum(momentum)
            .eps(eps));
    *result = new at::Tensor(output);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Conv2d(Tensor input, Tensor weight, Tensor bias,
                   int64_t *stride_data, int64_t stride_len,
                   int64_t *padding_data, int64_t padding_len,
                   int64_t *dilation_data, int64_t dilation_len, int64_t groups,
                   Tensor *result) {
  try {
    auto output = torch::nn::functional::conv2d(
        *input, *weight,
        torch::nn::functional::Conv2dFuncOptions()
            .bias(bias ? *bias : at::Tensor())
            .stride(torch::IntArrayRef(stride_data, stride_len))
            .padding(torch::IntArrayRef(padding_data, padding_len))
            .dilation(torch::IntArrayRef(dilation_data, dilation_len))
            .groups(groups));
    *result = new at::Tensor(output);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *ConvTranspose2d(Tensor input, Tensor weight, Tensor bias,
                            int64_t *stride_data, int64_t stride_len,
                            int64_t *padding_data, int64_t padding_len,
                            int64_t *output_padding_data,
                            int64_t output_padding_len, int64_t groups,
                            int64_t *dilation_data, int64_t dilation_len,
                            Tensor *result) {
  try {
    auto output = torch::nn::functional::conv_transpose2d(
        *input, *weight,
        torch::nn::functional::ConvTranspose2dFuncOptions()
            .bias(bias ? *bias : at::Tensor())
            .stride(torch::IntArrayRef(stride_data, stride_len))
            .padding(torch::IntArrayRef(padding_data, padding_len))
            .output_padding(
                torch::IntArrayRef(output_padding_data, output_padding_len))
            .groups(groups)
            .dilation(torch::IntArrayRef(dilation_data, dilation_len)));
    *result = new at::Tensor(output);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *BinaryCrossEntropy(Tensor input, Tensor target, Tensor weight,
                               const char *reduction, Tensor *result) {
  static std::unordered_map<std::string, torch::nn::BCELossOptions::reduction_t>
      reduce_map = {
          {"none", torch::kNone},
          {"mean", torch::kMean},
          {"sum", torch::kSum},
      };
  try {
    auto output = torch::nn::functional::binary_cross_entropy(
        *input, *target,
        torch::nn::functional::BinaryCrossEntropyFuncOptions()
            .weight((weight ? *weight : torch::Tensor()))
            .reduction(reduce_map[std::string(reduction)]));
    *result = new at::Tensor(output);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *CrossEntropy(Tensor input, Tensor target, Tensor weight,
                         int64_t ignore_index, const char *reduction,
                         Tensor *result) {
  static std::unordered_map<std::string,
                            torch::nn::CrossEntropyLossOptions::reduction_t>
      reduce_map = {
          {"none", torch::kNone},
          {"mean", torch::kMean},
          {"sum", torch::kSum},
      };
  try {
    auto output = torch::nn::functional::cross_entropy(
        *input, *target,
        torch::nn::functional::CrossEntropyFuncOptions()
            .weight((weight ? *weight : torch::Tensor()))
            .ignore_index(ignore_index)
            .reduction(reduce_map[std::string(reduction)]));
    *result = new at::Tensor(output);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *FRelu(Tensor input, int8_t inplace, Tensor *result) {
  try {
    auto out = torch::nn::functional::relu(
        *input, torch::nn::functional::ReLUFuncOptions(inplace));
    *result = new at::Tensor(out);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *FLeakyRelu(Tensor input, double negative_slope, int8_t inplace,
                       Tensor *result) {
  try {
    auto out = torch::nn::functional::leaky_relu(
        *input, torch::nn::functional::LeakyReLUFuncOptions()
                    .negative_slope(negative_slope)
                    .inplace(inplace));
    *result = new at::Tensor(out);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *NllLoss(Tensor input, Tensor target, Tensor weight,
                    int64_t ignore_index, const char *reduction,
                    Tensor *result) {
  static std::unordered_map<std::string, torch::nn::NLLLossOptions::reduction_t>
      reduce_map = {
          {"none", torch::kNone},
          {"mean", torch::kMean},
          {"sum", torch::kSum},
      };
  try {
    auto output = torch::nn::functional::nll_loss(
        *input, *target,
        torch::nn::functional::NLLLossFuncOptions()
            .weight((weight ? *weight : torch::Tensor()))
            .ignore_index(ignore_index)
            .reduction(reduce_map[std::string(reduction)]));
    *result = new at::Tensor(output);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Linear(Tensor input, Tensor weight, Tensor bias, Tensor *result) {
  try {
    auto out = torch::nn::functional::linear(*input, *weight,
                                             (bias ? *bias : torch::Tensor()));
    *result = new at::Tensor(out);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *MaxPool2d(Tensor input, int64_t *kernel_data, int64_t kernel_len,
                      int64_t *stride_data, int64_t stride_len,
                      int64_t *padding_data, int64_t padding_len,
                      int64_t *dilation_data, int64_t dilation_len,
                      int8_t ceil_mode, Tensor *result) {
  try {
    auto out = torch::nn::functional::max_pool2d(
        *input, torch::nn::functional::MaxPool2dFuncOptions(
                    torch::IntArrayRef(kernel_data, kernel_len))
                    .stride(torch::IntArrayRef(stride_data, stride_len))
                    .padding(torch::IntArrayRef(padding_data, padding_len))
                    .dilation(torch::IntArrayRef(dilation_data, dilation_len))
                    .ceil_mode(ceil_mode));
    *result = new at::Tensor(out);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *AdaptiveAvgPool2d(Tensor input, int64_t *output_size_data,
                              int64_t output_size_len, Tensor *result) {
  try {
    auto out = torch::nn::functional::adaptive_avg_pool2d(
        *input, torch::nn::functional::AdaptiveAvgPool2dFuncOptions(
                    torch::IntArrayRef(output_size_data, output_size_len)));
    *result = new at::Tensor(out);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *MultiHeadAttention(Tensor query, Tensor key, Tensor value,
    int64_t num_heads, Tensor mask, Tensor dropout, Tensor *result) {
    try {
        auto q = static_cast<torch::Tensor*>(query);
        auto k = static_cast<torch::Tensor*>(key);
        auto v = static_cast<torch::Tensor*>(value);

        // Check dimensions - expect 3D inputs (S, N, E)
        TORCH_CHECK(q->dim() == 3 && k->dim() == 3 && v->dim() == 3,
            "query, key, value must be 3-dimensional (seq_len, batch, embed_dim)");

        // Get dimensions
        auto seq_len = q->size(0);
        auto batch_size = q->size(1);
        auto embed_dim = q->size(2);
        auto head_dim = embed_dim / num_heads;

        // Check shapes
        TORCH_CHECK(k->size(2) == embed_dim && v->size(2) == embed_dim,
            "key and value must have the same embed_dim as query");
        TORCH_CHECK(k->size(0) == v->size(0),
            "key and value must have the same sequence length");
        TORCH_CHECK(embed_dim % num_heads == 0,
            "embed_dim must be divisible by num_heads");

        // Scale factor
        float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // Reshape and transpose: (S, N, E) -> (N, H, S, D)
        auto q_4d = q->reshape({seq_len, batch_size, num_heads, head_dim}).permute({1, 2, 0, 3});
        auto k_4d = k->reshape({k->size(0), batch_size, num_heads, head_dim}).permute({1, 2, 0, 3});
        auto v_4d = v->reshape({v->size(0), batch_size, num_heads, head_dim}).permute({1, 2, 0, 3});

        // Scaled dot-product attention
        auto attn_weights = torch::matmul(q_4d, k_4d.transpose(-2, -1)) * scaling;

        // Apply mask if provided
        // In the MultiHeadAttention function, update the mask handling:
        if (mask) {
            auto mask_tensor = *static_cast<torch::Tensor*>(mask);
            TORCH_CHECK(mask_tensor.scalar_type() == torch::kBool,
                "mask must be a boolean tensor");
            attn_weights = attn_weights.masked_fill(mask_tensor,
                -std::numeric_limits<float>::infinity());
        }
        attn_weights = torch::softmax(attn_weights, -1);

        // Apply dropout if provided
        if (dropout) {
            auto dropout_p = static_cast<torch::Tensor*>(dropout)->item<float>();
            attn_weights = torch::dropout(attn_weights, dropout_p, true);
        }

        auto attn_output = torch::matmul(attn_weights, v_4d);

        // Reshape back to (S, N, E)
        attn_output = attn_output.permute({2, 0, 1, 3})
                                .reshape({seq_len, batch_size, embed_dim});

        *result = new torch::Tensor(attn_output);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char *FlashAttention(Tensor query, Tensor key, Tensor value,
    double dropout_p, bool is_causal, Tensor *result) {
    try {
        auto q = static_cast<torch::Tensor*>(query);
        auto k = static_cast<torch::Tensor*>(key);
        auto v = static_cast<torch::Tensor*>(value);

        // Check if we can use Flash Attention
        TORCH_CHECK(q->device().is_cuda(), "Flash Attention requires CUDA tensors");
        TORCH_CHECK(q->scalar_type() == torch::kFloat16 || q->scalar_type() == torch::kBFloat16,
            "Flash Attention only supports fp16 or bf16 data types");

        // Check dimensions - expect (B, H, S, D) format
        TORCH_CHECK(q->dim() == 4, "query must be 4-dimensional (batch_size, num_heads, seq_len, head_dim)");
        TORCH_CHECK(k->dim() == 4 && v->dim() == 4, "key and value must be 4-dimensional");

        // Verify shapes match
        TORCH_CHECK(k->size(0) == q->size(0) && v->size(0) == q->size(0), "batch sizes must match");
        TORCH_CHECK(k->size(1) == q->size(1) && v->size(1) == q->size(1), "num_heads must match");
        TORCH_CHECK(k->size(3) == q->size(3) && v->size(3) == q->size(3), "head_dim must match");
        TORCH_CHECK(k->size(2) == v->size(2), "key and value sequence lengths must match");

        // Verify tensors are contiguous
        TORCH_CHECK(q->is_contiguous() && k->is_contiguous() && v->is_contiguous(),
            "input tensors must be contiguous");

        // Call Flash Attention
        auto [output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k,
              philox_seed, philox_offset, debug_mask] =
            torch::_scaled_dot_product_flash_attention(
                *q, *k, *v,
                dropout_p,
                is_causal,
                false,  // return_debug_mask
                std::nullopt  // scale
            );

        *result = new torch::Tensor(output);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}
