#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

#include "SYCLHelpers.h"
#include "Utils.h"

/**
 * @brief Perform topk after softmax on gating_output.
 * @param gating_output The gating output tensor of shape [n_tokens, n_experts].
 * @param n_topk The number of top experts to select.
 * @return A tuple of tensors (topk_weights, topk_indices, rows_for_experts,
 * offsets).
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> topk_softmax(
    const at::Tensor& gating_output,
    const int64_t n_topk,
    const bool renormalize) {
  auto shape = gating_output.sizes().vec();
  TORCH_CHECK(
      shape.size() == 2,
      "gating_output must be 2D tensor, but got ",
      shape.size(),
      "D");
  int n_tokens = shape[0];
  int n_experts = shape[1];

  TORCH_CHECK(
      n_experts <= 128,
      "n_experts only support up to 128, but got ",
      n_experts);

  int n_experts_aligned = (n_experts + 7) / 8 * 8; // align to 8
  auto topk_weights =
      at::empty({n_tokens, n_topk}, at::dtype(at::kFloat).device(at::kXPU));
  auto topk_indices =
      at::empty({n_tokens, n_topk}, at::dtype(at::kInt).device(at::kXPU));
  auto rows_for_experts =
      at::zeros({n_experts_aligned}, at::dtype(at::kInt).device(at::kXPU));
  auto offsets =
      at::empty({n_tokens, n_topk}, at::dtype(at::kInt).device(at::kXPU));
//   IPEX_DISPATCH_FLOATING_TYPES_AND2(
//       at::kBFloat16,
//       at::kHalf,
//       gating_output.scalar_type(),
//       "fused_topk_softmax_kernel",
//       [&]() {
//         TopKSoftmaxImpl::fused_topk_softmax<scalar_t>(
//             reinterpret_cast<scalar_t*>(gating_output.data_ptr()),
//             reinterpret_cast<float*>(topk_weights.data_ptr()),
//             reinterpret_cast<int*>(topk_indices.data_ptr()),
//             reinterpret_cast<int*>(rows_for_experts.data_ptr()),
//             reinterpret_cast<int*>(offsets.data_ptr()),
//             renormalize,
//             n_tokens,
//             n_experts,
//             n_topk);
//       });

  return std::make_tuple(topk_weights, topk_indices, rows_for_experts, offsets);
}