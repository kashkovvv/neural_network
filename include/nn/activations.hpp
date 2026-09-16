#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <stdexcept>
#include <utility>

#include "nn/tensor.hpp"

namespace nn::detail {

template <std::floating_point T, typename UnaryOperation>
void apply_unary_activation_inplace(Tensor<T>& tensor,
                                    UnaryOperation operation) {
  if (tensor.rank() == 0 && tensor.numel() == 0) {
    throw std::invalid_argument("tensor is an empty sentinel");
  }

  std::ranges::transform(tensor, tensor.begin(), std::move(operation));
}

}  // namespace nn::detail

namespace nn {

template <std::floating_point T>
[[nodiscard]] Tensor<T> relu(Tensor<T> tensor) {
  detail::apply_unary_activation_inplace(
      tensor, [](T element) { return element < T{} ? T{} : element; });

  return tensor;
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> sigmoid(Tensor<T> tensor) {
  detail::apply_unary_activation_inplace(tensor, [](T element) {
    if (element >= T{}) {
      const T exponential = std::exp(-element);

      return T{1} / (T{1} + exponential);
    }

    const T exponential = std::exp(element);

    return exponential / (T{1} + exponential);
  });

  return tensor;
}

}  // namespace nn
