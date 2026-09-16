#pragma once

#include <algorithm>
#include <concepts>
#include <stdexcept>

#include "nn/tensor.hpp"

namespace nn {

template <std::floating_point T>
[[nodiscard]] Tensor<T> relu(Tensor<T> tensor) {
  if (tensor.rank() == 0 && tensor.numel() == 0) {
    throw std::invalid_argument("tensor is an empty sentinel");
  }

  std::ranges::transform(tensor, tensor.begin(), [](T element) {
    return element < T{} ? T{} : element;
  });

  return tensor;
}

}  // namespace nn
