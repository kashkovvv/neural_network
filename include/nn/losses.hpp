#pragma once

#include <algorithm>
#include <concepts>
#include <stdexcept>

#include "nn/tensor.hpp"

namespace nn {

template <std::floating_point T>
[[nodiscard]] Tensor<T> mse_loss(Tensor<T> prediction,
                                 const Tensor<T>& target) {
  if (prediction.rank() == 0 && prediction.numel() == 0) {
    throw std::invalid_argument("tensor is an empty sentinel");
  }

  if (target.rank() == 0 && target.numel() == 0) {
    throw std::invalid_argument("tensor is an empty sentinel");
  }

  if (!std::ranges::equal(prediction.shape(), target.shape())) {
    throw std::invalid_argument("mse loss requires tensors with equal shapes");
  }

  prediction -= target;
  prediction *= prediction;

  return prediction.mean();
}

}  // namespace nn
