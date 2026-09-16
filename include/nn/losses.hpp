#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <stdexcept>

#include "nn/tensor.hpp"

namespace nn::detail {

template <std::floating_point T>
void validate_elementwise_loss_inputs(const Tensor<T>& prediction,
                                      const Tensor<T>& target) {
  if (prediction.rank() == 0 && prediction.numel() == 0) {
    throw std::invalid_argument("tensor is an empty sentinel");
  }

  if (target.rank() == 0 && target.numel() == 0) {
    throw std::invalid_argument("tensor is an empty sentinel");
  }

  if (!std::ranges::equal(prediction.shape(), target.shape())) {
    throw std::invalid_argument(
        "elementwise loss requires tensors with equal shapes");
  }
}

}  // namespace nn::detail

namespace nn {

template <std::floating_point T>
[[nodiscard]] Tensor<T> mse_loss(Tensor<T> prediction,
                                 const Tensor<T>& target) {
  detail::validate_elementwise_loss_inputs(prediction, target);

  prediction -= target;
  prediction *= prediction;

  return prediction.mean();
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> mae_loss(Tensor<T> prediction,
                                 const Tensor<T>& target) {
  detail::validate_elementwise_loss_inputs(prediction, target);

  prediction -= target;
  std::ranges::transform(prediction, prediction.begin(),
                         [](T error) { return std::abs(error); });

  return prediction.mean();
}

}  // namespace nn
