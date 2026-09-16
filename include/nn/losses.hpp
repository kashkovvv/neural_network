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

template <std::floating_point T>
[[nodiscard]] Tensor<T> huber_loss(Tensor<T> prediction,
                                   const Tensor<T>& target,
                                   typename Tensor<T>::value_type delta = T{
                                       1}) {
  detail::validate_elementwise_loss_inputs(prediction, target);

  if (delta <= T{} || !std::isfinite(delta)) {
    throw std::domain_error("huber loss delta must be finite and positive");
  }

  prediction -= target;
  std::ranges::transform(prediction, prediction.begin(), [delta](T error) {
    const T absolute_error = std::abs(error);

    if (absolute_error <= delta) {
      return (T{0.5} * error) * error;
    }

    return delta * (absolute_error - T{0.5} * delta);
  });

  return prediction.mean();
}

}  // namespace nn
