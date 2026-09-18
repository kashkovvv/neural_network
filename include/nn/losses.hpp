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

template <std::floating_point T>
[[nodiscard]] Tensor<T> binary_cross_entropy_with_logits(
    Tensor<T> logits, const Tensor<T>& target) {
  detail::validate_elementwise_loss_inputs(logits, target);

  if (!std::ranges::all_of(target, [](T target_value) {
        return target_value >= T{} && target_value <= T{1};
      })) {
    throw std::domain_error("binary cross entropy target must be in [0, 1]");
  }

  std::ranges::transform(
      logits, target, logits.begin(), [](T logit, T target_value) {
        if (logit >= T{}) {
          const T softplus_term = std::log1p(std::exp(-logit));

          if (target_value == T{1}) {
            return softplus_term;
          }

          return (T{1} - target_value) * logit + softplus_term;
        }

        const T softplus_term = std::log1p(std::exp(logit));

        if (target_value == T{}) {
          return softplus_term;
        }

        return -target_value * logit + softplus_term;
      });

  return logits.mean();
}

}  // namespace nn
