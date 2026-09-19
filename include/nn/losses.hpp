#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>

#include "nn/detail/compensated_accumulator.hpp"
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

template <std::floating_point T, typename BinaryOperation>
[[nodiscard]] T reduce_mean_elementwise_loss(const Tensor<T>& prediction,
                                             const Tensor<T>& target,
                                             BinaryOperation operation) {
  validate_elementwise_loss_inputs(prediction, target);

  const std::span<const T> prediction_elements = prediction.elements();
  const std::span<const T> target_elements = target.elements();

  assert(prediction_elements.size() == target_elements.size());

  CompensatedAccumulator<T> accumulator;

  for (std::size_t element_index = 0;
       element_index < prediction_elements.size(); ++element_index) {
    accumulator.add(std::invoke(operation, prediction_elements[element_index],
                                target_elements[element_index]));
  }

  return accumulator.result() / static_cast<T>(prediction_elements.size());
}

}  // namespace nn::detail

namespace nn {

template <std::floating_point T>
[[nodiscard]] Tensor<T> mse_loss(const Tensor<T>& prediction,
                                 const Tensor<T>& target) {
  const T mean_squared_error = detail::reduce_mean_elementwise_loss(
      prediction, target, [](T prediction_element, T target_element) {
        const T error = prediction_element - target_element;

        return error * error;
      });

  return Tensor<T>::scalar(mean_squared_error);
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> mae_loss(const Tensor<T>& prediction,
                                 const Tensor<T>& target) {
  const T mean_absolute_error = detail::reduce_mean_elementwise_loss(
      prediction, target, [](T prediction_element, T target_element) {
        return std::abs(prediction_element - target_element);
      });

  return Tensor<T>::scalar(mean_absolute_error);
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> huber_loss(const Tensor<T>& prediction,
                                   const Tensor<T>& target,
                                   typename Tensor<T>::value_type delta = T{
                                       1}) {
  if (delta <= T{} || !std::isfinite(delta)) {
    throw std::domain_error("huber loss delta must be finite and positive");
  }

  const T mean_huber_loss = detail::reduce_mean_elementwise_loss(
      prediction, target, [delta](T prediction_element, T target_element) {
        const T error = prediction_element - target_element;
        const T absolute_error = std::abs(error);

        return absolute_error <= delta
                   ? (T{0.5} * error) * error
                   : delta * (absolute_error - T{0.5} * delta);
      });

  return Tensor<T>::scalar(mean_huber_loss);
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> binary_cross_entropy_with_logits(
    const Tensor<T>& logits, const Tensor<T>& target) {
  const T mean_binary_cross_entropy_with_logits_loss =
      detail::reduce_mean_elementwise_loss(
          logits, target, [](T logit, T target_value) {
            if (!(target_value >= T{} && target_value <= T{1})) {
              throw std::domain_error(
                  "binary cross entropy target must be in [0, 1]");
            }

            T element_loss = std::log1p(std::exp(-std::abs(logit)));

            if (logit >= T{} && target_value != T{1}) {
              element_loss += (T{1} - target_value) * logit;
            } else if (logit < T{} && target_value != T{}) {
              element_loss += -target_value * logit;
            }

            return element_loss;
          });

  return Tensor<T>::scalar(mean_binary_cross_entropy_with_logits_loss);
}

}  // namespace nn
