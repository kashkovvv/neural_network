#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <span>
#include <stdexcept>
#include <utility>

#include "nn/tensor.hpp"

namespace nn::detail {

template <std::floating_point T>
void validate_activation_input(const Tensor<T>& tensor) {
  if (tensor.rank() == 0 && tensor.numel() == 0) {
    throw std::invalid_argument("tensor is an empty sentinel");
  }
}

template <std::floating_point T, typename UnaryOperation>
void apply_elementwise_activation_inplace(Tensor<T>& tensor,
                                          UnaryOperation operation) {
  validate_activation_input(tensor);

  std::ranges::transform(tensor, tensor.begin(), std::move(operation));
}

}  // namespace nn::detail

namespace nn {

template <std::floating_point T>
[[nodiscard]] Tensor<T> relu(Tensor<T> tensor) {
  detail::apply_elementwise_activation_inplace(
      tensor, [](T element) { return element < T{} ? T{} : element; });

  return tensor;
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> sigmoid(Tensor<T> tensor) {
  detail::apply_elementwise_activation_inplace(tensor, [](T element) {
    if (element >= T{}) {
      const T exponential = std::exp(-element);

      return T{1} / (T{1} + exponential);
    }

    const T exponential = std::exp(element);

    return exponential / (T{1} + exponential);
  });

  return tensor;
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> tanh(Tensor<T> tensor) {
  detail::apply_elementwise_activation_inplace(
      tensor, [](T element) { return std::tanh(element); });

  return tensor;
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> softmax(Tensor<T> tensor,
                                typename Tensor<T>::size_type axis) {
  detail::validate_activation_input(tensor);

  if (axis >= tensor.rank()) {
    throw std::out_of_range("softmax axis is out of range");
  }

  if (tensor.numel() == 0) {
    return tensor;
  }

  using size_type = typename Tensor<T>::size_type;

  std::span<T> elements = tensor.elements();
  const size_type axis_extent = tensor.shape()[axis];
  const size_type inner_size = tensor.strides()[axis];
  const size_type outer_block_size = axis_extent * inner_size;
  const size_type outer_size = tensor.numel() / outer_block_size;

  for (size_type outer_index = 0; outer_index < outer_size; ++outer_index) {
    for (size_type inner_index = 0; inner_index < inner_size; ++inner_index) {
      const size_type slice_offset =
          outer_index * outer_block_size + inner_index;
      T maximum = elements[slice_offset];

      for (size_type axis_index = 1; axis_index < axis_extent; ++axis_index) {
        const size_type element_offset = slice_offset + axis_index * inner_size;

        maximum = std::max(maximum, elements[element_offset]);
      }

      T exponential_sum = T{};

      for (size_type axis_index = 0; axis_index < axis_extent; ++axis_index) {
        const size_type element_offset = slice_offset + axis_index * inner_size;
        const T exponential = std::exp(elements[element_offset] - maximum);

        elements[element_offset] = exponential;
        exponential_sum += exponential;
      }

      for (size_type axis_index = 0; axis_index < axis_extent; ++axis_index) {
        const size_type element_offset = slice_offset + axis_index * inner_size;

        elements[element_offset] /= exponential_sum;
      }
    }
  }

  return tensor;
}

}  // namespace nn
