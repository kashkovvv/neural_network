#pragma once

#include <concepts>
#include <optional>
#include <stdexcept>
#include <utility>

#include "nn/detail/tensor_concepts.hpp"
#include "nn/tensor.hpp"

namespace nn::detail {

template <typename Element>
concept linear_element =
    tensor_element<Element> && std::floating_point<Element>;

}  // namespace nn::detail

namespace nn {

template <detail::linear_element T>
class Linear {
 public:
  using value_type = T;
  using tensor_type = Tensor<value_type>;
  using size_type = typename tensor_type::size_type;

  explicit Linear(tensor_type weights,
                  std::optional<tensor_type> bias = std::nullopt)
      : weights_(std::move(weights)), bias_(std::move(bias)) {
    if (weights_.rank() != 2) {
      throw std::invalid_argument("linear weights must have rank 2");
    }

    if (!bias_) {
      return;
    }

    if (bias_->rank() != 1) {
      throw std::invalid_argument("linear bias must have rank 1");
    }

    if (bias_->numel() != out_features()) {
      throw std::invalid_argument(
          "linear bias numel does not match out_features");
    }
  }

  Linear(const Linear&) = default;

  Linear(Linear&& other) noexcept
      : weights_(std::move(other.weights_)), bias_(std::move(other.bias_)) {
    other.bias_.reset();
  }

  Linear& operator=(const Linear& other) {
    if (this == &other) {
      return *this;
    }

    Linear temporary(other);

    swap(temporary);

    return *this;
  }

  Linear& operator=(Linear&& other) noexcept {
    if (this == &other) {
      return *this;
    }

    weights_ = std::move(other.weights_);
    bias_ = std::move(other.bias_);

    other.bias_.reset();

    return *this;
  }

  void swap(Linear& other) noexcept {
    weights_.swap(other.weights_);
    bias_.swap(other.bias_);
  }

  [[nodiscard]] size_type in_features() const noexcept {
    return is_empty_sentinel() ? size_type{0} : weights_.shape()[0];
  }

  [[nodiscard]] size_type out_features() const noexcept {
    return is_empty_sentinel() ? size_type{0} : weights_.shape()[1];
  }

  [[nodiscard]] const tensor_type& weights() const& noexcept {
    return weights_;
  }

  const tensor_type& weights() && = delete;
  const tensor_type& weights() const&& = delete;

  [[nodiscard]] const std::optional<tensor_type>& bias() const& noexcept {
    return bias_;
  }

  const std::optional<tensor_type>& bias() && = delete;
  const std::optional<tensor_type>& bias() const&& = delete;

  [[nodiscard]] tensor_type operator()(const tensor_type& input) const& {
    tensor_type result =
        input.rank() == 1 ? input.vecmat(weights_) : input.matmul(weights_);

    if (bias_) {
      result += *bias_;
    }

    return result;
  }

 private:
  [[nodiscard]] bool is_empty_sentinel() const noexcept {
    return weights_.rank() == 0 && weights_.numel() == 0 && !bias_;
  }

  tensor_type weights_;
  std::optional<tensor_type> bias_;
};

template <detail::linear_element T>
void swap(Linear<T>& lhs, Linear<T>& rhs) noexcept {
  lhs.swap(rhs);
}

}  // namespace nn
