#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nn {

template <typename T>
class Tensor {
 public:
  using value_type = T;
  using size_type = std::size_t;
  using shape_type = std::vector<size_type>;
  using strides_type = std::vector<size_type>;
  using storage_type = std::vector<value_type>;

  explicit Tensor(shape_type shape)
      : shape_(std::move(shape)), strides_(shape_.size()) {
    size_type running_stride = 1;

    for (size_type remaining_axes = shape_.size(); remaining_axes != 0;
         --remaining_axes) {
      const size_type axis = remaining_axes - 1;
      const size_type extent = shape_[axis];

      strides_[axis] = running_stride;
      running_stride = checked_multiply(running_stride, extent);
    }

    if (running_stride > storage_.max_size()) {
      throw std::length_error("tensor size exceeds storage max_size");
    }

    storage_.resize(running_stride);
  }

  [[nodiscard]] static Tensor full(shape_type shape, const value_type& value) {
    Tensor tensor(std::move(shape));
    std::ranges::fill(tensor.storage_, value);
    return tensor;
  }

  [[nodiscard]] static Tensor zeros(shape_type shape) {
    return full(std::move(shape), value_type{});
  }

  [[nodiscard]] static Tensor scalar(value_type value) {
    Tensor tensor(shape_type{});
    tensor.storage_.front() = std::move(value);
    return tensor;
  }

  [[nodiscard]] size_type rank() const noexcept { return shape_.size(); }

  [[nodiscard]] size_type numel() const noexcept { return storage_.size(); }

  [[nodiscard]] std::span<const size_type> shape() const& noexcept {
    return shape_;
  }

  std::span<const size_type> shape() && = delete;
  std::span<const size_type> shape() const&& = delete;

  [[nodiscard]] std::span<const size_type> strides() const& noexcept {
    return strides_;
  }

  std::span<const size_type> strides() && = delete;
  std::span<const size_type> strides() const&& = delete;

  [[nodiscard]] std::span<value_type> elements() & noexcept { return storage_; }

  [[nodiscard]] std::span<const value_type> elements() const& noexcept {
    return storage_;
  }

  std::span<value_type> elements() && = delete;
  std::span<const value_type> elements() const&& = delete;

 private:
  [[nodiscard]] static size_type checked_multiply(size_type lhs,
                                                  size_type rhs) {
    const size_type max_value = std::numeric_limits<size_type>::max();

    if (rhs != 0 && lhs > max_value / rhs) {
      throw std::overflow_error("tensor shape product overflows size_type");
    }

    return lhs * rhs;
  }

  shape_type shape_;
  strides_type strides_;
  storage_type storage_;
};

}  // namespace nn
