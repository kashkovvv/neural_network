#pragma once

#include <cstddef>
#include <limits>
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

    if (running_stride > data_.max_size()) {
      throw std::length_error("tensor size exceeds storage max_size");
    }

    data_.resize(running_stride);
  }

  [[nodiscard]] size_type rank() const noexcept { return shape_.size(); }

  [[nodiscard]] size_type numel() const noexcept { return data_.size(); }

  [[nodiscard]] const shape_type& shape() const noexcept { return shape_; }

  [[nodiscard]] const strides_type& strides() const noexcept {
    return strides_;
  }

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
  storage_type data_;
};

}  // namespace nn
