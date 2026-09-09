#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace nn::detail {

template <typename IndexType>
concept tensor_index =
    std::integral<std::remove_cvref_t<IndexType>> &&
    (!std::same_as<std::remove_cvref_t<IndexType>, bool>) &&
    (!std::same_as<std::remove_cvref_t<IndexType>, char>) &&
    (!std::same_as<std::remove_cvref_t<IndexType>, wchar_t>) &&
    (!std::same_as<std::remove_cvref_t<IndexType>, char8_t>) &&
    (!std::same_as<std::remove_cvref_t<IndexType>, char16_t>) &&
    (!std::same_as<std::remove_cvref_t<IndexType>, char32_t>);

}  // namespace nn::detail

namespace nn {

template <typename T>
class Tensor {
 public:
  using value_type = T;
  using size_type = std::size_t;
  using shape_type = std::vector<size_type>;
  using axes_type = std::vector<size_type>;
  using strides_type = std::vector<size_type>;
  using storage_type = std::vector<value_type>;

  explicit Tensor(shape_type shape) : shape_(std::move(shape)) {
    const size_type element_count = initialize_layout();
    storage_.resize(element_count);
  }

  Tensor(const Tensor&) = default;

  Tensor(Tensor&& other) noexcept
      : shape_(std::move(other.shape_)),
        strides_(std::move(other.strides_)),
        storage_(std::move(other.storage_)) {
    other.reset_to_empty_sentinel();
  }

  Tensor& operator=(const Tensor& other) {
    if (this == &other) {
      return *this;
    }

    Tensor temporary(other);

    swap(temporary);

    return *this;
  }

  Tensor& operator=(Tensor&& other) noexcept {
    if (this == &other) {
      return *this;
    }

    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    storage_ = std::move(other.storage_);

    other.reset_to_empty_sentinel();

    return *this;
  }

  void swap(Tensor& other) noexcept {
    std::swap(shape_, other.shape_);
    std::swap(strides_, other.strides_);
    std::swap(storage_, other.storage_);
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

  [[nodiscard]] static Tensor from_data(shape_type shape, storage_type data) {
    return Tensor(std::move(shape), std::move(data));
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

  void reshape(shape_type target_shape) & {
    validate_not_empty_sentinel();

    Layout target_layout = compute_layout(target_shape);

    if (target_layout.element_count != numel()) {
      throw std::invalid_argument(
          "new shape element count does not match tensor numel");
    }

    shape_.swap(target_shape);
    strides_.swap(target_layout.strides);
  }

  [[nodiscard]] Tensor permute(const axes_type& axes) const&
    requires std::copy_constructible<T>
  {
    validate_not_empty_sentinel();

    shape_type result_shape = compute_permuted_shape(axes);
    Layout result_layout = compute_layout(result_shape);

    assert(result_layout.element_count == numel());

    storage_type result_storage;
    result_storage.reserve(result_layout.element_count);

    for (size_type result_offset = 0;
         result_offset < result_layout.element_count; ++result_offset) {
      const size_type source_offset = map_result_offset_to_source(
          result_offset, axes, result_layout.strides);

      result_storage.emplace_back(storage_[source_offset]);
    }

    return Tensor(std::move(result_shape), std::move(result_layout),
                  std::move(result_storage));
  }

  [[nodiscard]] Tensor matrix_transpose() const&
    requires std::copy_constructible<T>
  {
    validate_not_empty_sentinel();

    const size_type axis_count = rank();

    if (axis_count < 2) {
      throw std::invalid_argument(
          "matrix transpose requires tensor rank of at least 2");
    }

    axes_type axes(axis_count);

    std::ranges::iota(axes, size_type{0});
    std::swap(axes[axis_count - 2], axes[axis_count - 1]);

    return permute(axes);
  }

  template <detail::tensor_index... IndexTypes>
  [[nodiscard]] value_type& operator[](IndexTypes... indices) & noexcept {
    return storage_[compute_offset(indices...)];
  }

  template <detail::tensor_index... IndexTypes>
  [[nodiscard]] const value_type& operator[](
      IndexTypes... indices) const& noexcept {
    return storage_[compute_offset(indices...)];
  }

  template <detail::tensor_index IndexType, std::size_t Extent>
  [[nodiscard]] value_type& operator[](
      std::span<IndexType, Extent> indices) & noexcept {
    return storage_[compute_offset(indices)];
  }

  template <detail::tensor_index IndexType, std::size_t Extent>
  [[nodiscard]] const value_type& operator[](
      std::span<IndexType, Extent> indices) const& noexcept {
    return storage_[compute_offset(indices)];
  }

  template <typename... Arguments>
  value_type& operator[](Arguments&&...) && = delete;

  template <typename... Arguments>
  const value_type& operator[](Arguments&&...) const&& = delete;

  template <detail::tensor_index... IndexTypes>
  [[nodiscard]] value_type& at(IndexTypes... indices) & {
    return storage_.at(compute_offset_checked(indices...));
  }

  template <detail::tensor_index... IndexTypes>
  [[nodiscard]] const value_type& at(IndexTypes... indices) const& {
    return storage_.at(compute_offset_checked(indices...));
  }

  template <detail::tensor_index IndexType, std::size_t Extent>
  [[nodiscard]] value_type& at(std::span<IndexType, Extent> indices) & {
    return storage_.at(compute_offset_checked(indices));
  }

  template <detail::tensor_index IndexType, std::size_t Extent>
  [[nodiscard]] const value_type& at(
      std::span<IndexType, Extent> indices) const& {
    return storage_.at(compute_offset_checked(indices));
  }

  template <typename... Arguments>
  value_type& at(Arguments&&...) && = delete;

  template <typename... Arguments>
  const value_type& at(Arguments&&...) const&& = delete;

  Tensor& operator+=(const Tensor& other) &
    requires std::floating_point<T>
  {
    apply_elementwise_inplace(other, std::plus<>{});

    return *this;
  }

  Tensor& operator+=(value_type value) &
    requires std::floating_point<T>
  {
    validate_not_empty_sentinel();

    std::ranges::transform(
        storage_, storage_.begin(),
        [value](value_type element) { return element + value; });

    return *this;
  }

  Tensor& operator-=(const Tensor& other) &
    requires std::floating_point<T>
  {
    apply_elementwise_inplace(other, std::minus<>{});

    return *this;
  }

  Tensor& operator-=(value_type value) &
    requires std::floating_point<T>
  {
    validate_not_empty_sentinel();

    std::ranges::transform(
        storage_, storage_.begin(),
        [value](value_type element) { return element - value; });

    return *this;
  }

  Tensor& operator*=(const Tensor& other) &
    requires std::floating_point<T>
  {
    apply_elementwise_inplace(other, std::multiplies<>{});

    return *this;
  }

  Tensor& operator*=(value_type value) &
    requires std::floating_point<T>
  {
    validate_not_empty_sentinel();

    std::ranges::transform(
        storage_, storage_.begin(),
        [value](value_type element) { return element * value; });

    return *this;
  }

  Tensor& operator/=(const Tensor& other) &
    requires std::floating_point<T>
  {
    apply_elementwise_inplace(other, std::divides<>{});

    return *this;
  }

  Tensor& operator/=(value_type value) &
    requires std::floating_point<T>
  {
    validate_not_empty_sentinel();

    std::ranges::transform(
        storage_, storage_.begin(),
        [value](value_type element) { return element / value; });

    return *this;
  }

 private:
  template <std::floating_point U>
  friend Tensor<U> operator+(Tensor<U>);

  template <std::floating_point U>
  friend Tensor<U> operator-(Tensor<U>);

  template <std::floating_point U>
  friend Tensor<U> operator+(Tensor<U>, const Tensor<U>&);

  template <std::floating_point U>
  friend Tensor<U> operator-(Tensor<U>, const Tensor<U>&);

  template <std::floating_point U>
  friend Tensor<U> operator-(typename Tensor<U>::value_type, Tensor<U>);

  template <std::floating_point U>
  friend Tensor<U> operator*(Tensor<U>, const Tensor<U>&);

  template <std::floating_point U>
  friend Tensor<U> operator/(Tensor<U>, const Tensor<U>&);

  template <std::floating_point U>
  friend Tensor<U> operator/(typename Tensor<U>::value_type, Tensor<U>);

  struct Layout {
    strides_type strides;
    size_type element_count;
  };

  Tensor(shape_type shape, storage_type data)
      : shape_(std::move(shape)), storage_(std::move(data)) {
    const size_type expected_element_count = initialize_layout();

    if (expected_element_count != storage_.size()) {
      throw std::invalid_argument("tensor data size does not match shape");
    }
  }

  Tensor(shape_type shape, Layout precomputed_layout, storage_type data)
      : shape_(std::move(shape)),
        strides_(std::move(precomputed_layout.strides)),
        storage_(std::move(data)) {
    assert(shape_.size() == strides_.size());
    assert(precomputed_layout.element_count == storage_.size());
  }

  [[nodiscard]] bool is_empty_sentinel() const noexcept {
    return shape_.empty() && strides_.empty() && storage_.empty();
  }

  void reset_to_empty_sentinel() noexcept {
    shape_.clear();
    strides_.clear();
    storage_.clear();
  }

  void validate_not_empty_sentinel() const {
    if (is_empty_sentinel()) {
      throw std::invalid_argument("tensor is an empty sentinel");
    }
  }

  template <typename BinaryOperation>
  void apply_elementwise_inplace(const Tensor& other,
                                 BinaryOperation operation) {
    validate_not_empty_sentinel();
    other.validate_not_empty_sentinel();

    if (shape_ == other.shape_) {
      std::ranges::transform(storage_, other.storage_, storage_.begin(),
                             operation);

      return;
    }

    const strides_type right_broadcast_strides =
        other.compute_broadcast_strides(shape_);
    const size_type axis_count = rank();

    for (size_type left_offset = 0; left_offset < numel(); ++left_offset) {
      size_type remaining_left_offset = left_offset;
      size_type right_offset = 0;

      for (size_type axis = 0; axis < axis_count; ++axis) {
        const size_type left_stride = strides_[axis];

        assert(left_stride != 0);

        const size_type left_index = remaining_left_offset / left_stride;

        remaining_left_offset %= left_stride;

        right_offset += left_index * right_broadcast_strides[axis];
      }

      assert(remaining_left_offset == 0);
      assert(right_offset < other.numel());

      storage_[left_offset] =
          operation(storage_[left_offset], other.storage_[right_offset]);
    }
  }

  template <typename BinaryOperation>
  [[nodiscard]] Tensor apply_elementwise(const Tensor& other,
                                         BinaryOperation operation) && {
    validate_not_empty_sentinel();
    other.validate_not_empty_sentinel();

    shape_type result_shape = compute_broadcast_shape(other.shape_);

    if (result_shape == shape_) {
      apply_elementwise_inplace(other, operation);

      return std::move(*this);
    }

    Layout result_layout = compute_layout(result_shape);
    const strides_type left_broadcast_strides =
        compute_broadcast_strides(result_shape);
    const strides_type right_broadcast_strides =
        other.compute_broadcast_strides(result_shape);
    const size_type axis_count = result_shape.size();

    storage_type result_storage;
    result_storage.reserve(result_layout.element_count);

    for (size_type result_offset = 0;
         result_offset < result_layout.element_count; ++result_offset) {
      size_type remaining_result_offset = result_offset;
      size_type left_offset = 0;
      size_type right_offset = 0;

      for (size_type result_axis = 0; result_axis < axis_count; ++result_axis) {
        const size_type result_stride = result_layout.strides[result_axis];

        assert(result_stride != 0);

        const size_type result_index = remaining_result_offset / result_stride;

        remaining_result_offset %= result_stride;

        left_offset += result_index * left_broadcast_strides[result_axis];
        right_offset += result_index * right_broadcast_strides[result_axis];
      }

      assert(remaining_result_offset == 0);
      assert(left_offset < numel());
      assert(right_offset < other.numel());

      result_storage.emplace_back(
          operation(storage_[left_offset], other.storage_[right_offset]));
    }

    return Tensor(std::move(result_shape), std::move(result_layout),
                  std::move(result_storage));
  }

  [[nodiscard]] strides_type compute_broadcast_strides(
      const shape_type& target_shape) const {
    const size_type source_rank = rank();
    const size_type target_rank = target_shape.size();

    if (source_rank > target_rank) {
      throw std::invalid_argument(
          "target tensor rank is less than source tensor rank");
    }

    strides_type broadcast_strides(target_rank, size_type{0});
    const size_type rank_difference = target_rank - source_rank;

    for (size_type source_axis = 0; source_axis < source_rank; ++source_axis) {
      const size_type target_axis = rank_difference + source_axis;
      const size_type source_extent = shape_[source_axis];
      const size_type target_extent = target_shape[target_axis];

      if (source_extent == 1) {
        continue;
      }

      if (source_extent != target_extent) {
        throw std::invalid_argument(
            "source tensor shape is not broadcast-compatible with target "
            "shape");
      }

      broadcast_strides[target_axis] = strides_[source_axis];
    }

    return broadcast_strides;
  }

  [[nodiscard]] shape_type compute_broadcast_shape(
      const shape_type& right_shape) const {
    const size_type left_rank = rank();
    const size_type right_rank = right_shape.size();
    const size_type result_rank = std::max(left_rank, right_rank);

    const size_type left_rank_difference = result_rank - left_rank;
    const size_type right_rank_difference = result_rank - right_rank;

    shape_type result_shape(result_rank);

    for (size_type result_axis = 0; result_axis < result_rank; ++result_axis) {
      size_type left_extent = 1;
      size_type right_extent = 1;
      size_type result_extent = 0;

      if (result_axis >= left_rank_difference) {
        const size_type left_axis = result_axis - left_rank_difference;
        left_extent = shape_[left_axis];
      }

      if (result_axis >= right_rank_difference) {
        const size_type right_axis = result_axis - right_rank_difference;
        right_extent = right_shape[right_axis];
      }

      if (left_extent == right_extent) {
        result_extent = left_extent;
      } else if (left_extent == 1) {
        result_extent = right_extent;
      } else if (right_extent == 1) {
        result_extent = left_extent;
      } else {
        throw std::invalid_argument(
            "tensor shapes are not broadcast-compatible");
      }

      result_shape[result_axis] = result_extent;
    }

    return result_shape;
  }

  [[nodiscard]] shape_type compute_permuted_shape(const axes_type& axes) const {
    const size_type axis_count = rank();

    if (axes.size() != axis_count) {
      throw std::invalid_argument(
          "permutation axis count does not match tensor rank");
    }

    shape_type result_shape;
    result_shape.reserve(axis_count);

    std::vector<bool> seen_source_axes(axis_count, false);

    for (size_type result_axis = 0; result_axis < axis_count; ++result_axis) {
      const size_type source_axis = axes[result_axis];

      if (source_axis >= axis_count) {
        throw std::out_of_range("permutation axis is out of range");
      }

      if (seen_source_axes[source_axis]) {
        throw std::invalid_argument("permutation axes contain duplicates");
      }

      seen_source_axes[source_axis] = true;
      result_shape.push_back(shape_[source_axis]);
    }

    return result_shape;
  }

  [[nodiscard]] size_type map_result_offset_to_source(
      size_type result_offset, const axes_type& axes,
      const strides_type& result_strides) const noexcept {
    const size_type axis_count = rank();

    assert(axes.size() == axis_count);
    assert(result_strides.size() == axis_count);
    assert(result_offset < numel());

    size_type remaining_result_offset = result_offset;
    size_type source_offset = 0;

    for (size_type result_axis = 0; result_axis < axis_count; ++result_axis) {
      const size_type result_stride = result_strides[result_axis];

      assert(result_stride != 0);

      const size_type result_index = remaining_result_offset / result_stride;
      const size_type source_axis = axes[result_axis];

      assert(source_axis < axis_count);
      assert(result_index < shape_[source_axis]);

      remaining_result_offset %= result_stride;
      source_offset += result_index * strides_[source_axis];
    }

    assert(remaining_result_offset == 0);
    assert(source_offset < numel());

    return source_offset;
  }

  template <detail::tensor_index IndexType>
  [[nodiscard]] size_type compute_axis_offset(size_type axis,
                                              IndexType index) const noexcept {
    assert(axis < rank());
    assert(std::in_range<size_type>(index));

    const size_type normalized_index = static_cast<size_type>(index);

    assert(normalized_index < shape_[axis]);

    return normalized_index * strides_[axis];
  }

  template <detail::tensor_index... IndexTypes>
  [[nodiscard]] size_type compute_offset(IndexTypes... indices) const noexcept {
    assert(sizeof...(IndexTypes) == rank());

    size_type axis = 0;
    size_type offset = 0;

    ((offset += compute_axis_offset(axis, indices), ++axis), ...);

    return offset;
  }

  template <detail::tensor_index IndexType, std::size_t Extent>
  [[nodiscard]] size_type compute_offset(
      std::span<IndexType, Extent> indices) const noexcept {
    assert(indices.size() == rank());

    size_type offset = 0;

    for (size_type axis = 0; axis < indices.size(); ++axis) {
      offset += compute_axis_offset(axis, indices[axis]);
    }

    return offset;
  }

  template <detail::tensor_index IndexType>
  [[nodiscard]] size_type compute_axis_offset_checked(size_type axis,
                                                      IndexType index) const {
    assert(axis < rank());

    if (!std::in_range<size_type>(index)) {
      throw std::out_of_range("tensor index is out of bounds");
    }

    const size_type normalized_index = static_cast<size_type>(index);

    if (normalized_index >= shape_[axis]) {
      throw std::out_of_range("tensor index is out of bounds");
    }

    return normalized_index * strides_[axis];
  }

  template <detail::tensor_index... IndexTypes>
  [[nodiscard]] size_type compute_offset_checked(IndexTypes... indices) const {
    if (sizeof...(IndexTypes) != rank()) {
      throw std::invalid_argument("tensor index count does not match rank");
    }

    size_type axis = 0;
    size_type offset = 0;

    ((offset += compute_axis_offset_checked(axis, indices), ++axis), ...);

    return offset;
  }

  template <detail::tensor_index IndexType, std::size_t Extent>
  [[nodiscard]] size_type compute_offset_checked(
      std::span<IndexType, Extent> indices) const {
    if (indices.size() != rank()) {
      throw std::invalid_argument("tensor index count does not match rank");
    }

    size_type offset = 0;

    for (size_type axis = 0; axis < indices.size(); ++axis) {
      offset += compute_axis_offset_checked(axis, indices[axis]);
    }

    return offset;
  }

  [[nodiscard]] Layout compute_layout(const shape_type& shape) const {
    strides_type strides(shape.size());
    size_type running_stride = 1;

    for (size_type remaining_axes = shape.size(); remaining_axes != 0;
         --remaining_axes) {
      const size_type axis = remaining_axes - 1;
      const size_type extent = shape[axis];

      strides[axis] = running_stride;
      running_stride = checked_multiply(running_stride, extent);
    }

    if (running_stride > storage_.max_size()) {
      throw std::length_error("tensor numel exceeds storage max_size");
    }

    return Layout{std::move(strides), running_stride};
  }

  [[nodiscard]] size_type initialize_layout() {
    Layout computed_layout = compute_layout(shape_);

    strides_.swap(computed_layout.strides);

    return computed_layout.element_count;
  }

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

template <typename T>
void swap(Tensor<T>& lhs, Tensor<T>& rhs) noexcept {
  lhs.swap(rhs);
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> operator+(Tensor<T> tensor) {
  tensor.validate_not_empty_sentinel();

  return tensor;
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> operator-(Tensor<T> tensor) {
  tensor.validate_not_empty_sentinel();

  std::ranges::transform(tensor.storage_, tensor.storage_.begin(),
                         std::negate<>{});

  return tensor;
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> operator+(Tensor<T> lhs, const Tensor<T>& rhs) {
  return std::move(lhs).apply_elementwise(rhs, std::plus<>{});
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> operator+(Tensor<T> tensor,
                                  typename Tensor<T>::value_type value) {
  tensor += value;

  return tensor;
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> operator+(typename Tensor<T>::value_type value,
                                  Tensor<T> tensor) {
  tensor += value;

  return tensor;
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> operator-(Tensor<T> lhs, const Tensor<T>& rhs) {
  return std::move(lhs).apply_elementwise(rhs, std::minus<>{});
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> operator-(Tensor<T> tensor,
                                  typename Tensor<T>::value_type value) {
  tensor -= value;

  return tensor;
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> operator-(typename Tensor<T>::value_type value,
                                  Tensor<T> tensor) {
  tensor.validate_not_empty_sentinel();

  std::ranges::transform(tensor.storage_, tensor.storage_.begin(),
                         [value](typename Tensor<T>::value_type element) {
                           return value - element;
                         });

  return tensor;
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> operator*(Tensor<T> lhs, const Tensor<T>& rhs) {
  return std::move(lhs).apply_elementwise(rhs, std::multiplies<>{});
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> operator*(Tensor<T> tensor,
                                  typename Tensor<T>::value_type value) {
  tensor *= value;

  return tensor;
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> operator*(typename Tensor<T>::value_type value,
                                  Tensor<T> tensor) {
  tensor *= value;

  return tensor;
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> operator/(Tensor<T> lhs, const Tensor<T>& rhs) {
  return std::move(lhs).apply_elementwise(rhs, std::divides<>{});
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> operator/(Tensor<T> tensor,
                                  typename Tensor<T>::value_type value) {
  tensor /= value;

  return tensor;
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> operator/(typename Tensor<T>::value_type value,
                                  Tensor<T> tensor) {
  tensor.validate_not_empty_sentinel();

  std::ranges::transform(tensor.storage_, tensor.storage_.begin(),
                         [value](typename Tensor<T>::value_type element) {
                           return value / element;
                         });

  return tensor;
}

}  // namespace nn
