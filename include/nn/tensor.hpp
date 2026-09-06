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
    other.shape_.clear();
    other.strides_.clear();
    other.storage_.clear();
  }

  Tensor& operator=(const Tensor& other) {
    if (this == &other) {
      return *this;
    }

    Tensor temporary(other);

    std::swap(shape_, temporary.shape_);
    std::swap(strides_, temporary.strides_);
    std::swap(storage_, temporary.storage_);

    return *this;
  }

  Tensor& operator=(Tensor&& other) noexcept {
    if (this == &other) {
      return *this;
    }

    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    storage_ = std::move(other.storage_);

    other.shape_.clear();
    other.strides_.clear();
    other.storage_.clear();

    return *this;
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

  void reshape(shape_type new_shape) & {
    validate_not_empty_sentinel();

    Layout layout = compute_layout(new_shape);

    if (layout.element_count != storage_.size()) {
      throw std::invalid_argument(
          "new shape element count does not match tensor numel");
    }

    shape_.swap(new_shape);
    strides_.swap(layout.strides);
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
    validate_elementwise_compatibility(other);

    std::ranges::transform(storage_, other.storage_, storage_.begin(),
                           std::plus<>{});

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
    validate_elementwise_compatibility(other);

    std::ranges::transform(storage_, other.storage_, storage_.begin(),
                           std::minus<>{});

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
    validate_elementwise_compatibility(other);

    std::ranges::transform(storage_, other.storage_, storage_.begin(),
                           std::multiplies<>{});

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
    validate_elementwise_compatibility(other);

    std::ranges::transform(storage_, other.storage_, storage_.begin(),
                           std::divides<>{});

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
  friend Tensor<U> operator-(typename Tensor<U>::value_type, Tensor<U>);

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

  [[nodiscard]] bool is_empty_sentinel() const noexcept {
    return shape_.empty() && strides_.empty() && storage_.empty();
  }

  void validate_not_empty_sentinel() const {
    if (is_empty_sentinel()) {
      throw std::invalid_argument("tensor is an empty sentinel");
    }
  }

  void validate_elementwise_compatibility(const Tensor& other) const {
    validate_not_empty_sentinel();
    other.validate_not_empty_sentinel();

    if (shape_ != other.shape_) {
      throw std::invalid_argument("operands have different shapes");
    }
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
    Layout layout = compute_layout(shape_);

    strides_.swap(layout.strides);

    return layout.element_count;
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
  lhs += rhs;

  return lhs;
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
  lhs -= rhs;

  return lhs;
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
  lhs *= rhs;

  return lhs;
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
  lhs /= rhs;

  return lhs;
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
