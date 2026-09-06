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

  Tensor& operator-=(const Tensor& other) &
    requires std::floating_point<T>
  {
    validate_elementwise_compatibility(other);

    std::ranges::transform(storage_, other.storage_, storage_.begin(),
                           std::minus<>{});

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

 private:
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

  void validate_elementwise_compatibility(const Tensor& other) const {
    if (is_empty_sentinel() || other.is_empty_sentinel()) {
      throw std::invalid_argument("one of the operands is an empty sentinel");
    }

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

  [[nodiscard]] size_type initialize_layout() {
    strides_.resize(shape_.size());

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

    return running_stride;
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
[[nodiscard]] Tensor<T> operator+(Tensor<T> lhs, const Tensor<T>& rhs) {
  lhs += rhs;

  return lhs;
}

template <std::floating_point T>
[[nodiscard]] Tensor<T> operator-(Tensor<T> lhs, const Tensor<T>& rhs) {
  lhs -= rhs;

  return lhs;
}

}  // namespace nn
