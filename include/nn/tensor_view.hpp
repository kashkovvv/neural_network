#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "nn/detail/tensor_concepts.hpp"
#include "nn/detail/tensor_indexing.hpp"

namespace nn {

template <detail::tensor_element T>
class Tensor;

template <detail::tensor_view_element Element>
class TensorView {
 public:
  using element_type = Element;
  using value_type = std::remove_cv_t<element_type>;
  using size_type = std::size_t;
  using shape_type = std::vector<size_type>;
  using strides_type = std::vector<size_type>;
  using reference = element_type&;

  TensorView() = delete;

  TensorView(const TensorView&) = default;

  TensorView(TensorView&& other) noexcept
      : storage_(other.storage_),
        shape_(std::move(other.shape_)),
        strides_(std::move(other.strides_)),
        origin_offset_(other.origin_offset_),
        element_count_(other.element_count_),
        is_contiguous_(other.is_contiguous_) {
    other.reset_to_empty_sentinel();
  }

  TensorView& operator=(const TensorView& other) {
    if (this == &other) {
      return *this;
    }

    TensorView temporary(other);

    swap(temporary);

    return *this;
  }

  TensorView& operator=(TensorView&& other) noexcept {
    if (this == &other) {
      return *this;
    }

    storage_ = other.storage_;
    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    origin_offset_ = other.origin_offset_;
    element_count_ = other.element_count_;
    is_contiguous_ = other.is_contiguous_;

    other.reset_to_empty_sentinel();

    return *this;
  }

  template <detail::tensor_view_element OtherElement>
    requires std::convertible_to<OtherElement (*)[], element_type (*)[]>
  TensorView(const TensorView<OtherElement>& other)
      : storage_(other.storage_),
        shape_(other.shape_),
        strides_(other.strides_),
        origin_offset_(other.origin_offset_),
        element_count_(other.element_count_),
        is_contiguous_(other.is_contiguous_) {}

  void swap(TensorView& other) noexcept {
    std::swap(storage_, other.storage_);
    std::swap(shape_, other.shape_);
    std::swap(strides_, other.strides_);
    std::swap(origin_offset_, other.origin_offset_);
    std::swap(element_count_, other.element_count_);
    std::swap(is_contiguous_, other.is_contiguous_);
  }

  [[nodiscard]] size_type rank() const noexcept { return shape_.size(); }

  [[nodiscard]] size_type numel() const noexcept { return element_count_; }

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

  [[nodiscard]] bool is_contiguous() const noexcept { return is_contiguous_; }

  [[nodiscard]] TensorView slice(size_type axis, size_type start,
                                 size_type stop, size_type step = 1) const {
    validate_not_empty_sentinel();

    if (axis >= rank()) {
      throw std::out_of_range("slice axis is out of range");
    }

    if (start > stop) {
      throw std::invalid_argument("slice start exceeds stop");
    }

    if (stop > shape_[axis]) {
      throw std::out_of_range("slice stop is out of range");
    }

    if (step == 0) {
      throw std::invalid_argument("slice step must be positive");
    }

    shape_type result_shape = shape_;
    const size_type distance = stop - start;
    const size_type result_extent =
        distance == 0 ? size_type{0} : 1 + (distance - 1) / step;
    result_shape[axis] = result_extent;

    strides_type result_strides = strides_;

    if (result_extent > 1) {
      result_strides[axis] *= step;
    }

    assert(element_count_ == 0 || shape_[axis] != 0);

    const size_type result_element_count =
        element_count_ == 0 ? size_type{0}
                            : element_count_ / shape_[axis] * result_extent;
    const size_type result_origin_offset =
        result_element_count == 0 ? size_type{0}
                                  : origin_offset_ + start * strides_[axis];
    const bool result_is_contiguous =
        compute_contiguity(result_shape, result_strides, result_element_count);

    return TensorView(storage_, std::move(result_shape),
                      std::move(result_strides), result_origin_offset,
                      result_element_count, result_is_contiguous);
  }

  [[nodiscard]] Tensor<value_type> to_tensor() const
    requires std::copy_constructible<value_type>;

  template <detail::tensor_index... IndexTypes>
  [[nodiscard]] reference operator[](IndexTypes... indices) const noexcept {
    assert(!is_empty_sentinel());

    const size_type view_offset =
        detail::compute_tensor_offset(shape_, strides_, indices...);

    return storage_[compute_storage_offset(view_offset)];
  }

  template <detail::tensor_index IndexType, std::size_t Extent>
  [[nodiscard]] reference operator[](
      std::span<IndexType, Extent> indices) const noexcept {
    assert(!is_empty_sentinel());

    const size_type view_offset =
        detail::compute_tensor_offset(shape_, strides_, indices);

    return storage_[compute_storage_offset(view_offset)];
  }

  template <detail::tensor_index... IndexTypes>
  [[nodiscard]] reference at(IndexTypes... indices) const {
    validate_not_empty_sentinel();

    const size_type view_offset =
        detail::compute_tensor_offset_checked(shape_, strides_, indices...);

    return storage_[compute_storage_offset(view_offset)];
  }

  template <detail::tensor_index IndexType, std::size_t Extent>
  [[nodiscard]] reference at(std::span<IndexType, Extent> indices) const {
    validate_not_empty_sentinel();

    const size_type view_offset =
        detail::compute_tensor_offset_checked(shape_, strides_, indices);

    return storage_[compute_storage_offset(view_offset)];
  }

 private:
  friend class Tensor<value_type>;

  template <detail::tensor_view_element>
  friend class TensorView;

  TensorView(std::span<element_type> storage, shape_type shape,
             strides_type strides, size_type origin_offset,
             size_type element_count, bool is_contiguous)
      : storage_(storage),
        shape_(std::move(shape)),
        strides_(std::move(strides)),
        origin_offset_(origin_offset),
        element_count_(element_count),
        is_contiguous_(is_contiguous) {
    assert(has_valid_layout());
    assert(is_contiguous_ ==
           compute_contiguity(shape_, strides_, element_count_));
  }

  [[nodiscard]] static bool compute_contiguity(
      std::span<const size_type> shape, std::span<const size_type> strides,
      size_type element_count) noexcept {
    assert(shape.size() == strides.size());

    if (element_count == 0) {
      return true;
    }

    size_type expected_stride = 1;

    for (size_type remaining_axes = shape.size(); remaining_axes != 0;
         --remaining_axes) {
      const size_type axis = remaining_axes - 1;
      const size_type extent = shape[axis];

      if (extent == 1) {
        continue;
      }

      if (strides[axis] != expected_stride) {
        return false;
      }

      expected_stride *= extent;
    }

    return true;
  }

  [[nodiscard]] bool has_valid_layout() const noexcept {
    if (shape_.size() != strides_.size()) {
      return false;
    }

    size_type expected_element_count = 1;
    const size_type max_size = std::numeric_limits<size_type>::max();

    for (const size_type extent : shape_) {
      if (extent != 0 && expected_element_count > max_size / extent) {
        return false;
      }

      expected_element_count *= extent;
    }

    if (expected_element_count != element_count_) {
      return false;
    }

    if (element_count_ == 0) {
      return origin_offset_ == 0;
    }

    if (origin_offset_ >= storage_.size()) {
      return false;
    }

    size_type remaining_storage = storage_.size() - 1 - origin_offset_;

    for (size_type axis = 0; axis < shape_.size(); ++axis) {
      const size_type maximum_index = shape_[axis] - 1;
      const size_type stride = strides_[axis];

      if (stride != 0 && maximum_index > remaining_storage / stride) {
        return false;
      }

      remaining_storage -= maximum_index * stride;
    }

    return true;
  }

  [[nodiscard]] bool is_empty_sentinel() const noexcept {
    return storage_.empty() && shape_.empty() && strides_.empty() &&
           origin_offset_ == 0 && element_count_ == 0 && is_contiguous_;
  }

  void validate_not_empty_sentinel() const {
    if (is_empty_sentinel()) {
      throw std::invalid_argument("operation on moved-from tensor view");
    }
  }

  [[nodiscard]] size_type compute_storage_offset(
      size_type view_offset) const noexcept {
    assert(origin_offset_ <= storage_.size());
    assert(view_offset < storage_.size() - origin_offset_);

    return origin_offset_ + view_offset;
  }

  [[nodiscard]] size_type map_logical_offset_to_storage_offset(
      size_type logical_offset) const noexcept {
    assert(logical_offset < numel());

    size_type remaining_logical_offset = logical_offset;
    size_type view_offset = 0;

    for (size_type remaining_axes = rank(); remaining_axes != 0;
         --remaining_axes) {
      const size_type axis = remaining_axes - 1;
      const size_type extent = shape_[axis];

      assert(extent != 0);

      const size_type index = remaining_logical_offset % extent;

      remaining_logical_offset /= extent;
      view_offset += index * strides_[axis];
    }

    assert(remaining_logical_offset == 0);

    return compute_storage_offset(view_offset);
  }

  void reset_to_empty_sentinel() noexcept {
    storage_ = std::span<element_type>{};
    shape_.clear();
    strides_.clear();
    origin_offset_ = 0;
    element_count_ = 0;
    is_contiguous_ = true;
  }

  std::span<element_type> storage_;
  shape_type shape_;
  strides_type strides_;
  size_type origin_offset_;
  size_type element_count_;
  bool is_contiguous_;
};

template <detail::tensor_view_element Element>
void swap(TensorView<Element>& lhs, TensorView<Element>& rhs) noexcept {
  lhs.swap(rhs);
}

}  // namespace nn
