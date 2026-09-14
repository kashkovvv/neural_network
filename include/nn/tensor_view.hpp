#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "nn/detail/tensor_indexing.hpp"

namespace nn::detail {

template <typename Element>
concept tensor_view_element =
    std::is_object_v<Element> && (!std::is_array_v<Element>) &&
    (!std::is_volatile_v<Element>) && requires { sizeof(Element); } &&
    (!std::is_abstract_v<Element>);

}  // namespace nn::detail

namespace nn {

template <typename T>
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
    assert(shape_.size() == strides_.size());
    assert(origin_offset_ <= storage_.size());
    assert(element_count_ == 0 || origin_offset_ < storage_.size());
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
