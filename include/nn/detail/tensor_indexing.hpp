#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

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

template <tensor_index IndexType>
[[nodiscard]] std::size_t compute_tensor_axis_offset(
    std::span<const std::size_t> shape, std::span<const std::size_t> strides,
    std::size_t axis, IndexType index) noexcept {
  assert(shape.size() == strides.size());
  assert(axis < shape.size());
  assert(std::in_range<std::size_t>(index));

  const std::size_t normalized_index = static_cast<std::size_t>(index);

  assert(normalized_index < shape[axis]);

  return normalized_index * strides[axis];
}

template <tensor_index... IndexTypes>
[[nodiscard]] std::size_t compute_tensor_offset(
    std::span<const std::size_t> shape, std::span<const std::size_t> strides,
    IndexTypes... indices) noexcept {
  assert(shape.size() == strides.size());
  assert(sizeof...(IndexTypes) == shape.size());

  std::size_t axis = 0;
  std::size_t offset = 0;

  ((offset += compute_tensor_axis_offset(shape, strides, axis, indices),
    ++axis),
   ...);

  return offset;
}

template <tensor_index IndexType, std::size_t Extent>
[[nodiscard]] std::size_t compute_tensor_offset(
    std::span<const std::size_t> shape, std::span<const std::size_t> strides,
    std::span<IndexType, Extent> indices) noexcept {
  assert(shape.size() == strides.size());
  assert(indices.size() == shape.size());

  std::size_t offset = 0;

  for (std::size_t axis = 0; axis < indices.size(); ++axis) {
    offset += compute_tensor_axis_offset(shape, strides, axis, indices[axis]);
  }

  return offset;
}

template <tensor_index IndexType>
[[nodiscard]] std::size_t compute_tensor_axis_offset_checked(
    std::span<const std::size_t> shape, std::span<const std::size_t> strides,
    std::size_t axis, IndexType index) {
  assert(shape.size() == strides.size());
  assert(axis < shape.size());

  if (!std::in_range<std::size_t>(index)) {
    throw std::out_of_range("tensor index is out of bounds");
  }

  const std::size_t normalized_index = static_cast<std::size_t>(index);

  if (normalized_index >= shape[axis]) {
    throw std::out_of_range("tensor index is out of bounds");
  }

  return normalized_index * strides[axis];
}

template <tensor_index... IndexTypes>
[[nodiscard]] std::size_t compute_tensor_offset_checked(
    std::span<const std::size_t> shape, std::span<const std::size_t> strides,
    IndexTypes... indices) {
  assert(shape.size() == strides.size());

  if (sizeof...(IndexTypes) != shape.size()) {
    throw std::invalid_argument("tensor index count does not match rank");
  }

  std::size_t axis = 0;
  std::size_t offset = 0;

  ((offset += compute_tensor_axis_offset_checked(shape, strides, axis, indices),
    ++axis),
   ...);

  return offset;
}

template <tensor_index IndexType, std::size_t Extent>
[[nodiscard]] std::size_t compute_tensor_offset_checked(
    std::span<const std::size_t> shape, std::span<const std::size_t> strides,
    std::span<IndexType, Extent> indices) {
  assert(shape.size() == strides.size());

  if (indices.size() != shape.size()) {
    throw std::invalid_argument("tensor index count does not match rank");
  }

  std::size_t offset = 0;

  for (std::size_t axis = 0; axis < indices.size(); ++axis) {
    offset +=
        compute_tensor_axis_offset_checked(shape, strides, axis, indices[axis]);
  }

  return offset;
}

}  // namespace nn::detail
