#include <algorithm>
#include <concepts>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "nn/tensor.hpp"

namespace {

using Tensor = nn::Tensor<float>;
using MutableView = nn::TensorView<float>;
using ConstView = nn::TensorView<const float>;

template <typename ViewType>
concept CanSlice = requires {
  std::declval<ViewType>().slice(Tensor::size_type{0}, Tensor::size_type{0},
                                 Tensor::size_type{0});
};

static_assert(CanSlice<MutableView&>);
static_assert(CanSlice<const MutableView&>);
static_assert(CanSlice<MutableView&&>);
static_assert(CanSlice<const MutableView&&>);
static_assert(CanSlice<ConstView&>);
static_assert(CanSlice<const ConstView&>);
static_assert(CanSlice<ConstView&&>);
static_assert(CanSlice<const ConstView&&>);

static_assert(
    std::same_as<decltype(std::declval<const MutableView&>().slice(0, 0, 0)),
                 MutableView>);
static_assert(
    std::same_as<decltype(std::declval<const ConstView&>().slice(0, 0, 0)),
                 ConstView>);
static_assert(!noexcept(std::declval<const MutableView&>().slice(0, 0, 0)));
static_assert(!noexcept(std::declval<const ConstView&>().slice(0, 0, 0)));

void expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename ExpectedException, typename Function>
void expect_throws(Function&& function, const char* missing_exception_message,
                   const char* wrong_exception_message) {
  try {
    std::forward<Function>(function)();
  } catch (const ExpectedException&) {
    return;
  } catch (...) {
    throw std::runtime_error(wrong_exception_message);
  }

  throw std::runtime_error(missing_exception_message);
}

void test_leading_axis_slice_is_contiguous() {
  Tensor tensor =
      Tensor::from_data({3, 4}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F,
                                 8.0F, 9.0F, 10.0F, 11.0F});
  MutableView source = tensor.view();
  MutableView result = source.slice(0, 1, 3);

  const Tensor::shape_type expected_shape{2, 4};
  const Tensor::strides_type expected_strides{4, 1};

  expect(result.rank() == 2, "leading-axis slice rank is incorrect");
  expect(result.numel() == 8, "leading-axis slice numel is incorrect");
  expect(std::ranges::equal(result.shape(), expected_shape),
         "leading-axis slice shape is incorrect");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "leading-axis slice strides are incorrect");
  expect(result.is_contiguous(), "complete-row slice must remain contiguous");
  expect(result.at(0, 0) == 4.0F && result.at(1, 3) == 11.0F,
         "leading-axis slice returned incorrect elements");

  result.at(0, 2) = 60.0F;

  expect(tensor.at(1, 2) == 60.0F,
         "leading-axis slice does not alias tensor storage");
}

void test_trailing_axis_slice_is_non_contiguous() {
  Tensor tensor =
      Tensor::from_data({3, 4}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F,
                                 8.0F, 9.0F, 10.0F, 11.0F});
  MutableView result = tensor.view().slice(1, 1, 3);

  const Tensor::shape_type expected_shape{3, 2};
  const Tensor::strides_type expected_strides{4, 1};

  expect(result.numel() == 6, "trailing-axis slice numel is incorrect");
  expect(std::ranges::equal(result.shape(), expected_shape),
         "trailing-axis slice shape is incorrect");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "trailing-axis slice strides are incorrect");
  expect(!result.is_contiguous(),
         "partial-column slice must be non-contiguous");
  expect(result.at(0, 0) == 1.0F && result.at(1, 1) == 6.0F &&
             result.at(2, 0) == 9.0F,
         "trailing-axis slice returned incorrect elements");

  tensor.at(2, 2) = 100.0F;

  expect(result.at(2, 1) == 100.0F,
         "tensor write is not visible through trailing-axis slice");
}

void test_nested_slice_accumulates_origin_offset() {
  Tensor tensor =
      Tensor::from_data({3, 4}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F,
                                 8.0F, 9.0F, 10.0F, 11.0F});
  MutableView result = tensor.view().slice(1, 1, 4).slice(0, 1, 3);

  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{4, 1};

  expect(std::ranges::equal(result.shape(), expected_shape),
         "nested slice shape is incorrect");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "nested slice strides are incorrect");
  expect(!result.is_contiguous(), "nested slice contiguity is incorrect");
  expect(result.at(0, 0) == 5.0F && result.at(1, 2) == 11.0F,
         "nested slice did not accumulate its origin offset");
}

void test_slicing_can_restore_contiguity() {
  Tensor tensor =
      Tensor::from_data({3, 4}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F,
                                 8.0F, 9.0F, 10.0F, 11.0F});
  MutableView columns = tensor.view().slice(1, 1, 3);
  MutableView row = columns.slice(0, 1, 2);

  const Tensor::shape_type expected_shape{1, 2};

  expect(!columns.is_contiguous(),
         "source column slice must be non-contiguous");
  expect(std::ranges::equal(row.shape(), expected_shape),
         "single-row subslice shape is incorrect");
  expect(row.is_contiguous(), "single-row subslice must restore contiguity");
  expect(row.at(0, 0) == 5.0F && row.at(0, 1) == 6.0F,
         "single-row subslice returned incorrect elements");
}

void test_full_range_slice_preserves_layout() {
  Tensor tensor =
      Tensor::from_data({2, 3}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
  MutableView source = tensor.view();
  MutableView result = source.slice(1, 0, 3);

  expect(std::ranges::equal(result.shape(), source.shape()),
         "full-range slice changed shape");
  expect(std::ranges::equal(result.strides(), source.strides()),
         "full-range slice changed strides");
  expect(result.numel() == source.numel(), "full-range slice changed numel");
  expect(result.is_contiguous() == source.is_contiguous(),
         "full-range slice changed contiguity");
  expect(result.at(1, 2) == 5.0F,
         "full-range slice returned an incorrect element");
}

void test_empty_slice_is_valid_and_contiguous() {
  Tensor tensor =
      Tensor::from_data({2, 3}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
  MutableView result = tensor.view().slice(1, 2, 2);

  const Tensor::shape_type expected_shape{2, 0};
  const Tensor::strides_type expected_strides{3, 1};

  expect(result.rank() == 2, "empty slice rank is incorrect");
  expect(result.numel() == 0, "empty slice numel must be zero");
  expect(std::ranges::equal(result.shape(), expected_shape),
         "empty slice shape is incorrect");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "empty slice strides are incorrect");
  expect(result.is_contiguous(), "empty slice must be contiguous");

  expect_throws<std::out_of_range>(
      [&result] { static_cast<void>(result.at(0, 0)); },
      "empty slice access did not throw std::out_of_range",
      "empty slice access produced the wrong exception type");
}

void test_zero_extent_source_can_be_sliced() {
  Tensor tensor({2, 0, 4});
  MutableView result = tensor.view().slice(2, 4, 4).slice(1, 0, 0);

  const Tensor::shape_type expected_shape{2, 0, 0};
  const Tensor::strides_type expected_strides{0, 4, 1};

  expect(result.numel() == 0, "zero-extent subslice numel must be zero");
  expect(std::ranges::equal(result.shape(), expected_shape),
         "zero-extent subslice shape is incorrect");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "zero-extent subslice strides are incorrect");
  expect(result.is_contiguous(), "zero-extent subslice must be contiguous");

  expect_throws<std::out_of_range>(
      [&result] { static_cast<void>(result.at(0, 0, 0)); },
      "zero-extent subslice access did not throw std::out_of_range",
      "zero-extent subslice access produced the wrong exception type");
}

void test_const_view_slice_preserves_element_constness() {
  Tensor tensor =
      Tensor::from_data({2, 3}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
  const Tensor& const_tensor = tensor;
  ConstView result = const_tensor.view().slice(1, 1, 3);

  expect(result.at(1, 0) == 4.0F,
         "const view slice returned an incorrect element");

  tensor.at(1, 1) = 40.0F;

  expect(result.at(1, 0) == 40.0F,
         "tensor write is not visible through const view slice");
}

void test_const_mutable_view_handle_preserves_mutability() {
  Tensor tensor = Tensor::from_data({3}, {1.0F, 2.0F, 3.0F});
  const MutableView source = tensor.view();
  MutableView result = source.slice(0, 1, 3);

  result.at(0) = 20.0F;

  expect(tensor.at(1) == 20.0F,
         "const mutable-view handle made slice elements const");
}

void test_slice_rejects_invalid_arguments_without_mutation() {
  Tensor tensor =
      Tensor::from_data({2, 3}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
  MutableView source = tensor.view();
  const Tensor::shape_type expected_shape(source.shape().begin(),
                                          source.shape().end());
  const Tensor::strides_type expected_strides(source.strides().begin(),
                                              source.strides().end());

  expect_throws<std::out_of_range>(
      [&source] { static_cast<void>(source.slice(2, 0, 0)); },
      "slice accepted an axis equal to rank",
      "invalid slice axis produced the wrong exception type");

  expect_throws<std::invalid_argument>(
      [&source] { static_cast<void>(source.slice(1, 2, 1)); },
      "slice accepted a reversed range",
      "reversed slice range produced the wrong exception type");

  expect_throws<std::out_of_range>(
      [&source] { static_cast<void>(source.slice(1, 0, 4)); },
      "slice accepted stop beyond the source extent",
      "out-of-range slice stop produced the wrong exception type");

  expect(std::ranges::equal(source.shape(), expected_shape),
         "failed slice changed source shape");
  expect(std::ranges::equal(source.strides(), expected_strides),
         "failed slice changed source strides");
  expect(source.numel() == 6, "failed slice changed source numel");
  expect(source.is_contiguous(), "failed slice changed source contiguity");
  expect(source.at(1, 2) == 5.0F,
         "failed slice changed source storage binding");
}

void test_scalar_and_moved_from_views_reject_slice() {
  Tensor scalar = Tensor::scalar(1.0F);
  MutableView scalar_view = scalar.view();

  expect_throws<std::out_of_range>(
      [&scalar_view] { static_cast<void>(scalar_view.slice(0, 0, 0)); },
      "scalar view accepted a slice axis",
      "scalar view slice produced the wrong exception type");

  Tensor tensor = Tensor::from_data({2}, {1.0F, 2.0F});
  MutableView source = tensor.view();
  MutableView destination(std::move(source));
  static_cast<void>(destination);

  expect_throws<std::invalid_argument>(
      [&source] { static_cast<void>(source.slice(0, 0, 0)); },
      "moved-from view accepted slicing",
      "moved-from view slicing produced the wrong exception type");
}

}  // namespace

int main() {
  try {
    test_leading_axis_slice_is_contiguous();
    test_trailing_axis_slice_is_non_contiguous();
    test_nested_slice_accumulates_origin_offset();
    test_slicing_can_restore_contiguity();
    test_full_range_slice_preserves_layout();
    test_empty_slice_is_valid_and_contiguous();
    test_zero_extent_source_can_be_sliced();
    test_const_view_slice_preserves_element_constness();
    test_const_mutable_view_handle_preserves_mutability();
    test_slice_rejects_invalid_arguments_without_mutation();
    test_scalar_and_moved_from_views_reject_slice();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor view slice tests passed\n";
  return EXIT_SUCCESS;
}
