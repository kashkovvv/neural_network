#include <algorithm>
#include <concepts>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

#include "nn/tensor.hpp"

namespace {

using Tensor = nn::Tensor<float>;
using MutableView = nn::TensorView<float>;
using ConstView = nn::TensorView<const float>;

template <typename SourceType>
concept CanSliceWithStep = requires {
  std::declval<SourceType>().slice(Tensor::size_type{0}, Tensor::size_type{0},
                                   Tensor::size_type{0}, Tensor::size_type{1});
};

static_assert(CanSliceWithStep<MutableView&>);
static_assert(CanSliceWithStep<const MutableView&>);
static_assert(CanSliceWithStep<MutableView&&>);
static_assert(CanSliceWithStep<const MutableView&&>);
static_assert(CanSliceWithStep<ConstView&>);
static_assert(CanSliceWithStep<const ConstView&>);
static_assert(CanSliceWithStep<ConstView&&>);
static_assert(CanSliceWithStep<const ConstView&&>);

static_assert(CanSliceWithStep<Tensor&>);
static_assert(CanSliceWithStep<const Tensor&>);
static_assert(!CanSliceWithStep<Tensor&&>);
static_assert(!CanSliceWithStep<const Tensor&&>);

static_assert(
    std::same_as<decltype(std::declval<const MutableView&>().slice(0, 0, 0, 1)),
                 MutableView>);
static_assert(
    std::same_as<decltype(std::declval<const ConstView&>().slice(0, 0, 0, 1)),
                 ConstView>);
static_assert(std::same_as<decltype(std::declval<Tensor&>().slice(0, 0, 0, 1)),
                           MutableView>);
static_assert(
    std::same_as<decltype(std::declval<const Tensor&>().slice(0, 0, 0, 1)),
                 ConstView>);

static_assert(!noexcept(std::declval<const MutableView&>().slice(0, 0, 0, 1)));
static_assert(!noexcept(std::declval<const ConstView&>().slice(0, 0, 0, 1)));
static_assert(!noexcept(std::declval<Tensor&>().slice(0, 0, 0, 1)));
static_assert(!noexcept(std::declval<const Tensor&>().slice(0, 0, 0, 1)));

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

void test_strided_slice_selects_expected_elements() {
  Tensor tensor =
      Tensor::from_data({8}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F});
  MutableView result = tensor.view().slice(0, 1, 8, 3);

  const Tensor::shape_type expected_shape{3};
  const Tensor::strides_type expected_strides{3};

  expect(std::ranges::equal(result.shape(), expected_shape),
         "strided slice shape is incorrect");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "strided slice strides are incorrect");
  expect(result.numel() == 3, "strided slice numel is incorrect");
  expect(!result.is_contiguous(),
         "multi-element strided slice must be non-contiguous");
  expect(result.at(0) == 1.0F && result.at(1) == 4.0F && result.at(2) == 7.0F,
         "strided slice returned incorrect elements");

  result.at(1) = 40.0F;

  expect(tensor.at(4) == 40.0F, "strided slice does not alias tensor storage");
}

void test_leading_axis_step_skips_complete_blocks() {
  Tensor tensor = Tensor::from_data(
      {5, 3}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F,
               10.0F, 11.0F, 12.0F, 13.0F, 14.0F});
  MutableView result = tensor.view().slice(0, 0, 5, 2);

  const Tensor::shape_type expected_shape{3, 3};
  const Tensor::strides_type expected_strides{6, 1};

  expect(std::ranges::equal(result.shape(), expected_shape),
         "leading-axis strided slice shape is incorrect");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "leading-axis strided slice strides are incorrect");
  expect(!result.is_contiguous(),
         "leading-axis strided slice must be non-contiguous");
  expect(result.at(0, 0) == 0.0F && result.at(1, 2) == 8.0F &&
             result.at(2, 2) == 14.0F,
         "leading-axis strided slice returned incorrect elements");
}

void test_nested_steps_compose_strides_and_origin() {
  Tensor tensor = Tensor::from_data(
      {10}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F});
  MutableView result = tensor.view().slice(0, 1, 10, 2).slice(0, 1, 5, 2);

  const Tensor::shape_type expected_shape{2};
  const Tensor::strides_type expected_strides{4};

  expect(std::ranges::equal(result.shape(), expected_shape),
         "nested strided slice shape is incorrect");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "nested strided slice did not compose strides");
  expect(result.at(0) == 3.0F && result.at(1) == 7.0F,
         "nested strided slice did not compose origin offsets");
}

void test_subslice_can_restore_contiguity_after_step() {
  Tensor tensor =
      Tensor::from_data({3, 4}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F,
                                 8.0F, 9.0F, 10.0F, 11.0F});
  MutableView skipped_rows = tensor.view().slice(0, 0, 3, 2);
  MutableView result = skipped_rows.slice(0, 1, 2);

  const Tensor::shape_type expected_shape{1, 4};
  const Tensor::strides_type expected_strides{8, 1};

  expect(!skipped_rows.is_contiguous(),
         "multi-row stepped slice must be non-contiguous");
  expect(std::ranges::equal(result.shape(), expected_shape),
         "contiguity-restoring subslice shape is incorrect");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "contiguity-restoring subslice strides are incorrect");
  expect(result.is_contiguous(), "single-row subslice must restore contiguity");
  expect(result.at(0, 0) == 8.0F && result.at(0, 3) == 11.0F,
         "contiguity-restoring subslice returned incorrect elements");
}

void test_singleton_result_avoids_irrelevant_stride_overflow() {
  Tensor tensor =
      Tensor::from_data({3, 4}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F,
                                 8.0F, 9.0F, 10.0F, 11.0F});
  const Tensor::size_type maximum_step =
      std::numeric_limits<Tensor::size_type>::max();
  MutableView result = tensor.view().slice(0, 1, 2, maximum_step);

  const Tensor::shape_type expected_shape{1, 4};
  const Tensor::strides_type expected_strides{4, 1};

  expect(std::ranges::equal(result.shape(), expected_shape),
         "singleton strided slice shape is incorrect");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "singleton strided slice changed an irrelevant stride");
  expect(result.is_contiguous(),
         "single complete row selected with a large step must be contiguous");
  expect(result.at(0, 0) == 4.0F && result.at(0, 3) == 7.0F,
         "singleton strided slice returned incorrect elements");
}

void test_empty_source_accepts_nonzero_step() {
  Tensor tensor({2, 0, 4});
  MutableView result = tensor.view().slice(1, 0, 0, 2);

  const Tensor::shape_type expected_shape{2, 0, 4};
  const Tensor::strides_type expected_strides{0, 4, 1};

  expect(std::ranges::equal(result.shape(), expected_shape),
         "empty strided slice shape is incorrect");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "empty strided slice changed an irrelevant stride");
  expect(result.numel() == 0, "empty strided slice numel must be zero");
  expect(result.is_contiguous(), "empty strided slice must be contiguous");
}

void test_zero_step_is_rejected_without_mutation() {
  Tensor tensor =
      Tensor::from_data({2, 3}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
  MutableView source = tensor.view();
  const Tensor::shape_type expected_shape(source.shape().begin(),
                                          source.shape().end());
  const Tensor::strides_type expected_strides(source.strides().begin(),
                                              source.strides().end());

  expect_throws<std::invalid_argument>(
      [&source] { static_cast<void>(source.slice(1, 0, 3, 0)); },
      "strided slice accepted a zero step",
      "zero slice step produced the wrong exception type");

  expect(std::ranges::equal(source.shape(), expected_shape),
         "zero-step failure changed source shape");
  expect(std::ranges::equal(source.strides(), expected_strides),
         "zero-step failure changed source strides");
  expect(source.numel() == 6, "zero-step failure changed source numel");
  expect(source.is_contiguous(), "zero-step failure changed source contiguity");
  expect(source.at(1, 2) == 5.0F,
         "zero-step failure changed source storage binding");
}

void test_direct_tensor_strided_slice_preserves_constness_and_aliasing() {
  Tensor tensor =
      Tensor::from_data({2, 6}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F,
                                 8.0F, 9.0F, 10.0F, 11.0F});
  MutableView mutable_result = tensor.slice(1, 1, 6, 2);
  const Tensor& const_tensor = tensor;
  ConstView const_result = const_tensor.slice(1, 0, 6, 2);

  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{6, 2};

  expect(std::ranges::equal(mutable_result.shape(), expected_shape),
         "direct mutable strided slice shape is incorrect");
  expect(std::ranges::equal(mutable_result.strides(), expected_strides),
         "direct mutable strided slice strides are incorrect");
  expect(mutable_result.at(1, 2) == 11.0F,
         "direct mutable strided slice returned an incorrect element");
  expect(const_result.at(1, 2) == 10.0F,
         "direct const strided slice returned an incorrect element");

  mutable_result.at(0, 1) = 30.0F;

  expect(tensor.at(0, 3) == 30.0F,
         "direct mutable strided slice does not alias tensor storage");
  expect(const_result.at(0, 1) == 2.0F,
         "direct const strided slice reads an incorrect storage location");
}

}  // namespace

int main() {
  try {
    test_strided_slice_selects_expected_elements();
    test_leading_axis_step_skips_complete_blocks();
    test_nested_steps_compose_strides_and_origin();
    test_subslice_can_restore_contiguity_after_step();
    test_singleton_result_avoids_irrelevant_stride_overflow();
    test_empty_source_accepts_nonzero_step();
    test_zero_step_is_rejected_without_mutation();
    test_direct_tensor_strided_slice_preserves_constness_and_aliasing();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor view strided slice tests passed\n";
  return EXIT_SUCCESS;
}
