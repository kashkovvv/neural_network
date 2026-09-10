#include <algorithm>
#include <concepts>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "nn/tensor.hpp"

namespace {

using Tensor = nn::Tensor<float>;
using DoubleTensor = nn::Tensor<double>;
using IntegerTensor = nn::Tensor<int>;

template <typename TensorType>
concept CanSumAxes = requires(TensorType&& tensor) {
  std::forward<TensorType>(tensor).sum(
      typename std::remove_cvref_t<TensorType>::axes_type{}, false);
};

static_assert(std::same_as<decltype(std::declval<const Tensor&>().sum(
                               std::declval<const Tensor::axes_type&>())),
                           Tensor>);
static_assert(CanSumAxes<Tensor&>);
static_assert(CanSumAxes<const Tensor&>);
static_assert(CanSumAxes<Tensor&&>);
static_assert(CanSumAxes<const Tensor&&>);
static_assert(CanSumAxes<DoubleTensor&>);
static_assert(!CanSumAxes<IntegerTensor&>);
static_assert(!noexcept(std::declval<const Tensor&>().sum(
    std::declval<const Tensor::axes_type&>(), false)));

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

void expect_state(const Tensor& tensor,
                  const Tensor::shape_type& expected_shape,
                  const Tensor::strides_type& expected_strides,
                  const Tensor::storage_type& expected_elements,
                  const char* shape_message, const char* strides_message,
                  const char* elements_message) {
  expect(std::ranges::equal(tensor.shape(), expected_shape), shape_message);
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         strides_message);
  expect(std::ranges::equal(tensor.elements(), expected_elements),
         elements_message);
}

void expect_empty_sentinel(const Tensor& tensor) {
  expect(tensor.rank() == 0, "sentinel rank is not zero");
  expect(tensor.numel() == 0, "sentinel numel is not zero");
  expect(tensor.shape().empty(), "sentinel shape is not empty");
  expect(tensor.strides().empty(), "sentinel strides are not empty");
  expect(tensor.elements().empty(), "sentinel elements are not empty");
}

[[nodiscard]] Tensor make_rank_three_tensor() {
  return Tensor::from_data(
      {2, 3, 4}, {1.0F,  2.0F,  3.0F,  4.0F,  5.0F,  6.0F,  7.0F,  8.0F,
                  9.0F,  10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F,
                  17.0F, 18.0F, 19.0F, 20.0F, 21.0F, 22.0F, 23.0F, 24.0F});
}

void test_sum_reduces_one_axis_without_changing_source() {
  const Tensor tensor = make_rank_three_tensor();
  const Tensor::shape_type source_shape{2, 3, 4};
  const Tensor::strides_type source_strides{12, 4, 1};
  const Tensor::storage_type source_elements{
      1.0F,  2.0F,  3.0F,  4.0F,  5.0F,  6.0F,  7.0F,  8.0F,
      9.0F,  10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F,
      17.0F, 18.0F, 19.0F, 20.0F, 21.0F, 22.0F, 23.0F, 24.0F};

  const Tensor result = tensor.sum({1});

  expect_state(result, {2, 4}, {4, 1},
               {15.0F, 18.0F, 21.0F, 24.0F, 51.0F, 54.0F, 57.0F, 60.0F},
               "single-axis sum produced an incorrect shape",
               "single-axis sum produced incorrect strides",
               "single-axis sum produced incorrect values");
  expect_state(tensor, source_shape, source_strides, source_elements,
               "axis sum changed the source shape",
               "axis sum changed the source strides",
               "axis sum changed the source elements");
}

void test_sum_reduces_multiple_axes_independently_of_axis_order() {
  const Tensor tensor = make_rank_three_tensor();

  const Tensor ascending_result = tensor.sum({0, 2});
  const Tensor descending_result = tensor.sum({2, 0});

  expect_state(ascending_result, {3}, {1}, {68.0F, 100.0F, 132.0F},
               "multi-axis sum produced an incorrect shape",
               "multi-axis sum produced incorrect strides",
               "multi-axis sum produced incorrect values");
  expect_state(descending_result, {3}, {1}, {68.0F, 100.0F, 132.0F},
               "axis order changed the sum shape",
               "axis order changed the sum strides",
               "axis order changed the sum values");
}

void test_sum_keepdims_preserves_reduced_axis_positions() {
  const Tensor tensor =
      Tensor::from_data({2, 3, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F,
                                    8.0F, 9.0F, 10.0F, 11.0F, 12.0F});

  const Tensor one_axis_result = tensor.sum({1}, true);
  const Tensor multiple_axis_result = tensor.sum({0, 2}, true);

  expect_state(one_axis_result, {2, 1, 2}, {2, 2, 1},
               {9.0F, 12.0F, 27.0F, 30.0F},
               "keepdims single-axis sum produced an incorrect shape",
               "keepdims single-axis sum produced incorrect strides",
               "keepdims single-axis sum produced incorrect values");
  expect_state(multiple_axis_result, {1, 3, 1}, {3, 1, 1},
               {18.0F, 26.0F, 34.0F},
               "keepdims multi-axis sum produced an incorrect shape",
               "keepdims multi-axis sum produced incorrect strides",
               "keepdims multi-axis sum produced incorrect values");
}

void test_sum_all_axes_returns_scalar_or_all_singleton_shape() {
  const Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});

  const Tensor scalar_result = tensor.sum({0, 1});
  const Tensor keepdims_result = tensor.sum({1, 0}, true);

  expect_state(scalar_result, {}, {}, {10.0F},
               "all-axis sum did not produce a scalar shape",
               "all-axis sum did not produce scalar strides",
               "all-axis sum produced an incorrect scalar value");
  expect_state(keepdims_result, {1, 1}, {1, 1}, {10.0F},
               "keepdims all-axis sum produced an incorrect shape",
               "keepdims all-axis sum produced incorrect strides",
               "keepdims all-axis sum produced an incorrect value");
}

void test_sum_empty_axes_returns_independent_copy() {
  Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});

  Tensor result = tensor.sum({});
  const Tensor keepdims_result = tensor.sum({}, true);

  expect_state(result, {2, 2}, {2, 1}, {1.0F, 2.0F, 3.0F, 4.0F},
               "empty-axis sum changed shape", "empty-axis sum changed strides",
               "empty-axis sum changed elements");
  expect_state(keepdims_result, {2, 2}, {2, 1}, {1.0F, 2.0F, 3.0F, 4.0F},
               "keepdims affected empty-axis sum shape",
               "keepdims affected empty-axis sum strides",
               "keepdims affected empty-axis sum elements");
  expect(result.elements().data() != tensor.elements().data(),
         "empty-axis sum shares storage with its source");

  result[0, 0] = -1.0F;

  expect(tensor[0, 0] == 1.0F, "mutation of empty-axis sum changed its source");
}

void test_sum_handles_scalar_and_singleton_axes() {
  const Tensor scalar = Tensor::scalar(5.0F);
  const Tensor singleton =
      Tensor::from_data({2, 1, 2}, {1.0F, 2.0F, 3.0F, 4.0F});

  const Tensor scalar_result = scalar.sum({});
  const Tensor scalar_keepdims_result = scalar.sum({}, true);
  const Tensor singleton_result = singleton.sum({1});

  expect_state(scalar_result, {}, {}, {5.0F},
               "empty-axis scalar sum changed shape",
               "empty-axis scalar sum produced strides",
               "empty-axis scalar sum changed the value");
  expect_state(scalar_keepdims_result, {}, {}, {5.0F},
               "keepdims changed empty-axis scalar shape",
               "keepdims changed empty-axis scalar strides",
               "keepdims changed empty-axis scalar value");
  expect_state(singleton_result, {2, 2}, {2, 1}, {1.0F, 2.0F, 3.0F, 4.0F},
               "singleton-axis sum produced an incorrect shape",
               "singleton-axis sum produced incorrect strides",
               "singleton-axis sum changed values");

  expect_throws<std::out_of_range>(
      [&scalar] { static_cast<void>(scalar.sum({0})); },
      "axis sum accepted an axis for a rank-zero tensor",
      "invalid scalar axis produced the wrong exception type");
}

void test_sum_handles_zero_extent_reduction_domains() {
  const Tensor tensor({2, 0, 3});

  const Tensor reduced_zero_axis = tensor.sum({1});
  const Tensor reduced_zero_axis_keepdims = tensor.sum({1}, true);
  const Tensor retained_zero_axis = tensor.sum({2});
  const Tensor reduced_all_axes = tensor.sum({0, 1, 2});
  const Tensor reduced_all_axes_keepdims = tensor.sum({2, 0, 1}, true);

  expect_state(reduced_zero_axis, {2, 3}, {3, 1},
               {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F},
               "zero-axis reduction produced an incorrect shape",
               "zero-axis reduction produced incorrect strides",
               "zero-axis reduction did not use additive identities");
  expect_state(reduced_zero_axis_keepdims, {2, 1, 3}, {3, 3, 1},
               {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F},
               "keepdims zero-axis reduction produced an incorrect shape",
               "keepdims zero-axis reduction produced incorrect strides",
               "keepdims zero-axis reduction did not use identities");
  expect_state(retained_zero_axis, {2, 0}, {0, 1}, {},
               "sum did not retain an unreduced zero extent",
               "sum with an unreduced zero extent produced incorrect strides",
               "sum with an unreduced zero extent created elements");
  expect_state(reduced_all_axes, {}, {}, {0.0F},
               "all-axis empty-domain sum did not produce a scalar",
               "all-axis empty-domain sum produced scalar strides",
               "all-axis empty-domain sum did not produce additive identity");
  expect_state(reduced_all_axes_keepdims, {1, 1, 1}, {1, 1, 1}, {0.0F},
               "keepdims all-axis empty-domain sum produced an incorrect shape",
               "keepdims all-axis empty-domain sum produced incorrect strides",
               "keepdims all-axis empty-domain sum did not produce additive "
               "identity");
}

void test_sum_rejects_unrepresentable_result_shapes_without_changes() {
  const Tensor::size_type max_size =
      std::numeric_limits<Tensor::size_type>::max();
  const Tensor overflow_source({max_size, 0, 2});

  expect_throws<std::overflow_error>(
      [&overflow_source] { static_cast<void>(overflow_source.sum({1})); },
      "axis sum accepted a result shape whose product overflows size_type",
      "overflowing axis-sum result shape produced the wrong exception type");
  expect_state(overflow_source, {max_size, 0, 2}, {0, 2, 1}, {},
               "overflowing axis sum changed the source shape",
               "overflowing axis sum changed the source strides",
               "overflowing axis sum changed the source elements");

  const Tensor::size_type storage_max_size = Tensor::storage_type{}.max_size();

  expect(storage_max_size < max_size,
         "test requires storage max_size below size_type maximum");

  const Tensor::size_type oversized_extent = storage_max_size + 1;
  const Tensor oversized_source({oversized_extent, 0});

  expect_throws<std::length_error>(
      [&oversized_source] { static_cast<void>(oversized_source.sum({1})); },
      "axis sum accepted a result shape larger than storage max_size",
      "oversized axis-sum result shape produced the wrong exception type");
  expect_state(oversized_source, {oversized_extent, 0}, {0, 1}, {},
               "oversized axis sum changed the source shape",
               "oversized axis sum changed the source strides",
               "oversized axis sum changed the source elements");
}

void test_sum_rejects_invalid_axes_without_changing_source() {
  const Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});

  expect_throws<std::invalid_argument>(
      [&tensor] { static_cast<void>(tensor.sum({0, 0})); },
      "axis sum accepted a duplicate axis",
      "duplicate reduction axes produced the wrong exception type");
  expect_throws<std::out_of_range>(
      [&tensor] { static_cast<void>(tensor.sum({2})); },
      "axis sum accepted an out-of-range axis",
      "out-of-range reduction axis produced the wrong exception type");
  expect_state(tensor, {2, 2}, {2, 1}, {1.0F, 2.0F, 3.0F, 4.0F},
               "failed axis sum changed the source shape",
               "failed axis sum changed the source strides",
               "failed axis sum changed the source elements");
}

void test_sum_accepts_rvalue_without_consuming_source() {
  Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});

  const Tensor result = std::move(tensor).sum({0});

  expect_state(result, {2}, {1}, {4.0F, 6.0F},
               "rvalue axis sum produced an incorrect shape",
               "rvalue axis sum produced incorrect strides",
               "rvalue axis sum produced incorrect values");
  expect_state(tensor, {2, 2}, {2, 1}, {1.0F, 2.0F, 3.0F, 4.0F},
               "axis sum consumed its rvalue source shape",
               "axis sum consumed its rvalue source strides",
               "axis sum consumed its rvalue source elements");
}

void test_sum_rejects_moved_from_sentinel() {
  Tensor source = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor owner(std::move(source));
  static_cast<void>(owner);

  expect_throws<std::invalid_argument>(
      [&source] { static_cast<void>(source.sum({})); },
      "axis sum accepted a moved-from sentinel",
      "axis sum of a moved-from sentinel produced the wrong exception type");
  expect_empty_sentinel(source);
}

}  // namespace

int main() {
  try {
    test_sum_reduces_one_axis_without_changing_source();
    test_sum_reduces_multiple_axes_independently_of_axis_order();
    test_sum_keepdims_preserves_reduced_axis_positions();
    test_sum_all_axes_returns_scalar_or_all_singleton_shape();
    test_sum_empty_axes_returns_independent_copy();
    test_sum_handles_scalar_and_singleton_axes();
    test_sum_handles_zero_extent_reduction_domains();
    test_sum_rejects_unrepresentable_result_shapes_without_changes();
    test_sum_rejects_invalid_axes_without_changing_source();
    test_sum_accepts_rvalue_without_consuming_source();
    test_sum_rejects_moved_from_sentinel();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor axis sum tests passed\n";
  return EXIT_SUCCESS;
}
