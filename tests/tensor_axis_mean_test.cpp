#include <algorithm>
#include <cmath>
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
concept CanMeanAxes = requires(TensorType&& tensor) {
  std::forward<TensorType>(tensor).mean(
      typename std::remove_cvref_t<TensorType>::axes_type{}, false);
};

static_assert(std::same_as<decltype(std::declval<const Tensor&>().mean(
                               std::declval<const Tensor::axes_type&>())),
                           Tensor>);
static_assert(CanMeanAxes<Tensor&>);
static_assert(CanMeanAxes<const Tensor&>);
static_assert(CanMeanAxes<Tensor&&>);
static_assert(CanMeanAxes<const Tensor&&>);
static_assert(CanMeanAxes<DoubleTensor&>);
static_assert(!CanMeanAxes<IntegerTensor&>);
static_assert(!noexcept(std::declval<const Tensor&>().mean(
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

void expect_nan_state(const Tensor& tensor,
                      const Tensor::shape_type& expected_shape,
                      const Tensor::strides_type& expected_strides,
                      Tensor::size_type expected_numel,
                      const char* shape_message, const char* strides_message,
                      const char* elements_message) {
  expect(std::ranges::equal(tensor.shape(), expected_shape), shape_message);
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         strides_message);
  expect(tensor.numel() == expected_numel, elements_message);
  expect(std::ranges::all_of(
             tensor.elements(),
             [](Tensor::value_type value) { return std::isnan(value); }),
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

void test_mean_reduces_one_axis_without_changing_source() {
  const Tensor tensor = make_rank_three_tensor();
  const Tensor::shape_type source_shape{2, 3, 4};
  const Tensor::strides_type source_strides{12, 4, 1};
  const Tensor::storage_type source_elements{
      1.0F,  2.0F,  3.0F,  4.0F,  5.0F,  6.0F,  7.0F,  8.0F,
      9.0F,  10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F,
      17.0F, 18.0F, 19.0F, 20.0F, 21.0F, 22.0F, 23.0F, 24.0F};

  const Tensor result = tensor.mean({1});

  expect_state(result, {2, 4}, {4, 1},
               {5.0F, 6.0F, 7.0F, 8.0F, 17.0F, 18.0F, 19.0F, 20.0F},
               "single-axis mean produced an incorrect shape",
               "single-axis mean produced incorrect strides",
               "single-axis mean produced incorrect values");
  expect_state(tensor, source_shape, source_strides, source_elements,
               "axis mean changed the source shape",
               "axis mean changed the source strides",
               "axis mean changed the source elements");
}

void test_mean_reduces_multiple_axes_independently_of_axis_order() {
  const Tensor tensor = make_rank_three_tensor();

  const Tensor ascending_result = tensor.mean({0, 2});
  const Tensor descending_result = tensor.mean({2, 0});

  expect_state(ascending_result, {3}, {1}, {8.5F, 12.5F, 16.5F},
               "multi-axis mean produced an incorrect shape",
               "multi-axis mean produced incorrect strides",
               "multi-axis mean produced incorrect values");
  expect_state(descending_result, {3}, {1}, {8.5F, 12.5F, 16.5F},
               "axis order changed the mean shape",
               "axis order changed the mean strides",
               "axis order changed the mean values");
}

void test_mean_keepdims_preserves_reduced_axis_positions() {
  const Tensor tensor =
      Tensor::from_data({2, 3, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F,
                                    8.0F, 9.0F, 10.0F, 11.0F, 12.0F});

  const Tensor one_axis_result = tensor.mean({1}, true);
  const Tensor multiple_axis_result = tensor.mean({0, 2}, true);

  expect_state(one_axis_result, {2, 1, 2}, {2, 2, 1}, {3.0F, 4.0F, 9.0F, 10.0F},
               "keepdims single-axis mean produced an incorrect shape",
               "keepdims single-axis mean produced incorrect strides",
               "keepdims single-axis mean produced incorrect values");
  expect_state(multiple_axis_result, {1, 3, 1}, {3, 1, 1}, {4.5F, 6.5F, 8.5F},
               "keepdims multi-axis mean produced an incorrect shape",
               "keepdims multi-axis mean produced incorrect strides",
               "keepdims multi-axis mean produced incorrect values");
}

void test_mean_all_axes_returns_scalar_or_all_singleton_shape() {
  const Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});

  const Tensor scalar_result = tensor.mean({0, 1});
  const Tensor keepdims_result = tensor.mean({1, 0}, true);

  expect_state(scalar_result, {}, {}, {2.5F},
               "all-axis mean did not produce a scalar shape",
               "all-axis mean did not produce scalar strides",
               "all-axis mean produced an incorrect scalar value");
  expect_state(keepdims_result, {1, 1}, {1, 1}, {2.5F},
               "keepdims all-axis mean produced an incorrect shape",
               "keepdims all-axis mean produced incorrect strides",
               "keepdims all-axis mean produced an incorrect value");
}

void test_mean_empty_axes_returns_independent_copy() {
  Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  const Tensor zero_extent_tensor({2, 0, 3});

  Tensor result = tensor.mean({});
  const Tensor keepdims_result = tensor.mean({}, true);
  const Tensor zero_extent_result = zero_extent_tensor.mean({});

  expect_state(result, {2, 2}, {2, 1}, {1.0F, 2.0F, 3.0F, 4.0F},
               "empty-axis mean changed shape",
               "empty-axis mean changed strides",
               "empty-axis mean changed elements");
  expect_state(keepdims_result, {2, 2}, {2, 1}, {1.0F, 2.0F, 3.0F, 4.0F},
               "keepdims affected empty-axis mean shape",
               "keepdims affected empty-axis mean strides",
               "keepdims affected empty-axis mean elements");
  expect_state(zero_extent_result, {2, 0, 3}, {0, 3, 1}, {},
               "empty-axis mean changed a zero-extent shape",
               "empty-axis mean changed zero-extent strides",
               "empty-axis mean created zero-extent elements");
  expect(result.elements().data() != tensor.elements().data(),
         "empty-axis mean shares storage with its source");

  result[0, 0] = -1.0F;

  expect(tensor[0, 0] == 1.0F,
         "mutation of empty-axis mean changed its source");
}

void test_mean_handles_scalar_and_singleton_axes() {
  const Tensor scalar = Tensor::scalar(5.0F);
  const Tensor singleton =
      Tensor::from_data({2, 1, 2}, {1.0F, 2.0F, 3.0F, 4.0F});

  const Tensor scalar_result = scalar.mean({});
  const Tensor scalar_keepdims_result = scalar.mean({}, true);
  const Tensor singleton_result = singleton.mean({1});

  expect_state(scalar_result, {}, {}, {5.0F},
               "empty-axis scalar mean changed shape",
               "empty-axis scalar mean produced strides",
               "empty-axis scalar mean changed the value");
  expect_state(scalar_keepdims_result, {}, {}, {5.0F},
               "keepdims changed empty-axis scalar shape",
               "keepdims changed empty-axis scalar strides",
               "keepdims changed empty-axis scalar value");
  expect_state(singleton_result, {2, 2}, {2, 1}, {1.0F, 2.0F, 3.0F, 4.0F},
               "singleton-axis mean produced an incorrect shape",
               "singleton-axis mean produced incorrect strides",
               "singleton-axis mean changed values");

  expect_throws<std::out_of_range>(
      [&scalar] { static_cast<void>(scalar.mean({0})); },
      "axis mean accepted an axis for a rank-zero tensor",
      "invalid scalar axis produced the wrong exception type");
}

void test_mean_handles_zero_extent_reduction_domains() {
  const Tensor tensor({2, 0, 3});

  const Tensor reduced_zero_axis = tensor.mean({1});
  const Tensor reduced_zero_axis_keepdims = tensor.mean({1}, true);
  const Tensor retained_zero_axis = tensor.mean({2});
  const Tensor reduced_all_axes = tensor.mean({0, 1, 2});
  const Tensor reduced_all_axes_keepdims = tensor.mean({2, 0, 1}, true);

  expect_nan_state(reduced_zero_axis, {2, 3}, {3, 1}, 6,
                   "zero-axis mean produced an incorrect shape",
                   "zero-axis mean produced incorrect strides",
                   "zero-axis mean did not produce NaN values");
  expect_nan_state(reduced_zero_axis_keepdims, {2, 1, 3}, {3, 3, 1}, 6,
                   "keepdims zero-axis mean produced an incorrect shape",
                   "keepdims zero-axis mean produced incorrect strides",
                   "keepdims zero-axis mean did not produce NaN values");
  expect_state(retained_zero_axis, {2, 0}, {0, 1}, {},
               "mean did not retain an unreduced zero extent",
               "mean with an unreduced zero extent produced incorrect strides",
               "mean with an unreduced zero extent created elements");
  expect_nan_state(reduced_all_axes, {}, {}, 1,
                   "all-axis empty-domain mean did not produce a scalar",
                   "all-axis empty-domain mean produced scalar strides",
                   "all-axis empty-domain mean did not produce NaN");
  expect_nan_state(
      reduced_all_axes_keepdims, {1, 1, 1}, {1, 1, 1}, 1,
      "keepdims all-axis empty-domain mean produced an incorrect shape",
      "keepdims all-axis empty-domain mean produced incorrect strides",
      "keepdims all-axis empty-domain mean did not produce NaN");
}

void test_mean_avoids_reduction_extent_product_overflow() {
  const Tensor::size_type max_size =
      std::numeric_limits<Tensor::size_type>::max();
  const Tensor empty_result_source({max_size, 0, 2});
  const Tensor empty_domain_source({max_size, 2, 0, 1});

  const Tensor empty_result = empty_result_source.mean({0, 2});
  const Tensor empty_domain = empty_domain_source.mean({0, 1, 2});

  expect_state(empty_result, {0}, {1}, {},
               "mean with a large reduction domain produced an incorrect shape",
               "mean with a large reduction domain produced incorrect strides",
               "mean with a large reduction domain created elements");
  expect_nan_state(
      empty_domain, {1}, {1}, 1,
      "empty-domain mean with large reduced extents produced an incorrect "
      "shape",
      "empty-domain mean with large reduced extents produced incorrect strides",
      "empty-domain mean with large reduced extents did not produce NaN");
}

void test_mean_rejects_invalid_axes_without_changing_source() {
  const Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});

  expect_throws<std::invalid_argument>(
      [&tensor] { static_cast<void>(tensor.mean({0, 0})); },
      "axis mean accepted a duplicate axis",
      "duplicate reduction axes produced the wrong exception type");
  expect_throws<std::out_of_range>(
      [&tensor] { static_cast<void>(tensor.mean({2})); },
      "axis mean accepted an out-of-range axis",
      "out-of-range reduction axis produced the wrong exception type");
  expect_state(tensor, {2, 2}, {2, 1}, {1.0F, 2.0F, 3.0F, 4.0F},
               "failed axis mean changed the source shape",
               "failed axis mean changed the source strides",
               "failed axis mean changed the source elements");
}

void test_mean_accepts_rvalue_without_consuming_source() {
  Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});

  const Tensor result = std::move(tensor).mean({0});

  expect_state(result, {2}, {1}, {2.0F, 3.0F},
               "rvalue axis mean produced an incorrect shape",
               "rvalue axis mean produced incorrect strides",
               "rvalue axis mean produced incorrect values");
  expect_state(tensor, {2, 2}, {2, 1}, {1.0F, 2.0F, 3.0F, 4.0F},
               "axis mean consumed its rvalue source shape",
               "axis mean consumed its rvalue source strides",
               "axis mean consumed its rvalue source elements");
}

void test_mean_rejects_moved_from_sentinel() {
  Tensor source = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor owner(std::move(source));
  static_cast<void>(owner);

  expect_throws<std::invalid_argument>(
      [&source] { static_cast<void>(source.mean({})); },
      "axis mean accepted a moved-from sentinel",
      "axis mean of a moved-from sentinel produced the wrong exception type");
  expect_empty_sentinel(source);
}

}  // namespace

int main() {
  try {
    test_mean_reduces_one_axis_without_changing_source();
    test_mean_reduces_multiple_axes_independently_of_axis_order();
    test_mean_keepdims_preserves_reduced_axis_positions();
    test_mean_all_axes_returns_scalar_or_all_singleton_shape();
    test_mean_empty_axes_returns_independent_copy();
    test_mean_handles_scalar_and_singleton_axes();
    test_mean_handles_zero_extent_reduction_domains();
    test_mean_avoids_reduction_extent_product_overflow();
    test_mean_rejects_invalid_axes_without_changing_source();
    test_mean_accepts_rvalue_without_consuming_source();
    test_mean_rejects_moved_from_sentinel();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor axis mean tests passed\n";
  return EXIT_SUCCESS;
}
