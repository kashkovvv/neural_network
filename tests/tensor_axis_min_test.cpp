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
concept CanMinAxes = requires(TensorType&& tensor) {
  std::forward<TensorType>(tensor).min(
      typename std::remove_cvref_t<TensorType>::axes_type{}, false);
};

static_assert(std::same_as<decltype(std::declval<const Tensor&>().min(
                               std::declval<const Tensor::axes_type&>())),
                           Tensor>);
static_assert(CanMinAxes<Tensor&>);
static_assert(CanMinAxes<const Tensor&>);
static_assert(CanMinAxes<Tensor&&>);
static_assert(CanMinAxes<const Tensor&&>);
static_assert(CanMinAxes<DoubleTensor&>);
static_assert(!CanMinAxes<IntegerTensor&>);
static_assert(!noexcept(std::declval<const Tensor&>().min(
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

void test_min_reduces_one_axis_without_changing_source() {
  const Tensor tensor = make_rank_three_tensor();
  const Tensor::shape_type source_shape{2, 3, 4};
  const Tensor::strides_type source_strides{12, 4, 1};
  const Tensor::storage_type source_elements{
      1.0F,  2.0F,  3.0F,  4.0F,  5.0F,  6.0F,  7.0F,  8.0F,
      9.0F,  10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F,
      17.0F, 18.0F, 19.0F, 20.0F, 21.0F, 22.0F, 23.0F, 24.0F};

  const Tensor result = tensor.min({1});

  expect_state(result, {2, 4}, {4, 1},
               {1.0F, 2.0F, 3.0F, 4.0F, 13.0F, 14.0F, 15.0F, 16.0F},
               "single-axis min produced an incorrect shape",
               "single-axis min produced incorrect strides",
               "single-axis min produced incorrect values");
  expect_state(tensor, source_shape, source_strides, source_elements,
               "axis min changed the source shape",
               "axis min changed the source strides",
               "axis min changed the source elements");
}

void test_min_reduces_multiple_axes_independently_of_axis_order() {
  const Tensor tensor = make_rank_three_tensor();

  const Tensor ascending_result = tensor.min({0, 2});
  const Tensor descending_result = tensor.min({2, 0});

  expect_state(ascending_result, {3}, {1}, {1.0F, 5.0F, 9.0F},
               "multi-axis min produced an incorrect shape",
               "multi-axis min produced incorrect strides",
               "multi-axis min produced incorrect values");
  expect_state(descending_result, {3}, {1}, {1.0F, 5.0F, 9.0F},
               "axis order changed the min shape",
               "axis order changed the min strides",
               "axis order changed the min values");
}

void test_min_keepdims_preserves_reduced_axis_positions() {
  const Tensor tensor =
      Tensor::from_data({2, 3, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F,
                                    8.0F, 9.0F, 10.0F, 11.0F, 12.0F});

  const Tensor one_axis_result = tensor.min({1}, true);
  const Tensor multiple_axis_result = tensor.min({0, 2}, true);

  expect_state(one_axis_result, {2, 1, 2}, {2, 2, 1}, {1.0F, 2.0F, 7.0F, 8.0F},
               "keepdims single-axis min produced an incorrect shape",
               "keepdims single-axis min produced incorrect strides",
               "keepdims single-axis min produced incorrect values");
  expect_state(multiple_axis_result, {1, 3, 1}, {3, 1, 1}, {1.0F, 3.0F, 5.0F},
               "keepdims multi-axis min produced an incorrect shape",
               "keepdims multi-axis min produced incorrect strides",
               "keepdims multi-axis min produced incorrect values");
}

void test_min_all_axes_returns_scalar_or_all_singleton_shape() {
  const Tensor tensor = Tensor::from_data({2, 2}, {4.0F, 2.0F, -1.0F, 3.0F});

  const Tensor scalar_result = tensor.min({0, 1});
  const Tensor keepdims_result = tensor.min({1, 0}, true);

  expect_state(scalar_result, {}, {}, {-1.0F},
               "all-axis min did not produce a scalar shape",
               "all-axis min did not produce scalar strides",
               "all-axis min produced an incorrect scalar value");
  expect_state(keepdims_result, {1, 1}, {1, 1}, {-1.0F},
               "keepdims all-axis min produced an incorrect shape",
               "keepdims all-axis min produced incorrect strides",
               "keepdims all-axis min produced an incorrect value");
}

void test_min_empty_axes_returns_independent_copy() {
  Tensor tensor = Tensor::from_data({2, 2}, {4.0F, 2.0F, -1.0F, 3.0F});
  const Tensor zero_extent_tensor({2, 0, 3});

  Tensor result = tensor.min({});
  const Tensor keepdims_result = tensor.min({}, true);
  const Tensor zero_extent_result = zero_extent_tensor.min({});

  expect_state(result, {2, 2}, {2, 1}, {4.0F, 2.0F, -1.0F, 3.0F},
               "empty-axis min changed shape", "empty-axis min changed strides",
               "empty-axis min changed elements");
  expect_state(keepdims_result, {2, 2}, {2, 1}, {4.0F, 2.0F, -1.0F, 3.0F},
               "keepdims affected empty-axis min shape",
               "keepdims affected empty-axis min strides",
               "keepdims affected empty-axis min elements");
  expect_state(zero_extent_result, {2, 0, 3}, {0, 3, 1}, {},
               "empty-axis min changed a zero-extent shape",
               "empty-axis min changed zero-extent strides",
               "empty-axis min created zero-extent elements");
  expect(result.elements().data() != tensor.elements().data(),
         "empty-axis min shares storage with its source");

  result[0, 0] = 100.0F;

  expect(tensor[0, 0] == 4.0F, "mutation of empty-axis min changed its source");
}

void test_min_handles_scalar_and_singleton_axes() {
  const Tensor scalar = Tensor::scalar(5.0F);
  const Tensor singleton =
      Tensor::from_data({2, 1, 2}, {4.0F, 2.0F, -1.0F, 3.0F});

  const Tensor scalar_result = scalar.min({});
  const Tensor scalar_keepdims_result = scalar.min({}, true);
  const Tensor singleton_result = singleton.min({1});

  expect_state(scalar_result, {}, {}, {5.0F},
               "empty-axis scalar min changed shape",
               "empty-axis scalar min produced strides",
               "empty-axis scalar min changed the value");
  expect_state(scalar_keepdims_result, {}, {}, {5.0F},
               "keepdims changed empty-axis scalar shape",
               "keepdims changed empty-axis scalar strides",
               "keepdims changed empty-axis scalar value");
  expect_state(singleton_result, {2, 2}, {2, 1}, {4.0F, 2.0F, -1.0F, 3.0F},
               "singleton-axis min produced an incorrect shape",
               "singleton-axis min produced incorrect strides",
               "singleton-axis min changed values");

  expect_throws<std::out_of_range>(
      [&scalar] { static_cast<void>(scalar.min({0})); },
      "axis min accepted an axis for a rank-zero tensor",
      "invalid scalar axis produced the wrong exception type");
}

void test_min_propagates_nan_only_to_affected_outputs() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const Tensor tensor =
      Tensor::from_data({2, 3}, {4.0F, nan, 2.0F, 3.0F, 1.0F, 2.0F});

  const Tensor result = tensor.min({1});
  const Tensor keepdims_result = tensor.min({1}, true);

  expect(std::ranges::equal(result.shape(), Tensor::shape_type{2}),
         "NaN axis min produced an incorrect shape");
  expect(std::ranges::equal(result.strides(), Tensor::strides_type{1}),
         "NaN axis min produced incorrect strides");
  expect(result.numel() == 2, "NaN axis min produced incorrect numel");
  expect(std::isnan(result[0]), "axis min did not propagate NaN");
  expect(result[1] == 1.0F, "NaN axis min produced an incorrect finite value");
  expect(std::ranges::equal(keepdims_result.shape(), Tensor::shape_type{2, 1}),
         "keepdims NaN axis min produced an incorrect shape");
  expect(
      std::ranges::equal(keepdims_result.strides(), Tensor::strides_type{1, 1}),
      "keepdims NaN axis min produced incorrect strides");
  expect(keepdims_result.numel() == 2,
         "keepdims NaN axis min produced incorrect numel");
  expect(std::isnan(keepdims_result[0, 0]),
         "keepdims axis min did not propagate NaN");
  expect(keepdims_result[1, 0] == 1.0F,
         "keepdims NaN axis min produced an incorrect finite value");
}

void test_min_distinguishes_empty_domains_from_empty_results() {
  const Tensor tensor({2, 0, 3});

  expect_throws<std::domain_error>(
      [&tensor] { static_cast<void>(tensor.min({1})); },
      "axis min accepted an empty reduction domain",
      "empty reduction domain produced the wrong exception type");
  expect_throws<std::domain_error>(
      [&tensor] { static_cast<void>(tensor.min({1}, true)); },
      "keepdims axis min accepted an empty reduction domain",
      "keepdims empty reduction domain produced the wrong exception type");
  expect_throws<std::domain_error>(
      [&tensor] { static_cast<void>(tensor.min({0, 1, 2})); },
      "all-axis min accepted an empty reduction domain",
      "all-axis empty reduction domain produced the wrong exception type");
  expect_throws<std::domain_error>(
      [&tensor] { static_cast<void>(tensor.min({0, 1, 2}, true)); },
      "keepdims all-axis min accepted an empty reduction domain",
      "keepdims all-axis empty domain produced the wrong exception type");

  const Tensor retained_zero_axis = tensor.min({2});
  const Tensor retained_zero_axis_multiple = tensor.min({0, 2});

  expect_state(retained_zero_axis, {2, 0}, {0, 1}, {},
               "axis min did not retain a zero extent",
               "axis min with a retained zero extent produced wrong strides",
               "axis min with a retained zero extent created elements");
  expect_state(
      retained_zero_axis_multiple, {0}, {1}, {},
      "multi-axis min did not retain a zero extent",
      "multi-axis min with a retained zero extent produced wrong strides",
      "multi-axis min with a retained zero extent created elements");
  expect_state(tensor, {2, 0, 3}, {0, 3, 1}, {},
               "zero-extent axis min changed the source shape",
               "zero-extent axis min changed the source strides",
               "zero-extent axis min changed the source elements");
}

void test_min_rejects_invalid_axes_without_changing_source() {
  const Tensor tensor = Tensor::from_data({2, 2}, {4.0F, 2.0F, -1.0F, 3.0F});

  expect_throws<std::invalid_argument>(
      [&tensor] { static_cast<void>(tensor.min({0, 0})); },
      "axis min accepted a duplicate axis",
      "duplicate reduction axes produced the wrong exception type");
  expect_throws<std::out_of_range>(
      [&tensor] { static_cast<void>(tensor.min({2})); },
      "axis min accepted an out-of-range axis",
      "out-of-range reduction axis produced the wrong exception type");
  expect_state(tensor, {2, 2}, {2, 1}, {4.0F, 2.0F, -1.0F, 3.0F},
               "failed axis min changed the source shape",
               "failed axis min changed the source strides",
               "failed axis min changed the source elements");
}

void test_min_accepts_rvalue_without_consuming_source() {
  Tensor tensor = Tensor::from_data({2, 2}, {4.0F, 2.0F, -1.0F, 3.0F});

  const Tensor result = std::move(tensor).min({0});

  expect_state(result, {2}, {1}, {-1.0F, 2.0F},
               "rvalue axis min produced an incorrect shape",
               "rvalue axis min produced incorrect strides",
               "rvalue axis min produced incorrect values");
  expect_state(tensor, {2, 2}, {2, 1}, {4.0F, 2.0F, -1.0F, 3.0F},
               "axis min consumed its rvalue source shape",
               "axis min consumed its rvalue source strides",
               "axis min consumed its rvalue source elements");
}

void test_min_rejects_moved_from_sentinel() {
  Tensor source = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor owner(std::move(source));
  static_cast<void>(owner);

  expect_throws<std::invalid_argument>(
      [&source] { static_cast<void>(source.min({})); },
      "axis min accepted a moved-from sentinel",
      "axis min of a moved-from sentinel produced the wrong exception type");
  expect_empty_sentinel(source);
}

}  // namespace

int main() {
  try {
    test_min_reduces_one_axis_without_changing_source();
    test_min_reduces_multiple_axes_independently_of_axis_order();
    test_min_keepdims_preserves_reduced_axis_positions();
    test_min_all_axes_returns_scalar_or_all_singleton_shape();
    test_min_empty_axes_returns_independent_copy();
    test_min_handles_scalar_and_singleton_axes();
    test_min_propagates_nan_only_to_affected_outputs();
    test_min_distinguishes_empty_domains_from_empty_results();
    test_min_rejects_invalid_axes_without_changing_source();
    test_min_accepts_rvalue_without_consuming_source();
    test_min_rejects_moved_from_sentinel();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor axis min tests passed\n";
  return EXIT_SUCCESS;
}
