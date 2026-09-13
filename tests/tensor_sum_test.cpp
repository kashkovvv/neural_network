#include <algorithm>
#include <cmath>
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
using DoubleTensor = nn::Tensor<double>;
using IntegerTensor = nn::Tensor<int>;

template <typename TensorType>
concept CanSum = requires { std::declval<TensorType>().sum(); };

static_assert(
    std::same_as<decltype(std::declval<const Tensor&>().sum()), Tensor>);
static_assert(CanSum<Tensor&>);
static_assert(CanSum<const Tensor&>);
static_assert(CanSum<Tensor&&>);
static_assert(CanSum<const Tensor&&>);
static_assert(CanSum<DoubleTensor&>);
static_assert(!CanSum<IntegerTensor&>);
static_assert(!noexcept(std::declval<const Tensor&>().sum()));

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

void expect_scalar(const Tensor& tensor, Tensor::value_type expected_value,
                   const char* value_message) {
  expect(tensor.rank() == 0, "sum result rank is not zero");
  expect(tensor.numel() == 1, "sum result numel is not one");
  expect(tensor.shape().empty(), "sum result shape is not empty");
  expect(tensor.strides().empty(), "sum result strides are not empty");
  expect(tensor.at() == expected_value, value_message);
}

void expect_empty_sentinel(const Tensor& tensor) {
  expect(tensor.rank() == 0, "sentinel rank is not zero");
  expect(tensor.numel() == 0, "sentinel numel is not zero");
  expect(tensor.shape().empty(), "sentinel shape is not empty");
  expect(tensor.strides().empty(), "sentinel strides are not empty");
  expect(tensor.elements().empty(), "sentinel elements are not empty");
}

void test_sum_reduces_all_elements_without_changing_source() {
  const Tensor tensor =
      Tensor::from_data({2, 3}, {1.5F, -2.0F, 3.0F, 4.5F, -1.0F, 2.0F});
  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_elements{1.5F, -2.0F, 3.0F,
                                                4.5F, -1.0F, 2.0F};

  const Tensor result = tensor.sum();

  expect_scalar(result, 8.0F, "sum produced an incorrect value");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "sum changed the source shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "sum changed the source strides");
  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "sum changed the source elements");
}

void test_sum_supports_rank_zero_and_singleton_tensors() {
  const Tensor scalar_result = Tensor::scalar(-3.5F).sum();
  const Tensor singleton_result =
      Tensor::from_data({1, 1, 1}, {7.0F}).sum();

  expect_scalar(scalar_result, -3.5F,
                "sum changed a rank-zero tensor value");
  expect_scalar(singleton_result, 7.0F,
                "sum changed a singleton tensor value");
}

void test_sum_returns_additive_identity_for_zero_extent_tensors() {
  const Tensor one_axis_result = Tensor({0}).sum();
  const Tensor multiple_axis_result = Tensor({2, 0, 3}).sum();

  expect_scalar(one_axis_result, 0.0F,
                "sum of a zero-extent vector is not additive identity");
  expect_scalar(multiple_axis_result, 0.0F,
                "sum of a zero-extent tensor is not additive identity");
}

void test_sum_preserves_small_terms_during_cancellation() {
  const Tensor positive_large_first =
      Tensor::from_data({3}, {1.0e20F, 1.0F, -1.0e20F});
  const Tensor negative_large_first =
      Tensor::from_data({3}, {-1.0e20F, 1.0F, 1.0e20F});

  expect_scalar(positive_large_first.sum(), 1.0F,
                "sum lost a small positive term during cancellation");
  expect_scalar(negative_large_first.sum(), 1.0F,
                "sum lost a small term after negative cancellation");
}

void test_sum_uses_native_floating_point_behavior() {
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const Tensor infinity_result =
      Tensor::from_data({2}, {std::numeric_limits<float>::max(),
                              std::numeric_limits<float>::max()})
          .sum();
  const Tensor explicit_infinity_result =
      Tensor::from_data({3}, {1.0F, infinity, 2.0F}).sum();
  const Tensor negative_infinity_result =
      Tensor::from_data({2}, {-infinity, -1.0F}).sum();
  const Tensor opposite_infinities_result =
      Tensor::from_data({2}, {infinity, -infinity}).sum();
  const Tensor explicit_nan_result =
      Tensor::from_data({3}, {1.0F, nan, 2.0F}).sum();

  expect(std::isinf(infinity_result.at()) && infinity_result.at() > 0.0F,
         "sum did not preserve native floating-point overflow behavior");
  expect(std::isinf(explicit_infinity_result.at()) &&
             explicit_infinity_result.at() > 0.0F,
         "sum changed positive infinity into another value");
  expect(std::isinf(negative_infinity_result.at()) &&
             negative_infinity_result.at() < 0.0F,
         "sum changed negative infinity into another value");
  expect(std::isnan(opposite_infinities_result.at()),
         "sum of opposite infinities is not NaN");
  expect(std::isnan(explicit_nan_result.at()),
         "sum did not propagate an explicit NaN");
}

void test_sum_accepts_rvalue_without_consuming_source() {
  Tensor tensor = Tensor::from_data({3}, {1.0F, 2.0F, 3.0F});
  const Tensor::storage_type expected_elements{1.0F, 2.0F, 3.0F};

  const Tensor result = std::move(tensor).sum();

  expect_scalar(result, 6.0F, "rvalue sum produced an incorrect value");
  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "sum consumed its rvalue source");
}

void test_sum_rejects_moved_from_sentinel() {
  Tensor source = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor owner(std::move(source));
  static_cast<void>(owner);

  expect_throws<std::invalid_argument>(
      [&source] { static_cast<void>(source.sum()); },
      "sum accepted a moved-from sentinel",
      "sum of a moved-from sentinel produced the wrong exception type");
  expect_empty_sentinel(source);
}

}  // namespace

int main() {
  try {
    test_sum_reduces_all_elements_without_changing_source();
    test_sum_supports_rank_zero_and_singleton_tensors();
    test_sum_returns_additive_identity_for_zero_extent_tensors();
    test_sum_preserves_small_terms_during_cancellation();
    test_sum_uses_native_floating_point_behavior();
    test_sum_accepts_rvalue_without_consuming_source();
    test_sum_rejects_moved_from_sentinel();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor sum tests passed\n";
  return EXIT_SUCCESS;
}
