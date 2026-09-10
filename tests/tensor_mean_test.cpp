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
concept CanMean = requires { std::declval<TensorType>().mean(); };

static_assert(
    std::same_as<decltype(std::declval<const Tensor&>().mean()), Tensor>);
static_assert(CanMean<Tensor&>);
static_assert(CanMean<const Tensor&>);
static_assert(CanMean<Tensor&&>);
static_assert(CanMean<const Tensor&&>);
static_assert(CanMean<DoubleTensor&>);
static_assert(!CanMean<IntegerTensor&>);
static_assert(!noexcept(std::declval<const Tensor&>().mean()));

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
  expect(tensor.rank() == 0, "mean result rank is not zero");
  expect(tensor.numel() == 1, "mean result numel is not one");
  expect(tensor.shape().empty(), "mean result shape is not empty");
  expect(tensor.strides().empty(), "mean result strides are not empty");
  expect(tensor.at() == expected_value, value_message);
}

void expect_nan_scalar(const Tensor& tensor, const char* value_message) {
  expect(tensor.rank() == 0, "mean result rank is not zero");
  expect(tensor.numel() == 1, "mean result numel is not one");
  expect(tensor.shape().empty(), "mean result shape is not empty");
  expect(tensor.strides().empty(), "mean result strides are not empty");
  expect(std::isnan(tensor.at()), value_message);
}

void expect_empty_sentinel(const Tensor& tensor) {
  expect(tensor.rank() == 0, "sentinel rank is not zero");
  expect(tensor.numel() == 0, "sentinel numel is not zero");
  expect(tensor.shape().empty(), "sentinel shape is not empty");
  expect(tensor.strides().empty(), "sentinel strides are not empty");
  expect(tensor.elements().empty(), "sentinel elements are not empty");
}

void test_mean_reduces_all_elements_without_changing_source() {
  const Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 3.0F, 5.0F, 7.0F});
  const Tensor::shape_type expected_shape{2, 2};
  const Tensor::strides_type expected_strides{2, 1};
  const Tensor::storage_type expected_elements{1.0F, 3.0F, 5.0F, 7.0F};

  const Tensor result = tensor.mean();

  expect_scalar(result, 4.0F, "mean produced an incorrect value");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "mean changed the source shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "mean changed the source strides");
  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "mean changed the source elements");
}

void test_mean_supports_rank_zero_and_singleton_tensors() {
  const Tensor scalar_result = Tensor::scalar(-3.5F).mean();
  const Tensor singleton_result = Tensor::from_data({1, 1, 1}, {7.0F}).mean();

  expect_scalar(scalar_result, -3.5F, "mean changed a rank-zero tensor value");
  expect_scalar(singleton_result, 7.0F,
                "mean changed a singleton tensor value");
}

void test_mean_returns_nan_for_empty_reduction_domain() {
  const Tensor one_axis_result = Tensor({0}).mean();
  const Tensor multiple_axis_result = Tensor({2, 0, 3}).mean();

  expect_nan_scalar(one_axis_result,
                    "mean of an empty vector did not produce NaN");
  expect_nan_scalar(multiple_axis_result,
                    "mean of an empty tensor did not produce NaN");
}

void test_mean_uses_native_floating_point_behavior() {
  const Tensor infinity_result =
      Tensor::from_data({2}, {std::numeric_limits<float>::infinity(), 1.0F})
          .mean();
  const Tensor nan_result =
      Tensor::from_data({2}, {std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()})
          .mean();

  expect(std::isinf(infinity_result.at()) && infinity_result.at() > 0.0F,
         "mean did not preserve native floating-point infinity behavior");
  expect(std::isnan(nan_result.at()),
         "mean did not preserve native floating-point NaN behavior");
}

void test_mean_accepts_rvalue_without_consuming_source() {
  Tensor tensor = Tensor::from_data({3}, {1.0F, 2.0F, 6.0F});
  const Tensor::shape_type expected_shape{3};
  const Tensor::strides_type expected_strides{1};
  const Tensor::storage_type expected_elements{1.0F, 2.0F, 6.0F};

  const Tensor result = std::move(tensor).mean();

  expect_scalar(result, 3.0F, "rvalue mean produced an incorrect value");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "mean changed its rvalue source shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "mean changed its rvalue source strides");
  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "mean changed its rvalue source elements");
}

void test_mean_rejects_moved_from_sentinel() {
  Tensor source = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor owner(std::move(source));
  static_cast<void>(owner);

  expect_throws<std::invalid_argument>(
      [&source] { static_cast<void>(source.mean()); },
      "mean accepted a moved-from sentinel",
      "mean of a moved-from sentinel produced the wrong exception type");
  expect_empty_sentinel(source);
}

}  // namespace

int main() {
  try {
    test_mean_reduces_all_elements_without_changing_source();
    test_mean_supports_rank_zero_and_singleton_tensors();
    test_mean_returns_nan_for_empty_reduction_domain();
    test_mean_uses_native_floating_point_behavior();
    test_mean_accepts_rvalue_without_consuming_source();
    test_mean_rejects_moved_from_sentinel();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor mean tests passed\n";
  return EXIT_SUCCESS;
}
