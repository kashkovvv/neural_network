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
concept CanMin = requires { std::declval<TensorType>().min(); };

static_assert(
    std::same_as<decltype(std::declval<const Tensor&>().min()), Tensor>);
static_assert(CanMin<Tensor&>);
static_assert(CanMin<const Tensor&>);
static_assert(CanMin<Tensor&&>);
static_assert(CanMin<const Tensor&&>);
static_assert(CanMin<DoubleTensor&>);
static_assert(!CanMin<IntegerTensor&>);
static_assert(!noexcept(std::declval<const Tensor&>().min()));

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
  expect(tensor.rank() == 0, "min result rank is not zero");
  expect(tensor.numel() == 1, "min result numel is not one");
  expect(tensor.shape().empty(), "min result shape is not empty");
  expect(tensor.strides().empty(), "min result strides are not empty");
  expect(tensor.at() == expected_value, value_message);
}

void expect_nan_scalar(const Tensor& tensor, const char* value_message) {
  expect(tensor.rank() == 0, "min result rank is not zero");
  expect(tensor.numel() == 1, "min result numel is not one");
  expect(tensor.shape().empty(), "min result shape is not empty");
  expect(tensor.strides().empty(), "min result strides are not empty");
  expect(std::isnan(tensor.at()), value_message);
}

void expect_empty_sentinel(const Tensor& tensor) {
  expect(tensor.rank() == 0, "sentinel rank is not zero");
  expect(tensor.numel() == 0, "sentinel numel is not zero");
  expect(tensor.shape().empty(), "sentinel shape is not empty");
  expect(tensor.strides().empty(), "sentinel strides are not empty");
  expect(tensor.elements().empty(), "sentinel elements are not empty");
}

void test_min_reduces_all_elements_without_changing_source() {
  const Tensor tensor =
      Tensor::from_data({2, 3}, {4.0F, -2.5F, 7.0F, 0.0F, -1.0F, 3.0F});
  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_elements{4.0F, -2.5F, 7.0F,
                                               0.0F, -1.0F, 3.0F};

  const Tensor result = tensor.min();

  expect_scalar(result, -2.5F, "min produced an incorrect value");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "min changed the source shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "min changed the source strides");
  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "min changed the source elements");
}

void test_min_supports_rank_zero_singleton_and_equal_values() {
  const Tensor scalar_result = Tensor::scalar(-3.5F).min();
  const Tensor singleton_result = Tensor::from_data({1, 1, 1}, {7.0F}).min();
  const Tensor equal_result = Tensor::from_data({3}, {2.0F, 2.0F, 2.0F}).min();

  expect_scalar(scalar_result, -3.5F, "min changed a rank-zero tensor value");
  expect_scalar(singleton_result, 7.0F, "min changed a singleton tensor value");
  expect_scalar(equal_result, 2.0F, "min changed equal tensor values");
}

void test_min_propagates_nan_independently_of_position() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const Tensor leading_nan_result = Tensor::from_data({2}, {nan, -1.0F}).min();
  const Tensor trailing_nan_result = Tensor::from_data({2}, {-1.0F, nan}).min();

  expect_nan_scalar(leading_nan_result, "min did not propagate a leading NaN");
  expect_nan_scalar(trailing_nan_result,
                    "min did not propagate a trailing NaN");
}

void test_min_handles_infinities() {
  const float infinity = std::numeric_limits<float>::infinity();
  const Tensor negative_infinity_result =
      Tensor::from_data({3}, {infinity, 4.0F, -infinity}).min();
  const Tensor positive_infinity_result =
      Tensor::from_data({2}, {infinity, infinity}).min();

  expect_scalar(negative_infinity_result, -infinity,
                "min did not select negative infinity");
  expect_scalar(positive_infinity_result, infinity,
                "min changed an all-infinity result");
}

void test_min_rejects_empty_reduction_domains_without_changing_source() {
  const Tensor vector({0});
  const Tensor tensor({2, 0, 3});

  expect_throws<std::domain_error>(
      [&vector] { static_cast<void>(vector.min()); },
      "min accepted an empty vector reduction domain",
      "min of an empty vector produced the wrong exception type");
  expect_throws<std::domain_error>(
      [&tensor] { static_cast<void>(tensor.min()); },
      "min accepted an empty multidimensional reduction domain",
      "min of an empty tensor produced the wrong exception type");
  expect(std::ranges::equal(vector.shape(), Tensor::shape_type{0}),
         "failed min changed the empty vector shape");
  expect(std::ranges::equal(vector.strides(), Tensor::strides_type{1}),
         "failed min changed the empty vector strides");
  expect(vector.elements().empty(), "failed min created empty vector elements");
  expect(std::ranges::equal(tensor.shape(), Tensor::shape_type{2, 0, 3}),
         "failed min changed the empty tensor shape");
  expect(std::ranges::equal(tensor.strides(), Tensor::strides_type{0, 3, 1}),
         "failed min changed the empty tensor strides");
  expect(tensor.elements().empty(), "failed min created empty tensor elements");
}

void test_min_accepts_rvalue_without_consuming_source() {
  Tensor tensor = Tensor::from_data({3}, {3.0F, -2.0F, 1.0F});
  const Tensor::shape_type expected_shape{3};
  const Tensor::strides_type expected_strides{1};
  const Tensor::storage_type expected_elements{3.0F, -2.0F, 1.0F};

  const Tensor result = std::move(tensor).min();

  expect_scalar(result, -2.0F, "rvalue min produced an incorrect value");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "min changed its rvalue source shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "min changed its rvalue source strides");
  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "min changed its rvalue source elements");
}

void test_min_rejects_moved_from_sentinel() {
  Tensor source = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor owner(std::move(source));
  static_cast<void>(owner);

  expect_throws<std::invalid_argument>(
      [&source] { static_cast<void>(source.min()); },
      "min accepted a moved-from sentinel",
      "min of a moved-from sentinel produced the wrong exception type");
  expect_empty_sentinel(source);
}

}  // namespace

int main() {
  try {
    test_min_reduces_all_elements_without_changing_source();
    test_min_supports_rank_zero_singleton_and_equal_values();
    test_min_propagates_nan_independently_of_position();
    test_min_handles_infinities();
    test_min_rejects_empty_reduction_domains_without_changing_source();
    test_min_accepts_rvalue_without_consuming_source();
    test_min_rejects_moved_from_sentinel();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor min tests passed\n";
  return EXIT_SUCCESS;
}
