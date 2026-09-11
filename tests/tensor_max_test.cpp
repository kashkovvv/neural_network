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
concept CanMax = requires { std::declval<TensorType>().max(); };

static_assert(
    std::same_as<decltype(std::declval<const Tensor&>().max()), Tensor>);
static_assert(CanMax<Tensor&>);
static_assert(CanMax<const Tensor&>);
static_assert(CanMax<Tensor&&>);
static_assert(CanMax<const Tensor&&>);
static_assert(CanMax<DoubleTensor&>);
static_assert(!CanMax<IntegerTensor&>);
static_assert(!noexcept(std::declval<const Tensor&>().max()));

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
  expect(tensor.rank() == 0, "max result rank is not zero");
  expect(tensor.numel() == 1, "max result numel is not one");
  expect(tensor.shape().empty(), "max result shape is not empty");
  expect(tensor.strides().empty(), "max result strides are not empty");
  expect(tensor.at() == expected_value, value_message);
}

void expect_nan_scalar(const Tensor& tensor, const char* value_message) {
  expect(tensor.rank() == 0, "max result rank is not zero");
  expect(tensor.numel() == 1, "max result numel is not one");
  expect(tensor.shape().empty(), "max result shape is not empty");
  expect(tensor.strides().empty(), "max result strides are not empty");
  expect(std::isnan(tensor.at()), value_message);
}

void expect_empty_sentinel(const Tensor& tensor) {
  expect(tensor.rank() == 0, "sentinel rank is not zero");
  expect(tensor.numel() == 0, "sentinel numel is not zero");
  expect(tensor.shape().empty(), "sentinel shape is not empty");
  expect(tensor.strides().empty(), "sentinel strides are not empty");
  expect(tensor.elements().empty(), "sentinel elements are not empty");
}

void test_max_reduces_all_elements_without_changing_source() {
  const Tensor tensor =
      Tensor::from_data({2, 3}, {4.0F, -2.5F, 7.0F, 0.0F, -1.0F, 3.0F});
  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_elements{4.0F, -2.5F, 7.0F,
                                               0.0F, -1.0F, 3.0F};

  const Tensor result = tensor.max();

  expect_scalar(result, 7.0F, "max produced an incorrect value");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "max changed the source shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "max changed the source strides");
  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "max changed the source elements");
}

void test_max_supports_rank_zero_singleton_and_equal_values() {
  const Tensor scalar_result = Tensor::scalar(-3.5F).max();
  const Tensor singleton_result = Tensor::from_data({1, 1, 1}, {7.0F}).max();
  const Tensor equal_result = Tensor::from_data({3}, {2.0F, 2.0F, 2.0F}).max();

  expect_scalar(scalar_result, -3.5F, "max changed a rank-zero tensor value");
  expect_scalar(singleton_result, 7.0F, "max changed a singleton tensor value");
  expect_scalar(equal_result, 2.0F, "max changed equal tensor values");
}

void test_max_propagates_nan_independently_of_position() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const Tensor leading_nan_result = Tensor::from_data({2}, {nan, -1.0F}).max();
  const Tensor trailing_nan_result = Tensor::from_data({2}, {-1.0F, nan}).max();

  expect_nan_scalar(leading_nan_result, "max did not propagate a leading NaN");
  expect_nan_scalar(trailing_nan_result,
                    "max did not propagate a trailing NaN");
}

void test_max_handles_infinities() {
  const float infinity = std::numeric_limits<float>::infinity();
  const Tensor positive_infinity_result =
      Tensor::from_data({3}, {-infinity, 4.0F, infinity}).max();
  const Tensor negative_infinity_result =
      Tensor::from_data({2}, {-infinity, -infinity}).max();

  expect_scalar(positive_infinity_result, infinity,
                "max did not select positive infinity");
  expect_scalar(negative_infinity_result, -infinity,
                "max changed an all-negative-infinity result");
}

void test_max_rejects_empty_reduction_domains_without_changing_source() {
  const Tensor vector({0});
  const Tensor tensor({2, 0, 3});

  expect_throws<std::domain_error>(
      [&vector] { static_cast<void>(vector.max()); },
      "max accepted an empty vector reduction domain",
      "max of an empty vector produced the wrong exception type");
  expect_throws<std::domain_error>(
      [&tensor] { static_cast<void>(tensor.max()); },
      "max accepted an empty multidimensional reduction domain",
      "max of an empty tensor produced the wrong exception type");
  expect(std::ranges::equal(vector.shape(), Tensor::shape_type{0}),
         "failed max changed the empty vector shape");
  expect(std::ranges::equal(vector.strides(), Tensor::strides_type{1}),
         "failed max changed the empty vector strides");
  expect(vector.elements().empty(), "failed max created empty vector elements");
  expect(std::ranges::equal(tensor.shape(), Tensor::shape_type{2, 0, 3}),
         "failed max changed the empty tensor shape");
  expect(std::ranges::equal(tensor.strides(), Tensor::strides_type{0, 3, 1}),
         "failed max changed the empty tensor strides");
  expect(tensor.elements().empty(), "failed max created empty tensor elements");
}

void test_max_accepts_rvalue_without_consuming_source() {
  Tensor tensor = Tensor::from_data({3}, {3.0F, -2.0F, 1.0F});
  const Tensor::shape_type expected_shape{3};
  const Tensor::strides_type expected_strides{1};
  const Tensor::storage_type expected_elements{3.0F, -2.0F, 1.0F};

  const Tensor result = std::move(tensor).max();

  expect_scalar(result, 3.0F, "rvalue max produced an incorrect value");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "max changed its rvalue source shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "max changed its rvalue source strides");
  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "max changed its rvalue source elements");
}

void test_max_rejects_moved_from_sentinel() {
  Tensor source = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor owner(std::move(source));
  static_cast<void>(owner);

  expect_throws<std::invalid_argument>(
      [&source] { static_cast<void>(source.max()); },
      "max accepted a moved-from sentinel",
      "max of a moved-from sentinel produced the wrong exception type");
  expect_empty_sentinel(source);
}

}  // namespace

int main() {
  try {
    test_max_reduces_all_elements_without_changing_source();
    test_max_supports_rank_zero_singleton_and_equal_values();
    test_max_propagates_nan_independently_of_position();
    test_max_handles_infinities();
    test_max_rejects_empty_reduction_domains_without_changing_source();
    test_max_accepts_rvalue_without_consuming_source();
    test_max_rejects_moved_from_sentinel();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor max tests passed\n";
  return EXIT_SUCCESS;
}
