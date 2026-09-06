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
concept CanNegate = requires { -std::declval<TensorType>(); };

template <typename TensorType>
concept CanCallFreeNegate = requires {
  operator-(std::declval<TensorType>());
};

static_assert(
    std::same_as<decltype(-std::declval<const Tensor&>()), Tensor>);
static_assert(CanCallFreeNegate<const Tensor&>);
static_assert(CanNegate<Tensor&>);
static_assert(CanNegate<const Tensor&>);
static_assert(CanNegate<Tensor&&>);
static_assert(CanNegate<const Tensor&&>);
static_assert(CanNegate<DoubleTensor&>);
static_assert(!CanNegate<IntegerTensor&>);
static_assert(!noexcept(-std::declval<const Tensor&>()));

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

void expect_empty_sentinel(const Tensor& tensor) {
  expect(tensor.rank() == 0, "sentinel rank is not zero");
  expect(tensor.numel() == 0, "sentinel numel is not zero");
  expect(tensor.shape().empty(), "sentinel shape is not empty");
  expect(tensor.strides().empty(), "sentinel strides are not empty");
  expect(tensor.elements().empty(), "sentinel elements are not empty");
}

void expect_state(const Tensor& tensor, const Tensor::shape_type& shape,
                  const Tensor::strides_type& strides,
                  const Tensor::storage_type& elements,
                  const char* shape_message, const char* strides_message,
                  const char* elements_message) {
  expect(std::ranges::equal(tensor.shape(), shape), shape_message);
  expect(std::ranges::equal(tensor.strides(), strides), strides_message);
  expect(std::ranges::equal(tensor.elements(), elements), elements_message);
}

void test_unary_minus_negates_elements_without_changing_lvalue() {
  const Tensor tensor =
      Tensor::from_data({2, 2}, {1.0F, -2.0F, 3.5F, 0.0F});
  const Tensor::shape_type expected_shape{2, 2};
  const Tensor::strides_type expected_strides{2, 1};
  const Tensor::storage_type expected_elements{-1.0F, 2.0F, -3.5F, -0.0F};
  const Tensor::storage_type expected_source{1.0F, -2.0F, 3.5F, 0.0F};

  const Tensor result = -tensor;

  expect_state(result, expected_shape, expected_strides, expected_elements,
               "unary minus changed shape",
               "unary minus changed strides",
               "unary minus computed incorrect elements");
  expect(std::ranges::equal(tensor.elements(), expected_source),
         "unary minus changed the tensor lvalue");
  expect(std::signbit(result.elements().back()),
         "unary minus did not preserve the sign of negative zero");
}

void test_unary_minus_handles_special_floating_point_values() {
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const Tensor tensor =
      Tensor::from_data({4}, {infinity, -infinity, 0.0F, nan});

  const Tensor result = -tensor;

  expect(std::isinf(result.elements()[0]) &&
             std::signbit(result.elements()[0]),
         "unary minus handled positive infinity incorrectly");
  expect(std::isinf(result.elements()[1]) &&
             !std::signbit(result.elements()[1]),
         "unary minus handled negative infinity incorrectly");
  expect(result.elements()[2] == 0.0F &&
             std::signbit(result.elements()[2]),
         "unary minus handled positive zero incorrectly");
  expect(std::isnan(result.elements()[3]),
         "unary minus handled NaN incorrectly");
}

void test_unary_minus_supports_rank_zero_and_zero_extent_tensors() {
  const Tensor rank_zero_result = -Tensor::scalar(2.5F);

  expect(rank_zero_result.rank() == 0,
         "unary minus changed rank-zero tensor rank");
  expect(rank_zero_result.numel() == 1,
         "unary minus changed rank-zero tensor numel");
  expect(rank_zero_result.at() == -2.5F,
         "unary minus computed an incorrect rank-zero value");

  const Tensor zero_extent_result = -Tensor({2, 0, 4});
  const Tensor::shape_type expected_shape{2, 0, 4};
  const Tensor::strides_type expected_strides{0, 4, 1};

  expect_state(zero_extent_result, expected_shape, expected_strides, {},
               "zero-extent unary minus changed shape",
               "zero-extent unary minus changed strides",
               "zero-extent unary minus created elements");
}

void test_unary_minus_reuses_rvalue_storage() {
  Tensor source = Tensor::from_data({3}, {1.0F, -2.0F, 3.0F});
  const Tensor::value_type* const original_storage = source.elements().data();
  const Tensor::storage_type expected_elements{-1.0F, 2.0F, -3.0F};

  const Tensor result = -std::move(source);

  expect(result.elements().data() == original_storage,
         "unary minus did not reuse rvalue storage");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "rvalue unary minus computed incorrect elements");
  expect_empty_sentinel(source);
}

void test_unary_minus_rejects_moved_from_sentinel() {
  Tensor source = Tensor::from_data({1}, {1.0F});
  Tensor owner(std::move(source));
  static_cast<void>(owner);

  expect_throws<std::invalid_argument>(
      [&source] { static_cast<void>(-source); },
      "unary minus accepted an empty sentinel",
      "sentinel unary minus produced the wrong exception type");
  expect_empty_sentinel(source);
}

}  // namespace

int main() {
  try {
    test_unary_minus_negates_elements_without_changing_lvalue();
    test_unary_minus_handles_special_floating_point_values();
    test_unary_minus_supports_rank_zero_and_zero_extent_tensors();
    test_unary_minus_reuses_rvalue_storage();
    test_unary_minus_rejects_moved_from_sentinel();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor unary minus tests passed\n";
  return EXIT_SUCCESS;
}
