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

template <typename TensorType, typename ScalarType>
concept CanScalarMultiplyAssign = requires {
  std::declval<TensorType>() *= std::declval<ScalarType>();
};

static_assert(std::same_as<decltype(std::declval<Tensor&>() *=
                                    std::declval<float>()),
                           Tensor&>);
static_assert(CanScalarMultiplyAssign<DoubleTensor&, double>);
static_assert(CanScalarMultiplyAssign<Tensor&, int>);
static_assert(!CanScalarMultiplyAssign<const Tensor&, float>);
static_assert(!CanScalarMultiplyAssign<Tensor&&, float>);
static_assert(!CanScalarMultiplyAssign<const Tensor&&, float>);
static_assert(!CanScalarMultiplyAssign<IntegerTensor&, int>);
static_assert(!noexcept(std::declval<Tensor&>() *= std::declval<float>()));

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

void expect_state(const Tensor& tensor, const Tensor::shape_type& shape,
                  const Tensor::strides_type& strides,
                  const Tensor::storage_type& elements,
                  const char* shape_message, const char* strides_message,
                  const char* elements_message) {
  expect(std::ranges::equal(tensor.shape(), shape), shape_message);
  expect(std::ranges::equal(tensor.strides(), strides), strides_message);
  expect(std::ranges::equal(tensor.elements(), elements), elements_message);
}

void test_scalar_multiply_assign_updates_elements_and_returns_left_operand() {
  Tensor tensor =
      Tensor::from_data({2, 3}, {1.0F, -2.0F, 3.5F, 0.0F, 8.0F, -4.5F});
  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_elements{2.0F, -4.0F, 7.0F,
                                                0.0F, 16.0F, -9.0F};

  Tensor& result = (tensor *= 2.0F);

  expect(&result == &tensor,
         "scalar multiplication assignment returned the wrong object");
  expect_state(tensor, expected_shape, expected_strides, expected_elements,
               "scalar multiplication assignment changed shape",
               "scalar multiplication assignment changed strides",
               "scalar multiplication assignment computed incorrect elements");
}

void test_scalar_multiply_assign_preserves_native_special_value_semantics() {
  const float infinity = std::numeric_limits<float>::infinity();
  Tensor tensor = Tensor::from_data({3}, {infinity, -infinity, -0.0F});

  tensor *= 0.0F;

  expect(std::isnan(tensor.elements()[0]),
         "scalar multiplication assignment handled positive infinity times "
         "zero incorrectly");
  expect(std::isnan(tensor.elements()[1]),
         "scalar multiplication assignment handled negative infinity times "
         "zero incorrectly");
  expect(tensor.elements()[2] == 0.0F &&
             std::signbit(tensor.elements()[2]),
         "scalar multiplication assignment handled negative zero incorrectly");
}

void test_scalar_multiply_assign_supports_rank_zero_and_zero_extent_tensors() {
  Tensor rank_zero = Tensor::scalar(-2.5F);

  rank_zero *= -2.0F;

  expect(rank_zero.rank() == 0,
         "scalar multiplication assignment changed rank-zero tensor rank");
  expect(rank_zero.numel() == 1,
         "scalar multiplication assignment changed rank-zero tensor numel");
  expect(rank_zero.at() == 5.0F,
         "scalar multiplication assignment computed an incorrect rank-zero value");

  Tensor zero_extent({2, 0, 4});
  const Tensor::shape_type expected_shape{2, 0, 4};
  const Tensor::strides_type expected_strides{0, 4, 1};

  zero_extent *= 3.0F;

  expect_state(zero_extent, expected_shape, expected_strides, {},
               "zero-extent scalar multiplication assignment changed shape",
               "zero-extent scalar multiplication assignment changed strides",
               "zero-extent scalar multiplication assignment created elements");
}

void test_scalar_multiply_assign_copies_aliased_value_before_mutation() {
  Tensor tensor = Tensor::from_data({3}, {2.0F, 4.0F, 8.0F});
  const Tensor::storage_type expected_elements{4.0F, 8.0F, 16.0F};

  tensor *= tensor.elements().front();

  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "scalar multiplication assignment did not preserve an aliased value");
}

void test_scalar_multiply_assign_accepts_convertible_number() {
  Tensor tensor = Tensor::from_data({2}, {1.5F, -2.5F});
  const Tensor::storage_type expected_elements{3.0F, -5.0F};

  tensor *= 2;

  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "scalar multiplication assignment converted a number incorrectly");
}

void test_scalar_multiply_assign_rejects_moved_from_sentinel_without_change() {
  Tensor source = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor owner(std::move(source));
  static_cast<void>(owner);

  const Tensor::shape_type original_shape(source.shape().begin(),
                                           source.shape().end());
  const Tensor::strides_type original_strides(source.strides().begin(),
                                               source.strides().end());
  const Tensor::storage_type original_elements(source.elements().begin(),
                                                source.elements().end());

  expect_throws<std::invalid_argument>(
      [&source] { source *= 2.0F; },
      "moved-from scalar multiplication assignment did not throw "
      "std::invalid_argument",
      "moved-from scalar multiplication assignment produced the wrong "
      "exception type");

  expect_state(
      source, original_shape, original_strides, original_elements,
      "failed scalar multiplication assignment changed the sentinel shape",
      "failed scalar multiplication assignment changed the sentinel strides",
      "failed scalar multiplication assignment changed the sentinel elements");
}

}  // namespace

int main() {
  try {
    test_scalar_multiply_assign_updates_elements_and_returns_left_operand();
    test_scalar_multiply_assign_preserves_native_special_value_semantics();
    test_scalar_multiply_assign_supports_rank_zero_and_zero_extent_tensors();
    test_scalar_multiply_assign_copies_aliased_value_before_mutation();
    test_scalar_multiply_assign_accepts_convertible_number();
    test_scalar_multiply_assign_rejects_moved_from_sentinel_without_change();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor scalar multiplication assignment tests passed\n";
  return EXIT_SUCCESS;
}
