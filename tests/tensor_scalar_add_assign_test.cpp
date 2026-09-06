#include <algorithm>
#include <concepts>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "nn/tensor.hpp"

namespace {

using Tensor = nn::Tensor<float>;
using DoubleTensor = nn::Tensor<double>;
using IntegerTensor = nn::Tensor<int>;

template <typename TensorType, typename ScalarType>
concept CanScalarAddAssign = requires {
  std::declval<TensorType>() += std::declval<ScalarType>();
};

static_assert(std::same_as<decltype(std::declval<Tensor&>() +=
                                    std::declval<float>()),
                           Tensor&>);
static_assert(CanScalarAddAssign<DoubleTensor&, double>);
static_assert(CanScalarAddAssign<Tensor&, int>);
static_assert(!CanScalarAddAssign<const Tensor&, float>);
static_assert(!CanScalarAddAssign<Tensor&&, float>);
static_assert(!CanScalarAddAssign<const Tensor&&, float>);
static_assert(!CanScalarAddAssign<IntegerTensor&, int>);
static_assert(!noexcept(std::declval<Tensor&>() += std::declval<float>()));

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

void test_scalar_add_assign_updates_every_element_and_returns_left_operand() {
  Tensor tensor =
      Tensor::from_data({2, 3}, {1.0F, -2.0F, 3.5F, 0.0F, 8.0F, -4.5F});
  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_elements{3.0F, 0.0F, 5.5F,
                                                2.0F, 10.0F, -2.5F};

  Tensor& result = (tensor += 2.0F);

  expect(&result == &tensor,
         "scalar addition assignment returned the wrong object");
  expect_state(tensor, expected_shape, expected_strides, expected_elements,
               "scalar addition assignment changed shape",
               "scalar addition assignment changed strides",
               "scalar addition assignment computed incorrect elements");
}

void test_scalar_add_assign_supports_rank_zero_tensor() {
  Tensor tensor = Tensor::scalar(2.5F);

  tensor += -0.5F;

  expect(tensor.rank() == 0,
         "scalar addition assignment changed rank-zero tensor rank");
  expect(tensor.numel() == 1,
         "scalar addition assignment changed rank-zero tensor numel");
  expect(tensor.at() == 2.0F,
         "scalar addition assignment computed an incorrect rank-zero value");
}

void test_scalar_add_assign_supports_zero_extent_tensor() {
  Tensor tensor({2, 0, 4});
  const Tensor::shape_type expected_shape{2, 0, 4};
  const Tensor::strides_type expected_strides{0, 4, 1};

  tensor += 3.0F;

  expect_state(tensor, expected_shape, expected_strides, {},
               "zero-extent scalar addition assignment changed shape",
               "zero-extent scalar addition assignment changed strides",
               "zero-extent scalar addition assignment created elements");
}

void test_scalar_add_assign_copies_aliased_value_before_mutation() {
  Tensor tensor = Tensor::from_data({3}, {2.0F, 4.0F, 8.0F});
  const Tensor::storage_type expected_elements{4.0F, 6.0F, 10.0F};

  tensor += tensor.elements().front();

  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "scalar addition assignment did not preserve an aliased value");
}

void test_scalar_add_assign_accepts_convertible_number() {
  Tensor tensor = Tensor::from_data({2}, {1.5F, -2.5F});
  const Tensor::storage_type expected_elements{3.5F, -0.5F};

  tensor += 2;

  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "scalar addition assignment converted a number incorrectly");
}

void test_scalar_add_assign_rejects_moved_from_sentinel_without_change() {
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
      [&source] { source += 1.0F; },
      "moved-from scalar addition assignment did not throw "
      "std::invalid_argument",
      "moved-from scalar addition assignment produced the wrong exception "
      "type");

  expect_state(
      source, original_shape, original_strides, original_elements,
      "failed scalar addition assignment changed the sentinel shape",
      "failed scalar addition assignment changed the sentinel strides",
      "failed scalar addition assignment changed the sentinel elements");
}

}  // namespace

int main() {
  try {
    test_scalar_add_assign_updates_every_element_and_returns_left_operand();
    test_scalar_add_assign_supports_rank_zero_tensor();
    test_scalar_add_assign_supports_zero_extent_tensor();
    test_scalar_add_assign_copies_aliased_value_before_mutation();
    test_scalar_add_assign_accepts_convertible_number();
    test_scalar_add_assign_rejects_moved_from_sentinel_without_change();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor scalar addition assignment tests passed\n";
  return EXIT_SUCCESS;
}
