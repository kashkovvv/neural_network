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

template <typename Left, typename Right>
concept CanSubtractAssign = requires {
  std::declval<Left>() -= std::declval<Right>();
};

static_assert(std::same_as<decltype(std::declval<Tensor&>() -=
                                    std::declval<const Tensor&>()),
                           Tensor&>);
static_assert(CanSubtractAssign<DoubleTensor&, const DoubleTensor&>);
static_assert(!CanSubtractAssign<const Tensor&, const Tensor&>);
static_assert(!CanSubtractAssign<Tensor&&, const Tensor&>);
static_assert(!CanSubtractAssign<const Tensor&&, const Tensor&>);
static_assert(
    !CanSubtractAssign<IntegerTensor&, const IntegerTensor&>);
static_assert(!CanSubtractAssign<Tensor&, const DoubleTensor&>);
static_assert(
    !noexcept(std::declval<Tensor&>() -= std::declval<const Tensor&>()));

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

void expect_subtract_assign_rejected_without_left_change(
    Tensor& left, const Tensor& right) {
  const Tensor::shape_type original_shape(left.shape().begin(),
                                           left.shape().end());
  const Tensor::strides_type original_strides(left.strides().begin(),
                                               left.strides().end());
  const Tensor::storage_type original_elements(left.elements().begin(),
                                                left.elements().end());

  expect_throws<std::invalid_argument>(
      [&left, &right] { left -= right; },
      "invalid tensor subtraction assignment did not throw "
      "std::invalid_argument",
      "invalid tensor subtraction assignment produced the wrong exception "
      "type");

  expect_state(left, original_shape, original_strides, original_elements,
               "failed subtraction assignment changed the left shape",
               "failed subtraction assignment changed the left strides",
               "failed subtraction assignment changed the left elements");
}

void test_subtract_assign_computes_elementwise_and_returns_left_operand() {
  Tensor left =
      Tensor::from_data({2, 3}, {9.0F, 7.0F, 5.0F, 3.0F, 1.0F, -1.0F});
  const Tensor right =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_elements{8.0F, 5.0F, 2.0F,
                                                -1.0F, -4.0F, -7.0F};
  const Tensor::storage_type expected_right_elements{1.0F, 2.0F, 3.0F,
                                                      4.0F, 5.0F, 6.0F};

  Tensor& result = (left -= right);

  expect(&result == &left,
         "subtraction assignment returned the wrong object");
  expect_state(left, expected_shape, expected_strides, expected_elements,
               "subtraction assignment changed the left shape",
               "subtraction assignment changed the left strides",
               "subtraction assignment computed incorrect elements");
  expect(std::ranges::equal(right.elements(), expected_right_elements),
         "subtraction assignment changed the right elements");
}

void test_subtract_assign_supports_scalars() {
  Tensor left = Tensor::scalar(2.5F);
  const Tensor right = Tensor::scalar(-0.5F);

  left -= right;

  expect(left.rank() == 0, "scalar subtraction assignment changed rank");
  expect(left.numel() == 1, "scalar subtraction assignment changed numel");
  expect(left.at() == 3.0F,
         "scalar subtraction assignment computed an incorrect value");
}

void test_subtract_assign_supports_matching_zero_extent_tensors() {
  Tensor left({2, 0, 4});
  const Tensor right({2, 0, 4});
  const Tensor::shape_type expected_shape{2, 0, 4};
  const Tensor::strides_type expected_strides{0, 4, 1};

  left -= right;

  expect_state(left, expected_shape, expected_strides, {},
               "zero-extent subtraction assignment changed shape",
               "zero-extent subtraction assignment changed strides",
               "zero-extent subtraction assignment created elements");
}

void test_subtract_assign_supports_self_aliasing() {
  Tensor tensor = Tensor::from_data({2, 2}, {1.0F, -2.0F, 3.5F, 0.0F});
  const Tensor::storage_type expected_elements{0.0F, 0.0F, 0.0F, 0.0F};

  tensor -= tensor;

  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "self subtraction assignment computed incorrect elements");
}

void test_subtract_assign_rejects_different_shapes() {
  Tensor left =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor permuted =
      Tensor::from_data({3, 2}, {6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F});
  const Tensor broadcastable = Tensor::from_data({1, 3}, {1.0F, 2.0F, 3.0F});

  expect_subtract_assign_rejected_without_left_change(left, permuted);
  expect_subtract_assign_rejected_without_left_change(left, broadcastable);
}

void test_subtract_assign_rejects_moved_from_sentinels() {
  Tensor right_source = Tensor::from_data({1}, {1.0F});
  Tensor right_owner(std::move(right_source));
  static_cast<void>(right_owner);

  Tensor ordinary_left = Tensor::from_data({1}, {2.0F});
  expect_subtract_assign_rejected_without_left_change(ordinary_left,
                                                      right_source);

  Tensor left_source = Tensor::from_data({1}, {3.0F});
  Tensor left_owner(std::move(left_source));
  static_cast<void>(left_owner);
  const Tensor ordinary_right = Tensor::from_data({1}, {4.0F});
  expect_subtract_assign_rejected_without_left_change(left_source,
                                                      ordinary_right);

  Tensor second_source = Tensor::from_data({1}, {5.0F});
  Tensor second_owner(std::move(second_source));
  static_cast<void>(second_owner);
  expect_subtract_assign_rejected_without_left_change(left_source,
                                                      second_source);
}

}  // namespace

int main() {
  try {
    test_subtract_assign_computes_elementwise_and_returns_left_operand();
    test_subtract_assign_supports_scalars();
    test_subtract_assign_supports_matching_zero_extent_tensors();
    test_subtract_assign_supports_self_aliasing();
    test_subtract_assign_rejects_different_shapes();
    test_subtract_assign_rejects_moved_from_sentinels();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor subtraction assignment tests passed\n";
  return EXIT_SUCCESS;
}
