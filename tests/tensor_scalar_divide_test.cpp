#include <algorithm>
#include <cmath>
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
concept CanDivide = requires {
  std::declval<Left>() / std::declval<Right>();
};

template <typename Left, typename Right>
concept CanCallFreeDivide = requires {
  operator/(std::declval<Left>(), std::declval<Right>());
};

static_assert(std::same_as<decltype(std::declval<const Tensor&>() /
                                    std::declval<float>()),
                           Tensor>);
static_assert(std::same_as<decltype(std::declval<float>() /
                                    std::declval<const Tensor&>()),
                           Tensor>);
static_assert(CanCallFreeDivide<const Tensor&, float>);
static_assert(CanCallFreeDivide<float, const Tensor&>);
static_assert(CanDivide<DoubleTensor&, double>);
static_assert(CanDivide<double, DoubleTensor&>);
static_assert(CanDivide<Tensor&, int>);
static_assert(CanDivide<int, Tensor&>);
static_assert(CanDivide<Tensor&&, float>);
static_assert(CanDivide<float, Tensor&&>);
static_assert(!CanDivide<IntegerTensor&, int>);
static_assert(!CanDivide<int, IntegerTensor&>);
static_assert(!noexcept(std::declval<const Tensor&>() /
                        std::declval<float>()));
static_assert(!noexcept(std::declval<float>() /
                        std::declval<const Tensor&>()));

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

void test_scalar_divide_supports_both_orders_without_changing_lvalue() {
  const Tensor tensor =
      Tensor::from_data({2, 2}, {2.0F, -4.0F, 8.0F, 0.5F});
  const Tensor::shape_type expected_shape{2, 2};
  const Tensor::strides_type expected_strides{2, 1};
  const Tensor::storage_type expected_tensor_first{1.0F, -2.0F, 4.0F, 0.25F};
  const Tensor::storage_type expected_value_first{1.0F, -0.5F, 0.25F, 4.0F};
  const Tensor::storage_type expected_source{2.0F, -4.0F, 8.0F, 0.5F};

  const Tensor tensor_first = tensor / 2.0F;
  const Tensor value_first = 2.0F / tensor;

  expect_state(tensor_first, expected_shape, expected_strides,
               expected_tensor_first, "tensor-first scalar division changed shape",
               "tensor-first scalar division changed strides",
               "tensor-first scalar division computed incorrect elements");
  expect_state(value_first, expected_shape, expected_strides,
               expected_value_first, "value-first scalar division changed shape",
               "value-first scalar division changed strides",
               "value-first scalar division computed incorrect elements");
  expect(std::ranges::equal(tensor.elements(), expected_source),
         "scalar division changed the tensor lvalue");
}

void test_scalar_divide_accepts_convertible_numbers_in_both_orders() {
  const Tensor tensor = Tensor::from_data({2}, {4.0F, -2.0F});
  const Tensor::storage_type expected_tensor_first{2.0F, -1.0F};
  const Tensor::storage_type expected_value_first{0.5F, -1.0F};

  const Tensor tensor_first = tensor / 2;
  const Tensor value_first = 2 / tensor;

  expect(std::ranges::equal(tensor_first.elements(), expected_tensor_first),
         "tensor-first scalar division converted a number incorrectly");
  expect(std::ranges::equal(value_first.elements(), expected_value_first),
         "value-first scalar division converted a number incorrectly");
}

void test_scalar_divide_supports_rank_zero_and_zero_extent_tensors() {
  const Tensor tensor_first_rank_zero = Tensor::scalar(-5.0F) / -2.0F;
  const Tensor value_first_rank_zero = -5.0F / Tensor::scalar(-2.0F);

  expect(tensor_first_rank_zero.rank() == 0 &&
             tensor_first_rank_zero.numel() == 1 &&
             tensor_first_rank_zero.at() == 2.5F,
         "tensor-first scalar division handled rank-zero incorrectly");
  expect(value_first_rank_zero.rank() == 0 &&
             value_first_rank_zero.numel() == 1 &&
             value_first_rank_zero.at() == 2.5F,
         "value-first scalar division handled rank-zero incorrectly");

  const Tensor tensor_first_zero_extent = Tensor({2, 0, 4}) / 3.0F;
  const Tensor value_first_zero_extent = 3.0F / Tensor({2, 0, 4});
  const Tensor::shape_type expected_shape{2, 0, 4};
  const Tensor::strides_type expected_strides{0, 4, 1};

  expect_state(tensor_first_zero_extent, expected_shape, expected_strides, {},
               "tensor-first zero-extent division changed shape",
               "tensor-first zero-extent division changed strides",
               "tensor-first zero-extent division created elements");
  expect_state(value_first_zero_extent, expected_shape, expected_strides, {},
               "value-first zero-extent division changed shape",
               "value-first zero-extent division changed strides",
               "value-first zero-extent division created elements");
}

void test_scalar_divide_preserves_native_zero_divisor_semantics() {
  const Tensor tensor = Tensor::from_data({2}, {0.0F, 2.0F});

  const Tensor result = 1.0F / tensor;

  expect(std::isinf(result.elements()[0]) &&
             !std::signbit(result.elements()[0]),
         "value-first scalar division handled division by zero incorrectly");
  expect(result.elements()[1] == 0.5F,
         "value-first scalar division changed an ordinary quotient");
}

void test_scalar_divide_reuses_rvalue_storage_in_both_orders() {
  Tensor tensor_first_source =
      Tensor::from_data({3}, {2.0F, 4.0F, 8.0F});
  const Tensor::value_type* const tensor_first_storage =
      tensor_first_source.elements().data();

  const Tensor tensor_first = std::move(tensor_first_source) / 2.0F;

  expect(tensor_first.elements().data() == tensor_first_storage,
         "tensor-first scalar division did not reuse rvalue storage");
  expect(std::ranges::equal(tensor_first.elements(),
                            Tensor::storage_type{1.0F, 2.0F, 4.0F}),
         "tensor-first rvalue scalar division computed incorrect elements");
  expect_empty_sentinel(tensor_first_source);

  Tensor value_first_source =
      Tensor::from_data({3}, {2.0F, 4.0F, 8.0F});
  const Tensor::value_type* const value_first_storage =
      value_first_source.elements().data();

  const Tensor value_first = 8.0F / std::move(value_first_source);

  expect(value_first.elements().data() == value_first_storage,
         "value-first scalar division did not reuse rvalue storage");
  expect(std::ranges::equal(value_first.elements(),
                            Tensor::storage_type{4.0F, 2.0F, 1.0F}),
         "value-first rvalue scalar division computed incorrect elements");
  expect_empty_sentinel(value_first_source);
}

void test_scalar_divide_rejects_sentinel_in_both_orders() {
  Tensor tensor_first_source = Tensor::from_data({1}, {1.0F});
  Tensor tensor_first_owner(std::move(tensor_first_source));
  static_cast<void>(tensor_first_owner);

  expect_throws<std::invalid_argument>(
      [&tensor_first_source] {
        static_cast<void>(tensor_first_source / 2.0F);
      },
      "tensor-first scalar division accepted an empty sentinel",
      "tensor-first sentinel division produced the wrong exception type");
  expect_empty_sentinel(tensor_first_source);

  Tensor value_first_source = Tensor::from_data({1}, {3.0F});
  Tensor value_first_owner(std::move(value_first_source));
  static_cast<void>(value_first_owner);

  expect_throws<std::invalid_argument>(
      [&value_first_source] {
        static_cast<void>(2.0F / value_first_source);
      },
      "value-first scalar division accepted an empty sentinel",
      "value-first sentinel division produced the wrong exception type");
  expect_empty_sentinel(value_first_source);
}

}  // namespace

int main() {
  try {
    test_scalar_divide_supports_both_orders_without_changing_lvalue();
    test_scalar_divide_accepts_convertible_numbers_in_both_orders();
    test_scalar_divide_supports_rank_zero_and_zero_extent_tensors();
    test_scalar_divide_preserves_native_zero_divisor_semantics();
    test_scalar_divide_reuses_rvalue_storage_in_both_orders();
    test_scalar_divide_rejects_sentinel_in_both_orders();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor scalar division tests passed\n";
  return EXIT_SUCCESS;
}
