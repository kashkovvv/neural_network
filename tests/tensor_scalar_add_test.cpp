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
concept CanAdd = requires {
  std::declval<Left>() + std::declval<Right>();
};

template <typename Left, typename Right>
concept CanCallFreeAdd = requires {
  operator+(std::declval<Left>(), std::declval<Right>());
};

static_assert(std::same_as<decltype(std::declval<const Tensor&>() +
                                    std::declval<float>()),
                           Tensor>);
static_assert(std::same_as<decltype(std::declval<float>() +
                                    std::declval<const Tensor&>()),
                           Tensor>);
static_assert(CanCallFreeAdd<const Tensor&, float>);
static_assert(CanCallFreeAdd<float, const Tensor&>);
static_assert(CanAdd<DoubleTensor&, double>);
static_assert(CanAdd<double, DoubleTensor&>);
static_assert(CanAdd<Tensor&, int>);
static_assert(CanAdd<int, Tensor&>);
static_assert(CanAdd<Tensor&&, float>);
static_assert(CanAdd<float, Tensor&&>);
static_assert(!CanAdd<IntegerTensor&, int>);
static_assert(!CanAdd<int, IntegerTensor&>);
static_assert(!noexcept(std::declval<const Tensor&>() +
                        std::declval<float>()));
static_assert(!noexcept(std::declval<float>() +
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

void test_scalar_add_supports_both_operand_orders_without_changing_lvalue() {
  const Tensor tensor =
      Tensor::from_data({2, 2}, {1.0F, -2.0F, 3.5F, 0.0F});
  const Tensor::shape_type expected_shape{2, 2};
  const Tensor::strides_type expected_strides{2, 1};
  const Tensor::storage_type expected_elements{3.0F, 0.0F, 5.5F, 2.0F};
  const Tensor::storage_type expected_source{1.0F, -2.0F, 3.5F, 0.0F};

  const Tensor tensor_first = tensor + 2.0F;
  const Tensor value_first = 2.0F + tensor;

  expect_state(tensor_first, expected_shape, expected_strides,
               expected_elements, "tensor-first scalar addition changed shape",
               "tensor-first scalar addition changed strides",
               "tensor-first scalar addition computed incorrect elements");
  expect_state(value_first, expected_shape, expected_strides,
               expected_elements, "value-first scalar addition changed shape",
               "value-first scalar addition changed strides",
               "value-first scalar addition computed incorrect elements");
  expect(std::ranges::equal(tensor.elements(), expected_source),
         "scalar addition changed the tensor lvalue");
}

void test_scalar_add_accepts_convertible_numbers_in_both_operand_orders() {
  const Tensor tensor = Tensor::from_data({2}, {1.5F, -2.5F});
  const Tensor::storage_type expected_elements{3.5F, -0.5F};

  const Tensor tensor_first = tensor + 2;
  const Tensor value_first = 2 + tensor;

  expect(std::ranges::equal(tensor_first.elements(), expected_elements),
         "tensor-first scalar addition converted a number incorrectly");
  expect(std::ranges::equal(value_first.elements(), expected_elements),
         "value-first scalar addition converted a number incorrectly");
}

void test_scalar_add_supports_rank_zero_and_zero_extent_tensors() {
  const Tensor rank_zero_result = Tensor::scalar(2.5F) + -0.5F;

  expect(rank_zero_result.rank() == 0,
         "scalar addition changed rank-zero tensor rank");
  expect(rank_zero_result.numel() == 1,
         "scalar addition changed rank-zero tensor numel");
  expect(rank_zero_result.at() == 2.0F,
         "scalar addition computed an incorrect rank-zero value");

  const Tensor zero_extent_result = 3.0F + Tensor({2, 0, 4});
  const Tensor::shape_type expected_shape{2, 0, 4};
  const Tensor::strides_type expected_strides{0, 4, 1};

  expect_state(zero_extent_result, expected_shape, expected_strides, {},
               "zero-extent scalar addition changed shape",
               "zero-extent scalar addition changed strides",
               "zero-extent scalar addition created elements");
}

void test_scalar_add_reuses_rvalue_tensor_storage_in_both_operand_orders() {
  Tensor tensor_first_source =
      Tensor::from_data({3}, {1.0F, 2.0F, 3.0F});
  const Tensor::value_type* const tensor_first_storage =
      tensor_first_source.elements().data();

  const Tensor tensor_first = std::move(tensor_first_source) + 4.0F;

  expect(tensor_first.elements().data() == tensor_first_storage,
         "tensor-first scalar addition did not reuse rvalue storage");
  expect(std::ranges::equal(tensor_first.elements(),
                            Tensor::storage_type{5.0F, 6.0F, 7.0F}),
         "tensor-first rvalue scalar addition computed incorrect elements");
  expect_empty_sentinel(tensor_first_source);

  Tensor value_first_source =
      Tensor::from_data({3}, {-1.0F, 0.0F, 2.0F});
  const Tensor::value_type* const value_first_storage =
      value_first_source.elements().data();

  const Tensor value_first = 3.0F + std::move(value_first_source);

  expect(value_first.elements().data() == value_first_storage,
         "value-first scalar addition did not reuse rvalue storage");
  expect(std::ranges::equal(value_first.elements(),
                            Tensor::storage_type{2.0F, 3.0F, 5.0F}),
         "value-first rvalue scalar addition computed incorrect elements");
  expect_empty_sentinel(value_first_source);
}

void test_scalar_add_rejects_moved_from_sentinel_in_both_operand_orders() {
  Tensor tensor_first_source = Tensor::from_data({1}, {1.0F});
  Tensor tensor_first_owner(std::move(tensor_first_source));
  static_cast<void>(tensor_first_owner);

  expect_throws<std::invalid_argument>(
      [&tensor_first_source] {
        static_cast<void>(tensor_first_source + 2.0F);
      },
      "tensor-first scalar addition accepted an empty sentinel",
      "tensor-first sentinel addition produced the wrong exception type");
  expect_empty_sentinel(tensor_first_source);

  Tensor value_first_source = Tensor::from_data({1}, {3.0F});
  Tensor value_first_owner(std::move(value_first_source));
  static_cast<void>(value_first_owner);

  expect_throws<std::invalid_argument>(
      [&value_first_source] {
        static_cast<void>(2.0F + value_first_source);
      },
      "value-first scalar addition accepted an empty sentinel",
      "value-first sentinel addition produced the wrong exception type");
  expect_empty_sentinel(value_first_source);
}

}  // namespace

int main() {
  try {
    test_scalar_add_supports_both_operand_orders_without_changing_lvalue();
    test_scalar_add_accepts_convertible_numbers_in_both_operand_orders();
    test_scalar_add_supports_rank_zero_and_zero_extent_tensors();
    test_scalar_add_reuses_rvalue_tensor_storage_in_both_operand_orders();
    test_scalar_add_rejects_moved_from_sentinel_in_both_operand_orders();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor scalar addition tests passed\n";
  return EXIT_SUCCESS;
}
