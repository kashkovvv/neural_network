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
concept CanSubtract = requires {
  std::declval<Left>() - std::declval<Right>();
};

template <typename Left, typename Right>
concept CanCallFreeSubtract = requires {
  operator-(std::declval<Left>(), std::declval<Right>());
};

static_assert(std::same_as<decltype(std::declval<const Tensor&>() -
                                    std::declval<float>()),
                           Tensor>);
static_assert(CanCallFreeSubtract<const Tensor&, float>);
static_assert(CanSubtract<DoubleTensor&, double>);
static_assert(CanSubtract<Tensor&, int>);
static_assert(CanSubtract<Tensor&&, float>);
static_assert(!CanSubtract<IntegerTensor&, int>);
static_assert(!noexcept(std::declval<const Tensor&>() -
                        std::declval<float>()));

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

void test_scalar_subtract_computes_result_without_changing_lvalue() {
  const Tensor tensor =
      Tensor::from_data({2, 2}, {1.0F, -2.0F, 3.5F, 0.0F});
  const Tensor::shape_type expected_shape{2, 2};
  const Tensor::strides_type expected_strides{2, 1};
  const Tensor::storage_type expected_elements{-1.0F, -4.0F, 1.5F, -2.0F};
  const Tensor::storage_type expected_source{1.0F, -2.0F, 3.5F, 0.0F};

  const Tensor result = tensor - 2.0F;

  expect_state(result, expected_shape, expected_strides, expected_elements,
               "scalar subtraction changed shape",
               "scalar subtraction changed strides",
               "scalar subtraction computed incorrect elements");
  expect(std::ranges::equal(tensor.elements(), expected_source),
         "scalar subtraction changed the tensor lvalue");
}

void test_scalar_subtract_accepts_convertible_number() {
  const Tensor tensor = Tensor::from_data({2}, {1.5F, -2.5F});
  const Tensor::storage_type expected_elements{-0.5F, -4.5F};

  const Tensor result = tensor - 2;

  expect(std::ranges::equal(result.elements(), expected_elements),
         "scalar subtraction converted a number incorrectly");
}

void test_scalar_subtract_supports_rank_zero_and_zero_extent_tensors() {
  const Tensor rank_zero_result = Tensor::scalar(2.5F) - -0.5F;

  expect(rank_zero_result.rank() == 0,
         "scalar subtraction changed rank-zero tensor rank");
  expect(rank_zero_result.numel() == 1,
         "scalar subtraction changed rank-zero tensor numel");
  expect(rank_zero_result.at() == 3.0F,
         "scalar subtraction computed an incorrect rank-zero value");

  const Tensor zero_extent_result = Tensor({2, 0, 4}) - 3.0F;
  const Tensor::shape_type expected_shape{2, 0, 4};
  const Tensor::strides_type expected_strides{0, 4, 1};

  expect_state(zero_extent_result, expected_shape, expected_strides, {},
               "zero-extent scalar subtraction changed shape",
               "zero-extent scalar subtraction changed strides",
               "zero-extent scalar subtraction created elements");
}

void test_scalar_subtract_reuses_rvalue_tensor_storage() {
  Tensor source = Tensor::from_data({3}, {1.0F, 2.0F, 3.0F});
  const Tensor::value_type* const original_storage = source.elements().data();
  const Tensor::storage_type expected_elements{-3.0F, -2.0F, -1.0F};

  const Tensor result = std::move(source) - 4.0F;

  expect(result.elements().data() == original_storage,
         "scalar subtraction did not reuse rvalue storage");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "rvalue scalar subtraction computed incorrect elements");
  expect_empty_sentinel(source);
}

void test_scalar_subtract_rejects_moved_from_sentinel() {
  Tensor source = Tensor::from_data({1}, {1.0F});
  Tensor owner(std::move(source));
  static_cast<void>(owner);

  expect_throws<std::invalid_argument>(
      [&source] { static_cast<void>(source - 2.0F); },
      "scalar subtraction accepted an empty sentinel",
      "sentinel scalar subtraction produced the wrong exception type");
  expect_empty_sentinel(source);
}

}  // namespace

int main() {
  try {
    test_scalar_subtract_computes_result_without_changing_lvalue();
    test_scalar_subtract_accepts_convertible_number();
    test_scalar_subtract_supports_rank_zero_and_zero_extent_tensors();
    test_scalar_subtract_reuses_rvalue_tensor_storage();
    test_scalar_subtract_rejects_moved_from_sentinel();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor scalar subtraction tests passed\n";
  return EXIT_SUCCESS;
}
