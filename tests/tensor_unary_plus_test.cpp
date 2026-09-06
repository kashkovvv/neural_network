#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
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
concept CanApplyUnaryPlus = requires { +std::declval<TensorType>(); };

template <typename TensorType>
concept CanCallFreeUnaryPlus = requires {
  operator+(std::declval<TensorType>());
};

static_assert(
    std::same_as<decltype(+std::declval<const Tensor&>()), Tensor>);
static_assert(CanCallFreeUnaryPlus<const Tensor&>);
static_assert(CanApplyUnaryPlus<Tensor&>);
static_assert(CanApplyUnaryPlus<const Tensor&>);
static_assert(CanApplyUnaryPlus<Tensor&&>);
static_assert(CanApplyUnaryPlus<const Tensor&&>);
static_assert(CanApplyUnaryPlus<DoubleTensor&>);
static_assert(!CanApplyUnaryPlus<IntegerTensor&>);
static_assert(!noexcept(+std::declval<const Tensor&>()));

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

[[nodiscard]] bool has_same_representation(float left, float right) {
  using Representation = std::array<std::byte, sizeof(float)>;

  return std::bit_cast<Representation>(left) ==
         std::bit_cast<Representation>(right);
}

void test_unary_plus_preserves_lvalue_state_and_element_representations() {
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const Tensor tensor =
      Tensor::from_data({2, 2}, {1.5F, -0.0F, infinity, nan});
  const Tensor::shape_type expected_shape{2, 2};
  const Tensor::strides_type expected_strides{2, 1};

  const Tensor result = +tensor;

  expect(std::ranges::equal(result.shape(), expected_shape),
         "unary plus changed shape");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "unary plus changed strides");
  expect(result.numel() == tensor.numel(), "unary plus changed numel");

  for (Tensor::size_type index = 0; index < tensor.numel(); ++index) {
    expect(has_same_representation(result.elements()[index],
                                   tensor.elements()[index]),
           "unary plus changed an element representation");
  }
}

void test_unary_plus_supports_rank_zero_and_zero_extent_tensors() {
  const Tensor rank_zero_result = +Tensor::scalar(-2.5F);

  expect(rank_zero_result.rank() == 0,
         "unary plus changed rank-zero tensor rank");
  expect(rank_zero_result.numel() == 1,
         "unary plus changed rank-zero tensor numel");
  expect(rank_zero_result.at() == -2.5F,
         "unary plus changed a rank-zero value");

  const Tensor zero_extent_result = +Tensor({2, 0, 4});
  const Tensor::shape_type expected_shape{2, 0, 4};
  const Tensor::strides_type expected_strides{0, 4, 1};

  expect(std::ranges::equal(zero_extent_result.shape(), expected_shape),
         "zero-extent unary plus changed shape");
  expect(std::ranges::equal(zero_extent_result.strides(), expected_strides),
         "zero-extent unary plus changed strides");
  expect(zero_extent_result.elements().empty(),
         "zero-extent unary plus created elements");
}

void test_unary_plus_reuses_rvalue_storage() {
  Tensor source = Tensor::from_data({3}, {1.0F, -2.0F, 3.0F});
  const Tensor::value_type* const original_storage = source.elements().data();
  const Tensor::storage_type expected_elements{1.0F, -2.0F, 3.0F};

  const Tensor result = +std::move(source);

  expect(result.elements().data() == original_storage,
         "unary plus did not reuse rvalue storage");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "rvalue unary plus changed elements");
  expect_empty_sentinel(source);
}

void test_unary_plus_rejects_moved_from_sentinel() {
  Tensor source = Tensor::from_data({1}, {1.0F});
  Tensor owner(std::move(source));
  static_cast<void>(owner);

  expect_throws<std::invalid_argument>(
      [&source] { static_cast<void>(+source); },
      "unary plus accepted an empty sentinel",
      "sentinel unary plus produced the wrong exception type");
  expect_empty_sentinel(source);
}

}  // namespace

int main() {
  try {
    test_unary_plus_preserves_lvalue_state_and_element_representations();
    test_unary_plus_supports_rank_zero_and_zero_extent_tensors();
    test_unary_plus_reuses_rvalue_storage();
    test_unary_plus_rejects_moved_from_sentinel();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor unary plus tests passed\n";
  return EXIT_SUCCESS;
}
