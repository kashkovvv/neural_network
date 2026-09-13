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

template <typename LeftTensor, typename RightTensor>
concept CanOuter = requires(LeftTensor&& left, RightTensor&& right) {
  std::forward<LeftTensor>(left).outer(std::forward<RightTensor>(right));
};

static_assert(std::same_as<decltype(std::declval<const Tensor&>().outer(
                               std::declval<const Tensor&>())),
                           Tensor>);
static_assert(CanOuter<Tensor&, Tensor&>);
static_assert(CanOuter<const Tensor&, const Tensor&>);
static_assert(CanOuter<Tensor&&, Tensor&>);
static_assert(CanOuter<const Tensor&&, const Tensor&>);
static_assert(CanOuter<Tensor&, Tensor&&>);
static_assert(CanOuter<DoubleTensor&, const DoubleTensor&>);
static_assert(!CanOuter<IntegerTensor&, const IntegerTensor&>);
static_assert(!CanOuter<Tensor&, const DoubleTensor&>);
static_assert(!noexcept(
    std::declval<const Tensor&>().outer(std::declval<const Tensor&>())));

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

void expect_tensor(const Tensor& tensor,
                   const Tensor::shape_type& expected_shape,
                   const Tensor::strides_type& expected_strides,
                   const Tensor::storage_type& expected_elements,
                   const char* message) {
  expect(std::ranges::equal(tensor.shape(), expected_shape), message);
  expect(std::ranges::equal(tensor.strides(), expected_strides), message);
  expect(std::ranges::equal(tensor.elements(), expected_elements), message);
}

void expect_empty_sentinel(const Tensor& tensor) {
  expect(tensor.rank() == 0, "sentinel rank is not zero");
  expect(tensor.numel() == 0, "sentinel numel is not zero");
  expect(tensor.shape().empty(), "sentinel shape is not empty");
  expect(tensor.strides().empty(), "sentinel strides are not empty");
  expect(tensor.elements().empty(), "sentinel elements are not empty");
}

void test_outer_computes_rectangular_product_without_changing_operands() {
  const Tensor left = Tensor::from_data({2}, {1.5F, -2.0F});
  const Tensor right = Tensor::from_data({3}, {2.0F, 0.0F, -1.0F});
  const Tensor::shape_type left_shape{2};
  const Tensor::shape_type right_shape{3};
  const Tensor::strides_type vector_strides{1};
  const Tensor::storage_type left_elements{1.5F, -2.0F};
  const Tensor::storage_type right_elements{2.0F, 0.0F, -1.0F};

  const Tensor result = left.outer(right);

  expect_tensor(result, {2, 3}, {3, 1}, {3.0F, 0.0F, -1.5F, -4.0F, -0.0F, 2.0F},
                "outer produced an incorrect result");
  expect_tensor(left, left_shape, vector_strides, left_elements,
                "outer changed the left operand");
  expect_tensor(right, right_shape, vector_strides, right_elements,
                "outer changed the right operand");
}

void test_outer_supports_singleton_vectors() {
  const Tensor one_row = Tensor::from_data({1}, {-2.0F})
                             .outer(Tensor::from_data({3}, {1.0F, 2.0F, 3.0F}));
  const Tensor one_column = Tensor::from_data({3}, {1.0F, 2.0F, 3.0F})
                                .outer(Tensor::from_data({1}, {-2.0F}));

  expect_tensor(one_row, {1, 3}, {3, 1}, {-2.0F, -4.0F, -6.0F},
                "outer produced an incorrect one-row result");
  expect_tensor(one_column, {3, 1}, {1, 1}, {-2.0F, -4.0F, -6.0F},
                "outer produced an incorrect one-column result");
}

void test_outer_supports_zero_extent_vectors() {
  const Tensor zero_rows =
      Tensor({0}).outer(Tensor::from_data({3}, {1.0F, 2.0F, 3.0F}));
  const Tensor zero_columns =
      Tensor::from_data({2}, {1.0F, 2.0F}).outer(Tensor({0}));
  const Tensor fully_empty = Tensor({0}).outer(Tensor({0}));

  expect_tensor(zero_rows, {0, 3}, {3, 1}, {},
                "outer produced an incorrect zero-row result");
  expect_tensor(zero_columns, {2, 0}, {0, 1}, {},
                "outer produced an incorrect zero-column result");
  expect_tensor(fully_empty, {0, 0}, {0, 1}, {},
                "outer produced an incorrect fully empty result");
}

void test_outer_uses_native_floating_point_behavior() {
  const Tensor result =
      Tensor::from_data({1}, {std::numeric_limits<float>::infinity()})
          .outer(Tensor::from_data({2}, {2.0F, 0.0F}));

  expect(std::isinf(result.at(0, 0)) && result.at(0, 0) > 0.0F,
         "outer did not preserve native floating-point infinity behavior");
  expect(std::isnan(result.at(0, 1)),
         "outer did not preserve native floating-point NaN behavior");
}

void test_outer_accepts_rvalues_without_consuming_operands() {
  Tensor left = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor right = Tensor::from_data({2}, {3.0F, 4.0F});
  const Tensor::storage_type expected_left{1.0F, 2.0F};
  const Tensor::storage_type expected_right{3.0F, 4.0F};

  const Tensor result = std::move(left).outer(std::move(right));

  expect_tensor(result, {2, 2}, {2, 1}, {3.0F, 4.0F, 6.0F, 8.0F},
                "rvalue outer produced an incorrect result");
  expect(std::ranges::equal(left.elements(), expected_left),
         "outer consumed its rvalue left operand");
  expect(std::ranges::equal(right.elements(), expected_right),
         "outer consumed its rvalue right operand");
}

void test_outer_rejects_non_vector_operands() {
  const Tensor vector = Tensor::from_data({2}, {1.0F, 2.0F});
  const Tensor scalar = Tensor::scalar(3.0F);
  const Tensor matrix = Tensor::from_data({1, 2}, {3.0F, 4.0F});

  expect_throws<std::invalid_argument>(
      [&scalar, &vector] { static_cast<void>(scalar.outer(vector)); },
      "outer accepted a rank-zero left operand",
      "outer with a rank-zero left operand produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&vector, &scalar] { static_cast<void>(vector.outer(scalar)); },
      "outer accepted a rank-zero right operand",
      "outer with a rank-zero right operand produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&matrix, &vector] { static_cast<void>(matrix.outer(vector)); },
      "outer accepted a rank-two left operand",
      "outer with a rank-two left operand produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&vector, &matrix] { static_cast<void>(vector.outer(matrix)); },
      "outer accepted a rank-two right operand",
      "outer with a rank-two right operand produced the wrong exception type");
}

void test_outer_rejects_moved_from_sentinel_on_each_side() {
  Tensor left_source = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor left_owner(std::move(left_source));
  Tensor right_source = Tensor::from_data({2}, {3.0F, 4.0F});
  Tensor right_owner(std::move(right_source));
  const Tensor valid = Tensor::from_data({2}, {5.0F, 6.0F});
  static_cast<void>(left_owner);
  static_cast<void>(right_owner);

  expect_throws<std::invalid_argument>(
      [&left_source, &valid] { static_cast<void>(left_source.outer(valid)); },
      "outer accepted a moved-from left operand",
      "outer with a moved-from left operand produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&right_source, &valid] { static_cast<void>(valid.outer(right_source)); },
      "outer accepted a moved-from right operand",
      "outer with a moved-from right operand produced the wrong exception "
      "type");
  expect_empty_sentinel(left_source);
  expect_empty_sentinel(right_source);
}

}  // namespace

int main() {
  try {
    test_outer_computes_rectangular_product_without_changing_operands();
    test_outer_supports_singleton_vectors();
    test_outer_supports_zero_extent_vectors();
    test_outer_uses_native_floating_point_behavior();
    test_outer_accepts_rvalues_without_consuming_operands();
    test_outer_rejects_non_vector_operands();
    test_outer_rejects_moved_from_sentinel_on_each_side();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor outer tests passed\n";
  return EXIT_SUCCESS;
}
