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
concept CanDot = requires(LeftTensor&& left, RightTensor&& right) {
  std::forward<LeftTensor>(left).dot(std::forward<RightTensor>(right));
};

static_assert(std::same_as<decltype(std::declval<const Tensor&>().dot(
                               std::declval<const Tensor&>())),
                           Tensor>);
static_assert(CanDot<Tensor&, Tensor&>);
static_assert(CanDot<const Tensor&, const Tensor&>);
static_assert(CanDot<Tensor&&, Tensor&>);
static_assert(CanDot<const Tensor&&, const Tensor&>);
static_assert(CanDot<Tensor&, Tensor&&>);
static_assert(CanDot<DoubleTensor&, const DoubleTensor&>);
static_assert(!CanDot<IntegerTensor&, const IntegerTensor&>);
static_assert(!CanDot<Tensor&, const DoubleTensor&>);
static_assert(!noexcept(
    std::declval<const Tensor&>().dot(std::declval<const Tensor&>())));

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
  expect(tensor.rank() == 0, "dot result rank is not zero");
  expect(tensor.numel() == 1, "dot result numel is not one");
  expect(tensor.shape().empty(), "dot result shape is not empty");
  expect(tensor.strides().empty(), "dot result strides are not empty");
  expect(tensor.at() == expected_value, value_message);
}

void expect_empty_sentinel(const Tensor& tensor) {
  expect(tensor.rank() == 0, "sentinel rank is not zero");
  expect(tensor.numel() == 0, "sentinel numel is not zero");
  expect(tensor.shape().empty(), "sentinel shape is not empty");
  expect(tensor.strides().empty(), "sentinel strides are not empty");
  expect(tensor.elements().empty(), "sentinel elements are not empty");
}

void test_dot_computes_vector_dot_product_without_changing_operands() {
  const Tensor left = Tensor::from_data({4}, {1.5F, -2.0F, 3.0F, 4.0F});
  const Tensor right = Tensor::from_data({4}, {2.0F, 5.0F, -1.0F, 0.5F});
  const Tensor::shape_type expected_shape{4};
  const Tensor::strides_type expected_strides{1};
  const Tensor::storage_type expected_left{1.5F, -2.0F, 3.0F, 4.0F};
  const Tensor::storage_type expected_right{2.0F, 5.0F, -1.0F, 0.5F};

  const Tensor result = left.dot(right);

  expect_scalar(result, -8.0F, "dot produced an incorrect value");
  expect(std::ranges::equal(left.shape(), expected_shape),
         "dot changed the left shape");
  expect(std::ranges::equal(left.strides(), expected_strides),
         "dot changed the left strides");
  expect(std::ranges::equal(left.elements(), expected_left),
         "dot changed the left operand");
  expect(std::ranges::equal(right.shape(), expected_shape),
         "dot changed the right shape");
  expect(std::ranges::equal(right.strides(), expected_strides),
         "dot changed the right strides");
  expect(std::ranges::equal(right.elements(), expected_right),
         "dot changed the right operand");
}

void test_dot_compensates_rounding_error_in_product_accumulation() {
  const Tensor left =
      Tensor::from_data({3}, {1.0e20F, 1.0F, -1.0e20F});
  const Tensor right = Tensor::from_data({3}, {1.0F, 1.0F, 1.0F});

  const Tensor result = left.dot(right);

  expect_scalar(result, 1.0F,
                "dot did not compensate product accumulation error");
}

void test_dot_supports_singleton_and_empty_vectors() {
  const Tensor singleton_result =
      Tensor::from_data({1}, {-3.0F}).dot(Tensor::from_data({1}, {2.5F}));
  const Tensor empty_result = Tensor({0}).dot(Tensor({0}));

  expect_scalar(singleton_result, -7.5F,
                "dot produced an incorrect singleton result");
  expect_scalar(empty_result, 0.0F,
                "dot of empty vectors is not additive identity");
}

void test_dot_uses_native_floating_point_behavior() {
  const Tensor infinity_result =
      Tensor::from_data({1}, {std::numeric_limits<float>::infinity()})
          .dot(Tensor::from_data({1}, {2.0F}));
  const Tensor overflow_result =
      Tensor::from_data({1}, {std::numeric_limits<float>::max()})
          .dot(Tensor::from_data({1}, {2.0F}));
  const Tensor nan_result =
      Tensor::from_data({1}, {std::numeric_limits<float>::infinity()})
          .dot(Tensor::from_data({1}, {0.0F}));

  expect(std::isinf(infinity_result.at()) && infinity_result.at() > 0.0F,
         "dot did not preserve native floating-point infinity behavior");
  expect(std::isinf(overflow_result.at()) && overflow_result.at() > 0.0F,
         "dot product overflow did not produce positive infinity");
  expect(std::isnan(nan_result.at()),
         "dot did not preserve native floating-point NaN behavior");
}

void test_dot_accepts_rvalues_without_consuming_operands() {
  Tensor left = Tensor::from_data({3}, {1.0F, 2.0F, 3.0F});
  Tensor right = Tensor::from_data({3}, {4.0F, 5.0F, 6.0F});
  const Tensor::storage_type expected_left{1.0F, 2.0F, 3.0F};
  const Tensor::storage_type expected_right{4.0F, 5.0F, 6.0F};

  const Tensor result = std::move(left).dot(std::move(right));

  expect_scalar(result, 32.0F, "rvalue dot produced an incorrect value");
  expect(std::ranges::equal(left.elements(), expected_left),
         "dot consumed its rvalue left operand");
  expect(std::ranges::equal(right.elements(), expected_right),
         "dot consumed its rvalue right operand");
}

void test_dot_rejects_non_vector_operands() {
  const Tensor vector = Tensor::from_data({2}, {1.0F, 2.0F});
  const Tensor scalar = Tensor::scalar(3.0F);
  const Tensor matrix = Tensor::from_data({1, 2}, {3.0F, 4.0F});

  expect_throws<std::invalid_argument>(
      [&scalar, &vector] { static_cast<void>(scalar.dot(vector)); },
      "dot accepted a rank-zero left operand",
      "dot with a rank-zero left operand produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&vector, &scalar] { static_cast<void>(vector.dot(scalar)); },
      "dot accepted a rank-zero right operand",
      "dot with a rank-zero right operand produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&matrix, &vector] { static_cast<void>(matrix.dot(vector)); },
      "dot accepted a rank-two left operand",
      "dot with a rank-two left operand produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&vector, &matrix] { static_cast<void>(vector.dot(matrix)); },
      "dot accepted a rank-two right operand",
      "dot with a rank-two right operand produced the wrong exception type");
}

void test_dot_rejects_different_vector_lengths_without_changes() {
  const Tensor left = Tensor::from_data({2}, {1.0F, 2.0F});
  const Tensor right = Tensor::from_data({3}, {3.0F, 4.0F, 5.0F});
  const Tensor::storage_type expected_left{1.0F, 2.0F};
  const Tensor::storage_type expected_right{3.0F, 4.0F, 5.0F};

  expect_throws<std::invalid_argument>(
      [&left, &right] { static_cast<void>(left.dot(right)); },
      "dot accepted vectors with different lengths",
      "dot with different vector lengths produced the wrong exception type");
  expect(std::ranges::equal(left.elements(), expected_left),
         "failed dot changed the left operand");
  expect(std::ranges::equal(right.elements(), expected_right),
         "failed dot changed the right operand");
}

void test_dot_rejects_moved_from_sentinel_on_each_side() {
  Tensor left_source = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor left_owner(std::move(left_source));
  Tensor right_source = Tensor::from_data({2}, {3.0F, 4.0F});
  Tensor right_owner(std::move(right_source));
  const Tensor valid = Tensor::from_data({2}, {5.0F, 6.0F});
  static_cast<void>(left_owner);
  static_cast<void>(right_owner);

  expect_throws<std::invalid_argument>(
      [&left_source, &valid] { static_cast<void>(left_source.dot(valid)); },
      "dot accepted a moved-from left operand",
      "dot with a moved-from left operand produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&right_source, &valid] { static_cast<void>(valid.dot(right_source)); },
      "dot accepted a moved-from right operand",
      "dot with a moved-from right operand produced the wrong exception type");
  expect_empty_sentinel(left_source);
  expect_empty_sentinel(right_source);
}

}  // namespace

int main() {
  try {
    test_dot_computes_vector_dot_product_without_changing_operands();
    test_dot_compensates_rounding_error_in_product_accumulation();
    test_dot_supports_singleton_and_empty_vectors();
    test_dot_uses_native_floating_point_behavior();
    test_dot_accepts_rvalues_without_consuming_operands();
    test_dot_rejects_non_vector_operands();
    test_dot_rejects_different_vector_lengths_without_changes();
    test_dot_rejects_moved_from_sentinel_on_each_side();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor dot tests passed\n";
  return EXIT_SUCCESS;
}
