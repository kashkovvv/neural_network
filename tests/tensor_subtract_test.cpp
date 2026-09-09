#include <algorithm>
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

template <typename Left, typename Right>
concept CanSubtract = requires {
  std::declval<Left>() - std::declval<Right>();
};

template <typename Left, typename Right>
concept CanCallFreeSubtract = requires {
  operator-(std::declval<Left>(), std::declval<Right>());
};

static_assert(std::same_as<decltype(std::declval<const Tensor&>() -
                                    std::declval<const Tensor&>()),
                           Tensor>);
static_assert(CanCallFreeSubtract<const Tensor&, const Tensor&>);
static_assert(CanSubtract<Tensor&, const Tensor&>);
static_assert(CanSubtract<const Tensor&, const Tensor&>);
static_assert(CanSubtract<Tensor&&, const Tensor&>);
static_assert(CanSubtract<const Tensor&, Tensor&&>);
static_assert(CanSubtract<DoubleTensor&, const DoubleTensor&>);
static_assert(!CanSubtract<IntegerTensor&, const IntegerTensor&>);
static_assert(!CanSubtract<Tensor&, const DoubleTensor&>);
static_assert(
    !noexcept(std::declval<const Tensor&>() - std::declval<const Tensor&>()));

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

void test_subtract_computes_elementwise_without_changing_lvalues() {
  const Tensor left =
      Tensor::from_data({2, 3}, {9.0F, 7.0F, 5.0F, 3.0F, 1.0F, -1.0F});
  const Tensor right =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_result{8.0F, 5.0F, 2.0F,
                                              -1.0F, -4.0F, -7.0F};
  const Tensor::storage_type expected_left{9.0F, 7.0F, 5.0F,
                                            3.0F, 1.0F, -1.0F};
  const Tensor::storage_type expected_right{1.0F, 2.0F, 3.0F,
                                             4.0F, 5.0F, 6.0F};

  const Tensor result = left - right;

  expect_state(result, expected_shape, expected_strides, expected_result,
               "subtraction produced an incorrect shape",
               "subtraction produced incorrect strides",
               "subtraction produced incorrect elements");
  expect(std::ranges::equal(left.elements(), expected_left),
         "subtraction changed the left lvalue elements");
  expect(std::ranges::equal(right.elements(), expected_right),
         "subtraction changed the right lvalue elements");
}

void test_subtract_broadcasts_either_lower_rank_operand_in_operand_order() {
  const Tensor vector = Tensor::from_data({3}, {10.0F, 20.0F, 30.0F});
  const Tensor matrix =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_matrix_minus_vector{
      -9.0F, -18.0F, -27.0F, -6.0F, -15.0F, -24.0F};
  const Tensor::storage_type expected_vector_minus_matrix{
      9.0F, 18.0F, 27.0F, 6.0F, 15.0F, 24.0F};

  const Tensor matrix_minus_vector = matrix - vector;
  const Tensor vector_minus_matrix = vector - matrix;

  expect_state(matrix_minus_vector, expected_shape, expected_strides,
               expected_matrix_minus_vector,
               "right broadcast changed the subtraction result shape",
               "right broadcast produced incorrect subtraction strides",
               "right broadcast changed subtraction operand order");
  expect_state(vector_minus_matrix, expected_shape, expected_strides,
               expected_vector_minus_matrix,
               "left broadcast changed the subtraction result shape",
               "left broadcast produced incorrect subtraction strides",
               "left broadcast changed subtraction operand order");
}

void test_subtract_broadcasts_singleton_axes_in_both_operands() {
  const Tensor left =
      Tensor::from_data({2, 1, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor right = Tensor::from_data({1, 2, 1}, {10.0F, 20.0F});
  const Tensor::shape_type expected_shape{2, 2, 3};
  const Tensor::strides_type expected_strides{6, 3, 1};
  const Tensor::storage_type expected_elements{
      -9.0F, -8.0F, -7.0F, -19.0F, -18.0F, -17.0F,
      -6.0F, -5.0F, -4.0F, -16.0F, -15.0F, -14.0F};

  const Tensor result = left - right;

  expect_state(result, expected_shape, expected_strides, expected_elements,
               "singleton broadcast produced an incorrect subtraction shape",
               "singleton broadcast produced incorrect subtraction strides",
               "singleton broadcast produced incorrect subtraction elements");
}

void test_subtract_broadcasts_rank_zero_tensor_from_either_side() {
  const Tensor scalar = Tensor::scalar(10.0F);
  const Tensor matrix =
      Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  const Tensor::storage_type expected_scalar_minus_matrix{9.0F, 8.0F, 7.0F,
                                                           6.0F};
  const Tensor::storage_type expected_matrix_minus_scalar{-9.0F, -8.0F,
                                                           -7.0F, -6.0F};

  const Tensor scalar_minus_matrix = scalar - matrix;
  const Tensor matrix_minus_scalar = matrix - scalar;

  expect(std::ranges::equal(scalar_minus_matrix.elements(),
                            expected_scalar_minus_matrix),
         "left rank-zero broadcast changed subtraction operand order");
  expect(std::ranges::equal(matrix_minus_scalar.elements(),
                            expected_matrix_minus_scalar),
         "right rank-zero broadcast changed subtraction operand order");
}

void test_subtract_supports_scalars_and_zero_extent_tensors() {
  const Tensor scalar_result = Tensor::scalar(2.5F) - Tensor::scalar(-0.5F);

  expect(scalar_result.rank() == 0, "scalar subtraction changed rank");
  expect(scalar_result.numel() == 1, "scalar subtraction changed numel");
  expect(scalar_result.at() == 3.0F,
         "scalar subtraction produced an incorrect value");

  const Tensor zero_extent_result = Tensor({2, 0, 4}) - Tensor({2, 0, 4});
  const Tensor::shape_type expected_shape{2, 0, 4};
  const Tensor::strides_type expected_strides{0, 4, 1};

  expect_state(zero_extent_result, expected_shape, expected_strides, {},
               "zero-extent subtraction changed shape",
               "zero-extent subtraction produced incorrect strides",
               "zero-extent subtraction created elements");
}

void test_subtract_broadcasts_zero_extent_axes() {
  const Tensor zero_extent({2, 0, 3});
  const Tensor row = Tensor::from_data({1, 3}, {1.0F, 2.0F, 3.0F});
  const Tensor singleton_axis =
      Tensor::from_data({2, 1, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor::shape_type expected_shape{2, 0, 3};
  const Tensor::strides_type expected_strides{0, 3, 1};

  const Tensor unchanged_shape_result = zero_extent - row;
  const Tensor expanded_left_result = singleton_axis - zero_extent;

  expect_state(unchanged_shape_result, expected_shape, expected_strides, {},
               "zero-extent broadcast produced an incorrect subtraction shape",
               "zero-extent broadcast produced incorrect subtraction strides",
               "zero-extent broadcast created subtraction elements");
  expect_state(expanded_left_result, expected_shape, expected_strides, {},
               "zero-extent broadcast did not expand the subtraction shape",
               "expanded zero-extent subtraction has incorrect strides",
               "expanded zero-extent subtraction contains elements");
}

void test_subtract_supports_aliased_lvalues_without_changing_the_source() {
  const Tensor tensor =
      Tensor::from_data({2, 2}, {1.0F, -2.0F, 3.5F, 0.0F});
  const Tensor::storage_type expected_source{1.0F, -2.0F, 3.5F, 0.0F};
  const Tensor::storage_type expected_result{0.0F, 0.0F, 0.0F, 0.0F};

  const Tensor result = tensor - tensor;

  expect(std::ranges::equal(result.elements(), expected_result),
         "subtraction of aliased lvalues computed incorrect elements");
  expect(std::ranges::equal(tensor.elements(), expected_source),
         "subtraction of aliased lvalues changed the source");
}

void test_subtract_reuses_rvalue_left_storage_and_consumes_it() {
  Tensor left = Tensor::from_data({3}, {5.0F, 7.0F, 9.0F});
  const Tensor right = Tensor::from_data({3}, {4.0F, 5.0F, 6.0F});
  const Tensor::value_type* const original_storage = left.elements().data();
  const Tensor::storage_type expected_elements{1.0F, 2.0F, 3.0F};

  const Tensor result = std::move(left) - right;

  expect(result.elements().data() == original_storage,
         "subtraction did not reuse the rvalue left storage");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "subtraction with an rvalue left computed incorrect elements");
  expect_empty_sentinel(left);
}

void test_subtract_reuses_rvalue_left_storage_when_right_is_broadcast() {
  Tensor left = Tensor::from_data(
      {2, 3}, {11.0F, 22.0F, 33.0F, 14.0F, 25.0F, 36.0F});
  const Tensor right = Tensor::from_data({3}, {10.0F, 20.0F, 30.0F});
  const Tensor::value_type* const original_storage = left.elements().data();
  const Tensor::storage_type expected_elements{1.0F, 2.0F, 3.0F,
                                                4.0F, 5.0F, 6.0F};

  const Tensor result = std::move(left) - right;

  expect(result.elements().data() == original_storage,
         "broadcast subtraction did not reuse rvalue left storage");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "broadcast subtraction with an rvalue left computed incorrect "
         "elements");
  expect_empty_sentinel(left);
}

void test_subtract_consumes_rvalue_left_when_broadcast_expands_it() {
  Tensor left = Tensor::from_data({3}, {10.0F, 20.0F, 30.0F});
  const Tensor right =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor::storage_type expected_elements{9.0F, 18.0F, 27.0F,
                                                6.0F, 15.0F, 24.0F};

  const Tensor result = std::move(left) - right;

  expect(std::ranges::equal(result.elements(), expected_elements),
         "expanded rvalue subtraction changed operand order");
  expect_empty_sentinel(left);
}

void test_subtract_does_not_consume_rvalue_right() {
  const Tensor left = Tensor::from_data({3}, {10.0F, 20.0F, 30.0F});
  Tensor right =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor::storage_type expected_result{9.0F, 18.0F, 27.0F,
                                              6.0F, 15.0F, 24.0F};
  const Tensor::storage_type expected_right{1.0F, 2.0F, 3.0F,
                                             4.0F, 5.0F, 6.0F};

  const Tensor result = left - std::move(right);

  expect(std::ranges::equal(result.elements(), expected_result),
         "subtraction with an rvalue right computed incorrect elements");
  expect(std::ranges::equal(right.elements(), expected_right),
         "subtraction consumed the rvalue right operand");
}

void test_subtract_rejects_incompatible_shapes_without_changing_lvalues() {
  const Tensor left =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor right =
      Tensor::from_data({3, 2}, {6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F});
  const Tensor::storage_type expected_left{1.0F, 2.0F, 3.0F,
                                            4.0F, 5.0F, 6.0F};
  const Tensor::storage_type expected_right{6.0F, 5.0F, 4.0F,
                                             3.0F, 2.0F, 1.0F};

  expect_throws<std::invalid_argument>(
      [&left, &right] { static_cast<void>(left - right); },
      "subtraction with different shapes did not throw std::invalid_argument",
      "subtraction with different shapes produced the wrong exception type");

  expect(std::ranges::equal(left.elements(), expected_left),
         "failed subtraction changed the left lvalue elements");
  expect(std::ranges::equal(right.elements(), expected_right),
         "failed subtraction changed the right lvalue elements");

  const Tensor zero_extent({2, 0, 3});
  const Tensor incompatible_zero_extent({2, 2, 3});

  expect_throws<std::invalid_argument>(
      [&zero_extent, &incompatible_zero_extent] {
        static_cast<void>(zero_extent - incompatible_zero_extent);
      },
      "subtraction accepted incompatible zero and non-singleton extents",
      "subtraction with incompatible zero extents produced the wrong exception "
      "type");
}

void test_subtract_rejects_overflowing_broadcast_result_shape() {
  const Tensor::size_type max_extent =
      std::numeric_limits<Tensor::size_type>::max();
  const Tensor left({0, max_extent, 1});
  const Tensor right({0, 1, 2});

  expect_throws<std::overflow_error>(
      [&left, &right] { static_cast<void>(left - right); },
      "subtraction accepted a broadcast result with overflowing strides",
      "subtraction with an overflowing result shape produced the wrong "
      "exception type");
}

void test_failed_subtract_consumes_rvalue_left() {
  Tensor left = Tensor::from_data({2}, {1.0F, 2.0F});
  const Tensor right = Tensor::from_data({3}, {3.0F, 4.0F, 5.0F});

  expect_throws<std::invalid_argument>(
      [&left, &right] { static_cast<void>(std::move(left) - right); },
      "failed subtraction with an rvalue left did not throw",
      "failed subtraction with an rvalue left produced the wrong exception "
      "type");

  expect_empty_sentinel(left);
}

void test_subtract_rejects_moved_from_sentinels() {
  Tensor left_source = Tensor::from_data({1}, {1.0F});
  Tensor left_owner(std::move(left_source));
  static_cast<void>(left_owner);
  const Tensor ordinary = Tensor::from_data({1}, {2.0F});

  expect_throws<std::invalid_argument>(
      [&left_source, &ordinary] {
        static_cast<void>(left_source - ordinary);
      },
      "subtraction accepted a sentinel left operand",
      "sentinel left subtraction produced the wrong exception type");
  expect_empty_sentinel(left_source);

  Tensor right_source = Tensor::from_data({1}, {3.0F});
  Tensor right_owner(std::move(right_source));
  static_cast<void>(right_owner);

  expect_throws<std::invalid_argument>(
      [&ordinary, &right_source] {
        static_cast<void>(ordinary - right_source);
      },
      "subtraction accepted a sentinel right operand",
      "sentinel right subtraction produced the wrong exception type");
  expect_empty_sentinel(right_source);
}

}  // namespace

int main() {
  try {
    test_subtract_computes_elementwise_without_changing_lvalues();
    test_subtract_broadcasts_either_lower_rank_operand_in_operand_order();
    test_subtract_broadcasts_singleton_axes_in_both_operands();
    test_subtract_broadcasts_rank_zero_tensor_from_either_side();
    test_subtract_supports_scalars_and_zero_extent_tensors();
    test_subtract_broadcasts_zero_extent_axes();
    test_subtract_supports_aliased_lvalues_without_changing_the_source();
    test_subtract_reuses_rvalue_left_storage_and_consumes_it();
    test_subtract_reuses_rvalue_left_storage_when_right_is_broadcast();
    test_subtract_consumes_rvalue_left_when_broadcast_expands_it();
    test_subtract_does_not_consume_rvalue_right();
    test_subtract_rejects_incompatible_shapes_without_changing_lvalues();
    test_subtract_rejects_overflowing_broadcast_result_shape();
    test_failed_subtract_consumes_rvalue_left();
    test_subtract_rejects_moved_from_sentinels();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor subtraction tests passed\n";
  return EXIT_SUCCESS;
}
