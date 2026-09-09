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
concept CanDivide = requires {
  std::declval<Left>() / std::declval<Right>();
};

template <typename Left, typename Right>
concept CanCallFreeDivide = requires {
  operator/(std::declval<Left>(), std::declval<Right>());
};

static_assert(std::same_as<decltype(std::declval<const Tensor&>() /
                                    std::declval<const Tensor&>()),
                           Tensor>);
static_assert(CanCallFreeDivide<const Tensor&, const Tensor&>);
static_assert(CanDivide<Tensor&, const Tensor&>);
static_assert(CanDivide<const Tensor&, const Tensor&>);
static_assert(CanDivide<Tensor&&, const Tensor&>);
static_assert(CanDivide<const Tensor&, Tensor&&>);
static_assert(CanDivide<DoubleTensor&, const DoubleTensor&>);
static_assert(!CanDivide<IntegerTensor&, const IntegerTensor&>);
static_assert(!CanDivide<Tensor&, const DoubleTensor&>);
static_assert(
    !noexcept(std::declval<const Tensor&>() / std::declval<const Tensor&>()));

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

void test_divide_computes_elementwise_without_changing_lvalues() {
  const Tensor left =
      Tensor::from_data({2, 3}, {12.0F, -6.0F, 9.0F, 1.0F, 0.5F, -8.0F});
  const Tensor right =
      Tensor::from_data({2, 3}, {3.0F, 2.0F, -3.0F, 4.0F, 0.25F, -2.0F});
  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_result{4.0F, -3.0F, -3.0F,
                                              0.25F, 2.0F, 4.0F};
  const Tensor::storage_type expected_left{12.0F, -6.0F, 9.0F,
                                            1.0F, 0.5F, -8.0F};
  const Tensor::storage_type expected_right{3.0F, 2.0F, -3.0F,
                                             4.0F, 0.25F, -2.0F};

  const Tensor result = left / right;

  expect_state(result, expected_shape, expected_strides, expected_result,
               "division produced an incorrect shape",
               "division produced incorrect strides",
               "division produced incorrect elements");
  expect(std::ranges::equal(left.elements(), expected_left),
         "division changed the left lvalue elements");
  expect(std::ranges::equal(right.elements(), expected_right),
         "division changed the right lvalue elements");
}

void test_divide_broadcasts_either_lower_rank_operand_in_operand_order() {
  const Tensor vector = Tensor::from_data({3}, {8.0F, 16.0F, 32.0F});
  const Tensor matrix =
      Tensor::from_data({2, 3}, {2.0F, 4.0F, 8.0F, 4.0F, 8.0F, 16.0F});
  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_matrix_divided_by_vector{
      0.25F, 0.25F, 0.25F, 0.5F, 0.5F, 0.5F};
  const Tensor::storage_type expected_vector_divided_by_matrix{
      4.0F, 4.0F, 4.0F, 2.0F, 2.0F, 2.0F};

  const Tensor matrix_divided_by_vector = matrix / vector;
  const Tensor vector_divided_by_matrix = vector / matrix;

  expect_state(matrix_divided_by_vector, expected_shape, expected_strides,
               expected_matrix_divided_by_vector,
               "right broadcast changed the division result shape",
               "right broadcast produced incorrect division strides",
               "right broadcast changed division operand order");
  expect_state(vector_divided_by_matrix, expected_shape, expected_strides,
               expected_vector_divided_by_matrix,
               "left broadcast changed the division result shape",
               "left broadcast produced incorrect division strides",
               "left broadcast changed division operand order");
}

void test_divide_broadcasts_singleton_axes_in_both_operands() {
  const Tensor left = Tensor::from_data(
      {2, 1, 3}, {8.0F, 16.0F, 32.0F, 4.0F, 8.0F, 16.0F});
  const Tensor right = Tensor::from_data({1, 2, 1}, {2.0F, 4.0F});
  const Tensor::shape_type expected_shape{2, 2, 3};
  const Tensor::strides_type expected_strides{6, 3, 1};
  const Tensor::storage_type expected_elements{
      4.0F, 8.0F, 16.0F, 2.0F, 4.0F, 8.0F,
      2.0F, 4.0F, 8.0F, 1.0F, 2.0F, 4.0F};

  const Tensor result = left / right;

  expect_state(result, expected_shape, expected_strides, expected_elements,
               "singleton broadcast produced an incorrect division shape",
               "singleton broadcast produced incorrect division strides",
               "singleton broadcast produced incorrect division elements");
}

void test_divide_broadcasts_rank_zero_tensor_from_either_side() {
  const Tensor scalar = Tensor::scalar(8.0F);
  const Tensor matrix =
      Tensor::from_data({2, 2}, {2.0F, 4.0F, 8.0F, 16.0F});
  const Tensor::storage_type expected_scalar_divided_by_matrix{4.0F, 2.0F,
                                                                1.0F, 0.5F};
  const Tensor::storage_type expected_matrix_divided_by_scalar{0.25F, 0.5F,
                                                                1.0F, 2.0F};

  const Tensor scalar_divided_by_matrix = scalar / matrix;
  const Tensor matrix_divided_by_scalar = matrix / scalar;

  expect(std::ranges::equal(scalar_divided_by_matrix.elements(),
                            expected_scalar_divided_by_matrix),
         "left rank-zero broadcast changed division operand order");
  expect(std::ranges::equal(matrix_divided_by_scalar.elements(),
                            expected_matrix_divided_by_scalar),
         "right rank-zero broadcast changed division operand order");
}

void test_divide_supports_scalars_and_zero_extent_tensors() {
  const Tensor scalar_result = Tensor::scalar(7.5F) / Tensor::scalar(2.5F);

  expect(scalar_result.rank() == 0, "scalar division changed rank");
  expect(scalar_result.numel() == 1, "scalar division changed numel");
  expect(scalar_result.at() == 3.0F,
         "scalar division produced an incorrect value");

  const Tensor zero_extent_result = Tensor({2, 0, 4}) / Tensor({2, 0, 4});
  const Tensor::shape_type expected_shape{2, 0, 4};
  const Tensor::strides_type expected_strides{0, 4, 1};

  expect_state(zero_extent_result, expected_shape, expected_strides, {},
               "zero-extent division changed shape",
               "zero-extent division produced incorrect strides",
               "zero-extent division created elements");
}

void test_divide_broadcasts_zero_extent_axes() {
  const Tensor zero_extent({2, 0, 3});
  const Tensor row = Tensor::from_data({1, 3}, {1.0F, 2.0F, 4.0F});
  const Tensor singleton_axis =
      Tensor::from_data({2, 1, 3}, {1.0F, 2.0F, 4.0F, 8.0F, 16.0F, 32.0F});
  const Tensor::shape_type expected_shape{2, 0, 3};
  const Tensor::strides_type expected_strides{0, 3, 1};

  const Tensor unchanged_shape_result = zero_extent / row;
  const Tensor expanded_left_result = singleton_axis / zero_extent;

  expect_state(unchanged_shape_result, expected_shape, expected_strides, {},
               "zero-extent broadcast produced an incorrect division shape",
               "zero-extent broadcast produced incorrect division strides",
               "zero-extent broadcast created division elements");
  expect_state(expanded_left_result, expected_shape, expected_strides, {},
               "zero-extent broadcast did not expand the division shape",
               "expanded zero-extent division has incorrect strides",
               "expanded zero-extent division contains elements");
}

void test_divide_supports_aliased_lvalues_without_changing_the_source() {
  const Tensor tensor =
      Tensor::from_data({2, 2}, {1.0F, -2.0F, 3.5F, -0.5F});
  const Tensor::storage_type expected_source{1.0F, -2.0F, 3.5F, -0.5F};
  const Tensor::storage_type expected_result{1.0F, 1.0F, 1.0F, 1.0F};

  const Tensor result = tensor / tensor;

  expect(std::ranges::equal(result.elements(), expected_result),
         "division of aliased lvalues computed incorrect elements");
  expect(std::ranges::equal(tensor.elements(), expected_source),
         "division of aliased lvalues changed the source");
}

void test_divide_reuses_rvalue_left_storage_and_consumes_it() {
  Tensor left = Tensor::from_data({3}, {8.0F, -9.0F, 7.0F});
  const Tensor right = Tensor::from_data({3}, {4.0F, -3.0F, 2.0F});
  const Tensor::value_type* const original_storage = left.elements().data();
  const Tensor::storage_type expected_elements{2.0F, 3.0F, 3.5F};

  const Tensor result = std::move(left) / right;

  expect(result.elements().data() == original_storage,
         "division did not reuse the rvalue left storage");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "division with an rvalue left computed incorrect elements");
  expect_empty_sentinel(left);
}

void test_divide_reuses_rvalue_left_storage_when_right_is_broadcast() {
  Tensor left =
      Tensor::from_data({2, 3}, {2.0F, 4.0F, 8.0F, 4.0F, 8.0F, 16.0F});
  const Tensor right = Tensor::from_data({3}, {8.0F, 16.0F, 32.0F});
  const Tensor::value_type* const original_storage = left.elements().data();
  const Tensor::storage_type expected_elements{0.25F, 0.25F, 0.25F,
                                                0.5F, 0.5F, 0.5F};

  const Tensor result = std::move(left) / right;

  expect(result.elements().data() == original_storage,
         "broadcast division did not reuse rvalue left storage");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "broadcast division with an rvalue left computed incorrect elements");
  expect_empty_sentinel(left);
}

void test_divide_consumes_rvalue_left_when_broadcast_expands_it() {
  Tensor left = Tensor::from_data({3}, {8.0F, 16.0F, 32.0F});
  const Tensor right =
      Tensor::from_data({2, 3}, {2.0F, 4.0F, 8.0F, 4.0F, 8.0F, 16.0F});
  const Tensor::storage_type expected_elements{4.0F, 4.0F, 4.0F,
                                                2.0F, 2.0F, 2.0F};

  const Tensor result = std::move(left) / right;

  expect(std::ranges::equal(result.elements(), expected_elements),
         "expanded rvalue division changed operand order");
  expect_empty_sentinel(left);
}

void test_divide_does_not_consume_rvalue_right() {
  const Tensor left = Tensor::from_data({3}, {8.0F, 16.0F, 32.0F});
  Tensor right =
      Tensor::from_data({2, 3}, {2.0F, 4.0F, 8.0F, 4.0F, 8.0F, 16.0F});
  const Tensor::storage_type expected_result{4.0F, 4.0F, 4.0F,
                                              2.0F, 2.0F, 2.0F};
  const Tensor::storage_type expected_right{2.0F, 4.0F, 8.0F,
                                             4.0F, 8.0F, 16.0F};

  const Tensor result = left / std::move(right);

  expect(std::ranges::equal(result.elements(), expected_result),
         "division with an rvalue right computed incorrect elements");
  expect(std::ranges::equal(right.elements(), expected_right),
         "division consumed the rvalue right operand");
}

void test_divide_rejects_incompatible_shapes_without_changing_lvalues() {
  const Tensor left =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor right =
      Tensor::from_data({3, 2}, {6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F});
  const Tensor::storage_type expected_left{1.0F, 2.0F, 3.0F,
                                            4.0F, 5.0F, 6.0F};
  const Tensor::storage_type expected_right{6.0F, 5.0F, 4.0F,
                                             3.0F, 2.0F, 1.0F};

  expect_throws<std::invalid_argument>(
      [&left, &right] { static_cast<void>(left / right); },
      "division with different shapes did not throw std::invalid_argument",
      "division with different shapes produced the wrong exception type");

  expect(std::ranges::equal(left.elements(), expected_left),
         "failed division changed the left lvalue elements");
  expect(std::ranges::equal(right.elements(), expected_right),
         "failed division changed the right lvalue elements");

  const Tensor zero_extent({2, 0, 3});
  const Tensor incompatible_zero_extent({2, 2, 3});

  expect_throws<std::invalid_argument>(
      [&zero_extent, &incompatible_zero_extent] {
        static_cast<void>(zero_extent / incompatible_zero_extent);
      },
      "division accepted incompatible zero and non-singleton extents",
      "division with incompatible zero extents produced the wrong exception "
      "type");
}

void test_divide_rejects_overflowing_broadcast_result_shape() {
  const Tensor::size_type max_extent =
      std::numeric_limits<Tensor::size_type>::max();
  const Tensor left({0, max_extent, 1});
  const Tensor right({0, 1, 2});

  expect_throws<std::overflow_error>(
      [&left, &right] { static_cast<void>(left / right); },
      "division accepted a broadcast result with overflowing strides",
      "division with an overflowing result shape produced the wrong exception "
      "type");
}

void test_failed_divide_consumes_rvalue_left() {
  Tensor left = Tensor::from_data({2}, {1.0F, 2.0F});
  const Tensor right = Tensor::from_data({3}, {3.0F, 4.0F, 5.0F});

  expect_throws<std::invalid_argument>(
      [&left, &right] { static_cast<void>(std::move(left) / right); },
      "failed division with an rvalue left did not throw",
      "failed division with an rvalue left produced the wrong exception type");

  expect_empty_sentinel(left);
}

void test_divide_rejects_moved_from_sentinels() {
  Tensor left_source = Tensor::from_data({1}, {1.0F});
  Tensor left_owner(std::move(left_source));
  static_cast<void>(left_owner);
  const Tensor ordinary = Tensor::from_data({1}, {2.0F});

  expect_throws<std::invalid_argument>(
      [&left_source, &ordinary] {
        static_cast<void>(left_source / ordinary);
      },
      "division accepted a sentinel left operand",
      "sentinel left division produced the wrong exception type");
  expect_empty_sentinel(left_source);

  Tensor right_source = Tensor::from_data({1}, {3.0F});
  Tensor right_owner(std::move(right_source));
  static_cast<void>(right_owner);

  expect_throws<std::invalid_argument>(
      [&ordinary, &right_source] {
        static_cast<void>(ordinary / right_source);
      },
      "division accepted a sentinel right operand",
      "sentinel right division produced the wrong exception type");
  expect_empty_sentinel(right_source);
}

}  // namespace

int main() {
  try {
    test_divide_computes_elementwise_without_changing_lvalues();
    test_divide_broadcasts_either_lower_rank_operand_in_operand_order();
    test_divide_broadcasts_singleton_axes_in_both_operands();
    test_divide_broadcasts_rank_zero_tensor_from_either_side();
    test_divide_supports_scalars_and_zero_extent_tensors();
    test_divide_broadcasts_zero_extent_axes();
    test_divide_supports_aliased_lvalues_without_changing_the_source();
    test_divide_reuses_rvalue_left_storage_and_consumes_it();
    test_divide_reuses_rvalue_left_storage_when_right_is_broadcast();
    test_divide_consumes_rvalue_left_when_broadcast_expands_it();
    test_divide_does_not_consume_rvalue_right();
    test_divide_rejects_incompatible_shapes_without_changing_lvalues();
    test_divide_rejects_overflowing_broadcast_result_shape();
    test_failed_divide_consumes_rvalue_left();
    test_divide_rejects_moved_from_sentinels();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor division tests passed\n";
  return EXIT_SUCCESS;
}
