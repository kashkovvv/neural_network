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
concept CanAdd = requires {
  std::declval<Left>() + std::declval<Right>();
};

template <typename Left, typename Right>
concept CanCallFreeAdd = requires {
  operator+(std::declval<Left>(), std::declval<Right>());
};

static_assert(std::same_as<decltype(std::declval<const Tensor&>() +
                                    std::declval<const Tensor&>()),
                           Tensor>);
static_assert(CanCallFreeAdd<const Tensor&, const Tensor&>);
static_assert(CanAdd<Tensor&, const Tensor&>);
static_assert(CanAdd<const Tensor&, const Tensor&>);
static_assert(CanAdd<Tensor&&, const Tensor&>);
static_assert(CanAdd<const Tensor&, Tensor&&>);
static_assert(CanAdd<DoubleTensor&, const DoubleTensor&>);
static_assert(!CanAdd<IntegerTensor&, const IntegerTensor&>);
static_assert(!CanAdd<Tensor&, const DoubleTensor&>);
static_assert(
    !noexcept(std::declval<const Tensor&>() + std::declval<const Tensor&>()));

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

void test_add_computes_elementwise_without_changing_lvalues() {
  const Tensor left =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor right =
      Tensor::from_data({2, 3}, {6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F});
  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_result{7.0F, 7.0F, 7.0F,
                                              7.0F, 7.0F, 7.0F};
  const Tensor::storage_type expected_left{1.0F, 2.0F, 3.0F,
                                            4.0F, 5.0F, 6.0F};
  const Tensor::storage_type expected_right{6.0F, 5.0F, 4.0F,
                                             3.0F, 2.0F, 1.0F};

  const Tensor result = left + right;

  expect_state(result, expected_shape, expected_strides, expected_result,
               "addition produced an incorrect shape",
               "addition produced incorrect strides",
               "addition produced incorrect elements");
  expect(std::ranges::equal(left.elements(), expected_left),
         "addition changed the left lvalue elements");
  expect(std::ranges::equal(right.elements(), expected_right),
         "addition changed the right lvalue elements");
}

void test_add_broadcasts_either_lower_rank_operand() {
  const Tensor vector = Tensor::from_data({3}, {10.0F, 20.0F, 30.0F});
  const Tensor matrix =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_elements{11.0F, 22.0F, 33.0F,
                                                14.0F, 25.0F, 36.0F};
  const Tensor::storage_type expected_vector{10.0F, 20.0F, 30.0F};
  const Tensor::storage_type expected_matrix{1.0F, 2.0F, 3.0F,
                                              4.0F, 5.0F, 6.0F};

  const Tensor matrix_plus_vector = matrix + vector;
  const Tensor vector_plus_matrix = vector + matrix;

  expect_state(matrix_plus_vector, expected_shape, expected_strides,
               expected_elements,
               "right broadcast changed the addition result shape",
               "right broadcast produced incorrect addition result strides",
               "right broadcast produced incorrect addition elements");
  expect_state(vector_plus_matrix, expected_shape, expected_strides,
               expected_elements,
               "left broadcast changed the addition result shape",
               "left broadcast produced incorrect addition result strides",
               "left broadcast produced incorrect addition elements");
  expect(std::ranges::equal(vector.elements(), expected_vector),
         "broadcast addition changed the vector operand");
  expect(std::ranges::equal(matrix.elements(), expected_matrix),
         "broadcast addition changed the matrix operand");
}

void test_add_broadcasts_singleton_axes_in_both_operands() {
  const Tensor left =
      Tensor::from_data({2, 1, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor right =
      Tensor::from_data({1, 4, 1}, {10.0F, 20.0F, 30.0F, 40.0F});
  const Tensor::shape_type expected_shape{2, 4, 3};
  const Tensor::strides_type expected_strides{12, 3, 1};
  const Tensor::storage_type expected_elements{
      11.0F, 12.0F, 13.0F, 21.0F, 22.0F, 23.0F,
      31.0F, 32.0F, 33.0F, 41.0F, 42.0F, 43.0F,
      14.0F, 15.0F, 16.0F, 24.0F, 25.0F, 26.0F,
      34.0F, 35.0F, 36.0F, 44.0F, 45.0F, 46.0F};

  const Tensor result = left + right;

  expect_state(result, expected_shape, expected_strides, expected_elements,
               "singleton broadcast produced an incorrect result shape",
               "singleton broadcast produced incorrect result strides",
               "singleton broadcast produced incorrect addition elements");
}

void test_add_broadcasts_rank_zero_tensor_from_either_side() {
  const Tensor scalar = Tensor::scalar(10.0F);
  const Tensor matrix =
      Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  const Tensor::shape_type expected_shape{2, 2};
  const Tensor::strides_type expected_strides{2, 1};
  const Tensor::storage_type expected_elements{11.0F, 12.0F, 13.0F, 14.0F};

  const Tensor scalar_plus_matrix = scalar + matrix;
  const Tensor matrix_plus_scalar = matrix + scalar;

  expect_state(scalar_plus_matrix, expected_shape, expected_strides,
               expected_elements,
               "left rank-zero broadcast produced an incorrect result shape",
               "left rank-zero broadcast produced incorrect result strides",
               "left rank-zero broadcast produced incorrect elements");
  expect_state(matrix_plus_scalar, expected_shape, expected_strides,
               expected_elements,
               "right rank-zero broadcast produced an incorrect result shape",
               "right rank-zero broadcast produced incorrect result strides",
               "right rank-zero broadcast produced incorrect elements");
}

void test_add_supports_scalars_and_zero_extent_tensors() {
  const Tensor scalar_result = Tensor::scalar(2.5F) + Tensor::scalar(-0.5F);

  expect(scalar_result.rank() == 0, "scalar addition changed rank");
  expect(scalar_result.numel() == 1, "scalar addition changed numel");
  expect(scalar_result.at() == 2.0F,
         "scalar addition produced an incorrect value");

  const Tensor zero_extent_result = Tensor({2, 0, 4}) + Tensor({2, 0, 4});
  const Tensor::shape_type expected_shape{2, 0, 4};
  const Tensor::strides_type expected_strides{0, 4, 1};

  expect_state(zero_extent_result, expected_shape, expected_strides, {},
               "zero-extent addition changed shape",
               "zero-extent addition produced incorrect strides",
               "zero-extent addition created elements");
}

void test_add_broadcasts_zero_extent_axes() {
  const Tensor zero_extent({2, 0, 3});
  const Tensor row = Tensor::from_data({1, 3}, {1.0F, 2.0F, 3.0F});
  const Tensor singleton_axis =
      Tensor::from_data({2, 1, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor::shape_type expected_shape{2, 0, 3};
  const Tensor::strides_type expected_strides{0, 3, 1};

  const Tensor unchanged_shape_result = zero_extent + row;
  const Tensor expanded_left_result = singleton_axis + zero_extent;

  expect_state(unchanged_shape_result, expected_shape, expected_strides, {},
               "zero-extent broadcast produced an incorrect result shape",
               "zero-extent broadcast produced incorrect result strides",
               "zero-extent broadcast created elements");
  expect_state(expanded_left_result, expected_shape, expected_strides, {},
               "zero-extent broadcast did not expand the left shape",
               "expanded zero-extent result has incorrect strides",
               "expanded zero-extent result contains elements");
}

void test_add_supports_aliased_lvalues_without_changing_the_source() {
  const Tensor tensor =
      Tensor::from_data({2, 2}, {1.0F, -2.0F, 3.5F, 0.0F});
  const Tensor::storage_type expected_source{1.0F, -2.0F, 3.5F, 0.0F};
  const Tensor::storage_type expected_result{2.0F, -4.0F, 7.0F, 0.0F};

  const Tensor result = tensor + tensor;

  expect(std::ranges::equal(result.elements(), expected_result),
         "addition of aliased lvalues computed incorrect elements");
  expect(std::ranges::equal(tensor.elements(), expected_source),
         "addition of aliased lvalues changed the source");
}

void test_add_reuses_rvalue_left_storage_and_consumes_it() {
  Tensor left = Tensor::from_data({3}, {1.0F, 2.0F, 3.0F});
  const Tensor right = Tensor::from_data({3}, {4.0F, 5.0F, 6.0F});
  const Tensor::value_type* const original_storage = left.elements().data();
  const Tensor::storage_type expected_elements{5.0F, 7.0F, 9.0F};

  const Tensor result = std::move(left) + right;

  expect(result.elements().data() == original_storage,
         "addition did not reuse the rvalue left storage");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "addition with an rvalue left computed incorrect elements");
  expect_empty_sentinel(left);
}

void test_add_reuses_rvalue_left_storage_when_right_is_broadcast() {
  Tensor left =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor right = Tensor::from_data({3}, {10.0F, 20.0F, 30.0F});
  const Tensor::value_type* const original_storage = left.elements().data();
  const Tensor::storage_type expected_elements{11.0F, 22.0F, 33.0F,
                                                14.0F, 25.0F, 36.0F};

  const Tensor result = std::move(left) + right;

  expect(result.elements().data() == original_storage,
         "broadcast addition did not reuse the rvalue left storage");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "broadcast addition with an rvalue left computed incorrect "
         "elements");
  expect_empty_sentinel(left);
}

void test_add_consumes_rvalue_left_when_broadcast_expands_it() {
  Tensor left = Tensor::from_data({3}, {10.0F, 20.0F, 30.0F});
  const Tensor right =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_elements{11.0F, 22.0F, 33.0F,
                                                14.0F, 25.0F, 36.0F};

  const Tensor result = std::move(left) + right;

  expect_state(result, expected_shape, expected_strides, expected_elements,
               "expanded rvalue addition produced an incorrect shape",
               "expanded rvalue addition produced incorrect strides",
               "expanded rvalue addition produced incorrect elements");
  expect_empty_sentinel(left);
}

void test_add_does_not_consume_rvalue_right() {
  const Tensor left = Tensor::from_data({3}, {10.0F, 20.0F, 30.0F});
  Tensor right =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor::storage_type expected_result{11.0F, 22.0F, 33.0F,
                                              14.0F, 25.0F, 36.0F};
  const Tensor::storage_type expected_right{1.0F, 2.0F, 3.0F,
                                             4.0F, 5.0F, 6.0F};

  const Tensor result = left + std::move(right);

  expect(std::ranges::equal(result.elements(), expected_result),
         "addition with an rvalue right computed incorrect elements");
  expect(std::ranges::equal(right.elements(), expected_right),
         "addition consumed the rvalue right operand");
}

void test_add_rejects_incompatible_shapes_without_changing_lvalues() {
  const Tensor left =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor right =
      Tensor::from_data({3, 2}, {6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F});
  const Tensor::storage_type expected_left{1.0F, 2.0F, 3.0F,
                                            4.0F, 5.0F, 6.0F};
  const Tensor::storage_type expected_right{6.0F, 5.0F, 4.0F,
                                             3.0F, 2.0F, 1.0F};

  expect_throws<std::invalid_argument>(
      [&left, &right] { static_cast<void>(left + right); },
      "addition with different shapes did not throw std::invalid_argument",
      "addition with different shapes produced the wrong exception type");

  expect(std::ranges::equal(left.elements(), expected_left),
         "failed addition changed the left lvalue elements");
  expect(std::ranges::equal(right.elements(), expected_right),
         "failed addition changed the right lvalue elements");

  const Tensor zero_extent({2, 0, 3});
  const Tensor incompatible_zero_extent({2, 2, 3});

  expect_throws<std::invalid_argument>(
      [&zero_extent, &incompatible_zero_extent] {
        static_cast<void>(zero_extent + incompatible_zero_extent);
      },
      "addition accepted incompatible zero and non-singleton extents",
      "addition with incompatible zero extents produced the wrong exception "
      "type");
}

void test_add_rejects_overflowing_broadcast_result_shape() {
  const Tensor::size_type max_extent =
      std::numeric_limits<Tensor::size_type>::max();
  const Tensor left({0, max_extent, 1});
  const Tensor right({0, 1, 2});
  const Tensor::shape_type expected_left_shape{0, max_extent, 1};
  const Tensor::strides_type expected_left_strides{max_extent, 1, 1};
  const Tensor::shape_type expected_right_shape{0, 1, 2};
  const Tensor::strides_type expected_right_strides{2, 2, 1};

  expect_throws<std::overflow_error>(
      [&left, &right] { static_cast<void>(left + right); },
      "addition accepted a broadcast result with overflowing strides",
      "addition with an overflowing result shape produced the wrong exception "
      "type");

  expect_state(left, expected_left_shape, expected_left_strides, {},
               "failed overflowing addition changed the left shape",
               "failed overflowing addition changed the left strides",
               "failed overflowing addition changed the left elements");
  expect_state(right, expected_right_shape, expected_right_strides, {},
               "failed overflowing addition changed the right shape",
               "failed overflowing addition changed the right strides",
               "failed overflowing addition changed the right elements");
}

void test_failed_add_consumes_rvalue_left() {
  Tensor left = Tensor::from_data({2}, {1.0F, 2.0F});
  const Tensor right = Tensor::from_data({3}, {3.0F, 4.0F, 5.0F});

  expect_throws<std::invalid_argument>(
      [&left, &right] { static_cast<void>(std::move(left) + right); },
      "failed addition with an rvalue left did not throw",
      "failed addition with an rvalue left produced the wrong exception type");

  expect_empty_sentinel(left);
}

void test_add_rejects_moved_from_sentinels() {
  Tensor left_source = Tensor::from_data({1}, {1.0F});
  Tensor left_owner(std::move(left_source));
  static_cast<void>(left_owner);
  const Tensor ordinary = Tensor::from_data({1}, {2.0F});

  expect_throws<std::invalid_argument>(
      [&left_source, &ordinary] {
        static_cast<void>(left_source + ordinary);
      },
      "addition accepted a sentinel left operand",
      "sentinel left addition produced the wrong exception type");
  expect_empty_sentinel(left_source);

  Tensor right_source = Tensor::from_data({1}, {3.0F});
  Tensor right_owner(std::move(right_source));
  static_cast<void>(right_owner);

  expect_throws<std::invalid_argument>(
      [&ordinary, &right_source] {
        static_cast<void>(ordinary + right_source);
      },
      "addition accepted a sentinel right operand",
      "sentinel right addition produced the wrong exception type");
  expect_empty_sentinel(right_source);
}

}  // namespace

int main() {
  try {
    test_add_computes_elementwise_without_changing_lvalues();
    test_add_broadcasts_either_lower_rank_operand();
    test_add_broadcasts_singleton_axes_in_both_operands();
    test_add_broadcasts_rank_zero_tensor_from_either_side();
    test_add_supports_scalars_and_zero_extent_tensors();
    test_add_broadcasts_zero_extent_axes();
    test_add_supports_aliased_lvalues_without_changing_the_source();
    test_add_reuses_rvalue_left_storage_and_consumes_it();
    test_add_reuses_rvalue_left_storage_when_right_is_broadcast();
    test_add_consumes_rvalue_left_when_broadcast_expands_it();
    test_add_does_not_consume_rvalue_right();
    test_add_rejects_incompatible_shapes_without_changing_lvalues();
    test_add_rejects_overflowing_broadcast_result_shape();
    test_failed_add_consumes_rvalue_left();
    test_add_rejects_moved_from_sentinels();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor addition tests passed\n";
  return EXIT_SUCCESS;
}
