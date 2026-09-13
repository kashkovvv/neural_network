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
concept CanMatmul = requires(LeftTensor&& left, RightTensor&& right) {
  std::forward<LeftTensor>(left).matmul(std::forward<RightTensor>(right));
};

static_assert(std::same_as<decltype(std::declval<const Tensor&>().matmul(
                               std::declval<const Tensor&>())),
                           Tensor>);
static_assert(CanMatmul<Tensor&, Tensor&>);
static_assert(CanMatmul<const Tensor&, const Tensor&>);
static_assert(CanMatmul<Tensor&&, Tensor&>);
static_assert(CanMatmul<const Tensor&&, const Tensor&>);
static_assert(CanMatmul<Tensor&, Tensor&&>);
static_assert(CanMatmul<DoubleTensor&, const DoubleTensor&>);
static_assert(!CanMatmul<IntegerTensor&, const IntegerTensor&>);
static_assert(!CanMatmul<Tensor&, const DoubleTensor&>);
static_assert(!noexcept(
    std::declval<const Tensor&>().matmul(std::declval<const Tensor&>())));

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

void test_matmul_multiplies_rectangular_matrices_without_changing_operands() {
  const Tensor left =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor right =
      Tensor::from_data({3, 2}, {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
  const Tensor::shape_type left_shape{2, 3};
  const Tensor::strides_type left_strides{3, 1};
  const Tensor::storage_type left_elements{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  const Tensor::shape_type right_shape{3, 2};
  const Tensor::strides_type right_strides{2, 1};
  const Tensor::storage_type right_elements{7.0F,  8.0F,  9.0F,
                                            10.0F, 11.0F, 12.0F};

  const Tensor result = left.matmul(right);

  expect_tensor(result, {2, 2}, {2, 1}, {58.0F, 64.0F, 139.0F, 154.0F},
                "matmul produced an incorrect rectangular result");
  expect_tensor(left, left_shape, left_strides, left_elements,
                "matmul changed the left operand");
  expect_tensor(right, right_shape, right_strides, right_elements,
                "matmul changed the right operand");
}

void test_matmul_supports_singleton_dimensions() {
  const Tensor one_element =
      Tensor::from_data({1, 3}, {1.0F, 2.0F, 3.0F})
          .matmul(Tensor::from_data({3, 1}, {4.0F, 5.0F, 6.0F}));
  const Tensor outer_like =
      Tensor::from_data({3, 1}, {1.0F, 2.0F, 3.0F})
          .matmul(Tensor::from_data({1, 2}, {4.0F, 5.0F}));

  expect_tensor(one_element, {1, 1}, {1, 1}, {32.0F},
                "matmul produced an incorrect one-element matrix");
  expect_tensor(outer_like, {3, 2}, {2, 1},
                {4.0F, 5.0F, 8.0F, 10.0F, 12.0F, 15.0F},
                "matmul produced an incorrect singleton-inner result");
}

void test_matmul_multiplies_matching_batches() {
  const Tensor left = Tensor::from_data(
      {2, 2, 3},
      {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 1.0F, 0.0F, 1.0F, 2.0F, 1.0F, 0.0F});
  const Tensor right =
      Tensor::from_data({2, 3, 2}, {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F, 1.0F,
                                    2.0F, 3.0F, 4.0F, 5.0F, 6.0F});

  const Tensor result = left.matmul(right);

  expect_tensor(result, {2, 2, 2}, {4, 2, 1},
                {58.0F, 64.0F, 139.0F, 154.0F, 6.0F, 8.0F, 5.0F, 8.0F},
                "matmul multiplied matching batches incorrectly");
}

void test_matmul_broadcasts_an_unbatched_operand() {
  const Tensor unbatched_left =
      Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  const Tensor batched_right =
      Tensor::from_data({2, 2, 1}, {5.0F, 6.0F, 7.0F, 8.0F});
  const Tensor batched_left =
      Tensor::from_data({2, 1, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  const Tensor unbatched_right =
      Tensor::from_data({2, 2}, {5.0F, 6.0F, 7.0F, 8.0F});

  const Tensor right_batched_result = unbatched_left.matmul(batched_right);
  const Tensor left_batched_result = batched_left.matmul(unbatched_right);

  expect_tensor(right_batched_result, {2, 2, 1}, {2, 1, 1},
                {17.0F, 39.0F, 23.0F, 53.0F},
                "matmul did not broadcast an unbatched left operand");
  expect_tensor(left_batched_result, {2, 1, 2}, {2, 2, 1},
                {19.0F, 22.0F, 43.0F, 50.0F},
                "matmul did not broadcast an unbatched right operand");
}

void test_matmul_broadcasts_missing_and_singleton_batch_axes() {
  const Tensor left = Tensor::from_data({2, 1, 1, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  const Tensor right =
      Tensor::from_data({3, 2, 1}, {5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F});

  const Tensor result = left.matmul(right);

  expect_tensor(result, {2, 3, 1, 1}, {3, 1, 1, 1},
                {17.0F, 23.0F, 29.0F, 39.0F, 53.0F, 67.0F},
                "matmul broadcast batch axes incorrectly");
}

void test_matmul_supports_zero_extent_dimensions() {
  const Tensor empty_inner = Tensor({2, 0}).matmul(Tensor({0, 3}));
  const Tensor zero_rows = Tensor({0, 2}).matmul(
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}));
  const Tensor zero_columns =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F})
          .matmul(Tensor({3, 0}));
  const Tensor fully_empty = Tensor({0, 0}).matmul(Tensor({0, 0}));

  expect_tensor(empty_inner, {2, 3}, {3, 1},
                {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F},
                "matmul did not use zero for an empty inner dimension");
  expect_tensor(zero_rows, {0, 3}, {3, 1}, {},
                "matmul produced an incorrect zero-row result");
  expect_tensor(zero_columns, {2, 0}, {0, 1}, {},
                "matmul produced an incorrect zero-column result");
  expect_tensor(fully_empty, {0, 0}, {0, 1}, {},
                "matmul produced an incorrect fully empty result");
}

void test_matmul_supports_zero_extent_batch_axes() {
  const Tensor empty_left_batch = Tensor({0, 2, 3}).matmul(Tensor({1, 3, 4}));
  const Tensor empty_right_batch =
      Tensor::from_data({1, 2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F})
          .matmul(Tensor({0, 3, 4}));
  const Tensor empty_inner_batch = Tensor({2, 2, 0}).matmul(Tensor({1, 0, 3}));

  expect_tensor(empty_left_batch, {0, 2, 4}, {8, 4, 1}, {},
                "matmul produced elements for an empty left batch");
  expect_tensor(empty_right_batch, {0, 2, 4}, {8, 4, 1}, {},
                "matmul produced elements for an empty right batch");
  expect_tensor(
      empty_inner_batch, {2, 2, 3}, {6, 3, 1},
      {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F},
      "batched matmul did not use zero for an empty inner axis");
}

void test_matmul_uses_native_floating_point_behavior() {
  const Tensor infinity_result =
      Tensor::from_data({1, 1}, {std::numeric_limits<float>::infinity()})
          .matmul(Tensor::from_data({1, 1}, {2.0F}));
  const Tensor nan_result =
      Tensor::from_data({1, 1}, {0.0F})
          .matmul(Tensor::from_data({1, 1},
                                    {std::numeric_limits<float>::infinity()}));

  expect(
      std::isinf(infinity_result.at(0, 0)) && infinity_result.at(0, 0) > 0.0F,
      "matmul did not preserve native floating-point infinity behavior");
  expect(std::isnan(nan_result.at(0, 0)),
         "matmul did not preserve native floating-point NaN behavior");
}

void test_matmul_accepts_rvalues_without_consuming_operands() {
  Tensor left = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  Tensor right = Tensor::from_data({2, 2}, {5.0F, 6.0F, 7.0F, 8.0F});
  const Tensor::storage_type expected_left{1.0F, 2.0F, 3.0F, 4.0F};
  const Tensor::storage_type expected_right{5.0F, 6.0F, 7.0F, 8.0F};

  const Tensor result = std::move(left).matmul(std::move(right));

  expect_tensor(result, {2, 2}, {2, 1}, {19.0F, 22.0F, 43.0F, 50.0F},
                "rvalue matmul produced an incorrect result");
  expect(std::ranges::equal(left.elements(), expected_left),
         "matmul consumed its rvalue left operand");
  expect(std::ranges::equal(right.elements(), expected_right),
         "matmul consumed its rvalue right operand");
}

void test_matmul_rejects_operands_below_rank_two() {
  const Tensor valid_matrix =
      Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  const Tensor scalar = Tensor::scalar(1.0F);
  const Tensor vector = Tensor::from_data({2}, {1.0F, 2.0F});

  expect_throws<std::invalid_argument>(
      [&scalar, &valid_matrix] {
        static_cast<void>(scalar.matmul(valid_matrix));
      },
      "matmul accepted a rank-zero left operand",
      "matmul with a rank-zero left operand produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&vector, &valid_matrix] {
        static_cast<void>(vector.matmul(valid_matrix));
      },
      "matmul accepted a rank-one left operand",
      "matmul with a rank-one left operand produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&valid_matrix, &scalar] {
        static_cast<void>(valid_matrix.matmul(scalar));
      },
      "matmul accepted a rank-zero right operand",
      "matmul with a rank-zero right operand produced the wrong exception "
      "type");
  expect_throws<std::invalid_argument>(
      [&valid_matrix, &vector] {
        static_cast<void>(valid_matrix.matmul(vector));
      },
      "matmul accepted a rank-one right operand",
      "matmul with a rank-one right operand produced the wrong exception type");
}

void test_matmul_rejects_incompatible_batch_axes_without_changes() {
  const Tensor left =
      Tensor::from_data({2, 2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F,
                                    8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
  const Tensor right = Tensor::from_data(
      {3, 3, 1}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F});
  const Tensor zero_batch_left = Tensor({0, 2, 3});
  const Tensor nonempty_batch_right = Tensor({2, 3, 1});
  const Tensor::storage_type expected_left(left.elements().begin(),
                                           left.elements().end());
  const Tensor::storage_type expected_right(right.elements().begin(),
                                            right.elements().end());

  expect_throws<std::invalid_argument>(
      [&left, &right] { static_cast<void>(left.matmul(right)); },
      "matmul accepted incompatible nonzero batch axes",
      "incompatible matmul batch axes produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&zero_batch_left, &nonempty_batch_right] {
        static_cast<void>(zero_batch_left.matmul(nonempty_batch_right));
      },
      "matmul accepted incompatible zero-extent batch axes",
      "incompatible zero-extent batch axes produced the wrong exception "
      "type");
  expect_tensor(left, {2, 2, 3}, {6, 3, 1}, expected_left,
                "failed batch broadcasting changed the left operand");
  expect_tensor(right, {3, 3, 1}, {3, 1, 1}, expected_right,
                "failed batch broadcasting changed the right operand");
  expect_tensor(zero_batch_left, {0, 2, 3}, {6, 3, 1}, {},
                "failed zero-extent broadcasting changed the left operand");
  expect_tensor(nonempty_batch_right, {2, 3, 1}, {3, 1, 1},
                {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F},
                "failed zero-extent broadcasting changed the right operand");
}

void test_matmul_rejects_inner_dimension_mismatch_without_changes() {
  const Tensor left =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor right = Tensor::from_data({2, 2}, {7.0F, 8.0F, 9.0F, 10.0F});
  const Tensor zero_row_left = Tensor({0, 3});
  const Tensor zero_column_right = Tensor({2, 0});
  const Tensor batched_left = Tensor({2, 2, 3});
  const Tensor batched_right = Tensor({1, 4, 2});
  const Tensor::storage_type expected_left{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  const Tensor::storage_type expected_right{7.0F, 8.0F, 9.0F, 10.0F};

  expect_throws<std::invalid_argument>(
      [&left, &right] { static_cast<void>(left.matmul(right)); },
      "matmul accepted incompatible inner dimensions",
      "matmul dimension mismatch produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&zero_row_left, &right] {
        static_cast<void>(zero_row_left.matmul(right));
      },
      "zero-row matmul accepted incompatible inner dimensions",
      "zero-row matmul mismatch produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&left, &zero_column_right] {
        static_cast<void>(left.matmul(zero_column_right));
      },
      "zero-column matmul accepted incompatible inner dimensions",
      "zero-column matmul mismatch produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&batched_left, &batched_right] {
        static_cast<void>(batched_left.matmul(batched_right));
      },
      "batched matmul accepted incompatible inner dimensions",
      "batched matmul mismatch produced the wrong exception type");
  expect(std::ranges::equal(left.elements(), expected_left),
         "failed matmul changed the left operand");
  expect(std::ranges::equal(right.elements(), expected_right),
         "failed matmul changed the right operand");
}

void test_matmul_rejects_moved_from_sentinel_on_each_side() {
  Tensor left_source = Tensor::from_data({1, 1}, {1.0F});
  Tensor left_owner(std::move(left_source));
  Tensor right_source = Tensor::from_data({1, 1}, {2.0F});
  Tensor right_owner(std::move(right_source));
  const Tensor valid = Tensor::from_data({1, 1}, {3.0F});
  static_cast<void>(left_owner);
  static_cast<void>(right_owner);

  expect_throws<std::invalid_argument>(
      [&left_source, &valid] { static_cast<void>(left_source.matmul(valid)); },
      "matmul accepted a moved-from left operand",
      "matmul with a moved-from left operand produced the wrong exception "
      "type");
  expect_throws<std::invalid_argument>(
      [&right_source, &valid] {
        static_cast<void>(valid.matmul(right_source));
      },
      "matmul accepted a moved-from right operand",
      "matmul with a moved-from right operand produced the wrong exception "
      "type");
  expect_empty_sentinel(left_source);
  expect_empty_sentinel(right_source);
}

}  // namespace

int main() {
  try {
    test_matmul_multiplies_rectangular_matrices_without_changing_operands();
    test_matmul_supports_singleton_dimensions();
    test_matmul_multiplies_matching_batches();
    test_matmul_broadcasts_an_unbatched_operand();
    test_matmul_broadcasts_missing_and_singleton_batch_axes();
    test_matmul_supports_zero_extent_dimensions();
    test_matmul_supports_zero_extent_batch_axes();
    test_matmul_uses_native_floating_point_behavior();
    test_matmul_accepts_rvalues_without_consuming_operands();
    test_matmul_rejects_operands_below_rank_two();
    test_matmul_rejects_incompatible_batch_axes_without_changes();
    test_matmul_rejects_inner_dimension_mismatch_without_changes();
    test_matmul_rejects_moved_from_sentinel_on_each_side();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor matmul tests passed\n";
  return EXIT_SUCCESS;
}
