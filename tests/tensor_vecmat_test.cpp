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

template <typename VectorTensor, typename MatrixTensor>
concept CanVecmat = requires(VectorTensor&& vector, MatrixTensor&& matrix) {
  std::forward<VectorTensor>(vector).vecmat(std::forward<MatrixTensor>(matrix));
};

static_assert(std::same_as<decltype(std::declval<const Tensor&>().vecmat(
                               std::declval<const Tensor&>())),
                           Tensor>);
static_assert(CanVecmat<Tensor&, Tensor&>);
static_assert(CanVecmat<const Tensor&, const Tensor&>);
static_assert(CanVecmat<Tensor&&, Tensor&>);
static_assert(CanVecmat<const Tensor&&, const Tensor&>);
static_assert(CanVecmat<Tensor&, Tensor&&>);
static_assert(CanVecmat<DoubleTensor&, const DoubleTensor&>);
static_assert(!CanVecmat<IntegerTensor&, const IntegerTensor&>);
static_assert(!CanVecmat<Tensor&, const DoubleTensor&>);
static_assert(!noexcept(
    std::declval<const Tensor&>().vecmat(std::declval<const Tensor&>())));

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

void test_vecmat_multiplies_rectangular_matrix_without_changing_operands() {
  const Tensor vector = Tensor::from_data({3}, {2.0F, -1.0F, 0.5F});
  const Tensor matrix =
      Tensor::from_data({3, 2}, {1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F});
  const Tensor::shape_type vector_shape{3};
  const Tensor::strides_type vector_strides{1};
  const Tensor::storage_type vector_elements{2.0F, -1.0F, 0.5F};
  const Tensor::shape_type matrix_shape{3, 2};
  const Tensor::strides_type matrix_strides{2, 1};
  const Tensor::storage_type matrix_elements{1.0F, 4.0F, 2.0F,
                                             5.0F, 3.0F, 6.0F};

  const Tensor result = vector.vecmat(matrix);

  expect_tensor(result, {2}, {1}, {1.5F, 6.0F},
                "vecmat produced an incorrect result");
  expect_tensor(vector, vector_shape, vector_strides, vector_elements,
                "vecmat changed the vector operand");
  expect_tensor(matrix, matrix_shape, matrix_strides, matrix_elements,
                "vecmat changed the matrix operand");
}

void test_vecmat_compensates_each_column_independently() {
  const Tensor vector = Tensor::from_data({3}, {1.0e20F, 1.0F, -1.0e20F});
  const Tensor matrix =
      Tensor::from_data({3, 2}, {1.0F, -1.0F, 1.0F, 1.0F, 1.0F, -1.0F});

  const Tensor result = vector.vecmat(matrix);

  expect_tensor(result, {2}, {1}, {1.0F, 1.0F},
                "vecmat did not compensate each column independently");
}

void test_vecmat_supports_singleton_dimensions() {
  const Tensor one_row =
      Tensor::from_data({1}, {2.5F})
          .vecmat(Tensor::from_data({1, 3}, {2.0F, -3.0F, 4.0F}));
  const Tensor one_column =
      Tensor::from_data({3}, {3.0F, 0.5F, 2.0F})
          .vecmat(Tensor::from_data({3, 1}, {1.0F, -2.0F, 4.0F}));

  expect_tensor(one_row, {3}, {1}, {5.0F, -7.5F, 10.0F},
                "vecmat produced an incorrect one-row result");
  expect_tensor(one_column, {1}, {1}, {10.0F},
                "vecmat produced an incorrect one-column result");
}

void test_vecmat_supports_zero_extent_dimensions() {
  const Tensor empty_inner = Tensor({0}).vecmat(Tensor({0, 3}));
  const Tensor zero_columns =
      Tensor::from_data({2}, {1.0F, 2.0F}).vecmat(Tensor({2, 0}));
  const Tensor fully_empty = Tensor({0}).vecmat(Tensor({0, 0}));

  expect_tensor(empty_inner, {3}, {1}, {0.0F, 0.0F, 0.0F},
                "vecmat did not use zero for an empty inner dimension");
  expect_tensor(zero_columns, {0}, {1}, {},
                "vecmat produced an incorrect zero-column result");
  expect_tensor(fully_empty, {0}, {1}, {},
                "vecmat produced an incorrect fully empty result");
}

void test_vecmat_uses_native_floating_point_behavior() {
  const Tensor infinity_result =
      Tensor::from_data({1}, {std::numeric_limits<float>::infinity()})
          .vecmat(Tensor::from_data({1, 1}, {2.0F}));
  const Tensor overflow_result =
      Tensor::from_data({1}, {std::numeric_limits<float>::max()})
          .vecmat(Tensor::from_data({1, 1}, {2.0F}));
  const Tensor nan_result =
      Tensor::from_data({1}, {0.0F})
          .vecmat(Tensor::from_data({1, 1},
                                    {std::numeric_limits<float>::infinity()}));

  expect(std::isinf(infinity_result.at(0)) && infinity_result.at(0) > 0.0F,
         "vecmat did not preserve native floating-point infinity behavior");
  expect(std::isinf(overflow_result.at(0)) && overflow_result.at(0) > 0.0F,
         "vecmat product overflow did not produce positive infinity");
  expect(std::isnan(nan_result.at(0)),
         "vecmat did not preserve native floating-point NaN behavior");
}

void test_vecmat_accepts_rvalues_without_consuming_operands() {
  Tensor vector = Tensor::from_data({2}, {5.0F, 6.0F});
  Tensor matrix = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  const Tensor::storage_type expected_vector{5.0F, 6.0F};
  const Tensor::storage_type expected_matrix{1.0F, 2.0F, 3.0F, 4.0F};

  const Tensor result = std::move(vector).vecmat(std::move(matrix));

  expect_tensor(result, {2}, {1}, {23.0F, 34.0F},
                "rvalue vecmat produced an incorrect result");
  expect(std::ranges::equal(vector.elements(), expected_vector),
         "vecmat consumed its rvalue vector operand");
  expect(std::ranges::equal(matrix.elements(), expected_matrix),
         "vecmat consumed its rvalue matrix operand");
}

void test_vecmat_rejects_invalid_operand_ranks() {
  const Tensor valid_vector = Tensor::from_data({2}, {1.0F, 2.0F});
  const Tensor valid_matrix = Tensor::from_data({2, 1}, {3.0F, 4.0F});
  const Tensor scalar = Tensor::scalar(1.0F);
  const Tensor rank_two = Tensor::from_data({1, 2}, {1.0F, 2.0F});
  const Tensor rank_one = Tensor::from_data({2}, {3.0F, 4.0F});
  const Tensor rank_three = Tensor({1, 2, 1});

  expect_throws<std::invalid_argument>(
      [&scalar, &valid_matrix] {
        static_cast<void>(scalar.vecmat(valid_matrix));
      },
      "vecmat accepted a rank-zero vector operand",
      "vecmat with a rank-zero vector produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&rank_two, &valid_matrix] {
        static_cast<void>(rank_two.vecmat(valid_matrix));
      },
      "vecmat accepted a rank-two vector operand",
      "vecmat with a rank-two vector produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&valid_vector, &scalar] {
        static_cast<void>(valid_vector.vecmat(scalar));
      },
      "vecmat accepted a rank-zero matrix operand",
      "vecmat with a rank-zero matrix produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&valid_vector, &rank_one] {
        static_cast<void>(valid_vector.vecmat(rank_one));
      },
      "vecmat accepted a rank-one matrix operand",
      "vecmat with a rank-one matrix produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&valid_vector, &rank_three] {
        static_cast<void>(valid_vector.vecmat(rank_three));
      },
      "vecmat accepted a rank-three matrix operand",
      "vecmat with a rank-three matrix produced the wrong exception type");
}

void test_vecmat_rejects_inner_dimension_mismatch_without_changes() {
  const Tensor vector = Tensor::from_data({2}, {1.0F, 2.0F});
  const Tensor matrix =
      Tensor::from_data({3, 2}, {3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
  const Tensor zero_column_matrix = Tensor({3, 0});
  const Tensor::storage_type expected_vector{1.0F, 2.0F};
  const Tensor::storage_type expected_matrix{3.0F, 4.0F, 5.0F,
                                             6.0F, 7.0F, 8.0F};

  expect_throws<std::invalid_argument>(
      [&vector, &matrix] { static_cast<void>(vector.vecmat(matrix)); },
      "vecmat accepted incompatible inner dimensions",
      "vecmat dimension mismatch produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&vector, &zero_column_matrix] {
        static_cast<void>(vector.vecmat(zero_column_matrix));
      },
      "zero-column vecmat accepted incompatible inner dimensions",
      "zero-column vecmat mismatch produced the wrong exception type");
  expect(std::ranges::equal(vector.elements(), expected_vector),
         "failed vecmat changed the vector operand");
  expect(std::ranges::equal(matrix.elements(), expected_matrix),
         "failed vecmat changed the matrix operand");
}

void test_vecmat_rejects_moved_from_sentinel_on_each_side() {
  Tensor vector_source = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor vector_owner(std::move(vector_source));
  Tensor matrix_source = Tensor::from_data({2, 1}, {3.0F, 4.0F});
  Tensor matrix_owner(std::move(matrix_source));
  const Tensor valid_vector = Tensor::from_data({2}, {5.0F, 6.0F});
  const Tensor valid_matrix = Tensor::from_data({2, 1}, {7.0F, 8.0F});
  static_cast<void>(vector_owner);
  static_cast<void>(matrix_owner);

  expect_throws<std::invalid_argument>(
      [&vector_source, &valid_matrix] {
        static_cast<void>(vector_source.vecmat(valid_matrix));
      },
      "vecmat accepted a moved-from vector operand",
      "vecmat with a moved-from vector produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&matrix_source, &valid_vector] {
        static_cast<void>(valid_vector.vecmat(matrix_source));
      },
      "vecmat accepted a moved-from matrix operand",
      "vecmat with a moved-from matrix produced the wrong exception type");
  expect_empty_sentinel(vector_source);
  expect_empty_sentinel(matrix_source);
}

}  // namespace

int main() {
  try {
    test_vecmat_multiplies_rectangular_matrix_without_changing_operands();
    test_vecmat_compensates_each_column_independently();
    test_vecmat_supports_singleton_dimensions();
    test_vecmat_supports_zero_extent_dimensions();
    test_vecmat_uses_native_floating_point_behavior();
    test_vecmat_accepts_rvalues_without_consuming_operands();
    test_vecmat_rejects_invalid_operand_ranks();
    test_vecmat_rejects_inner_dimension_mismatch_without_changes();
    test_vecmat_rejects_moved_from_sentinel_on_each_side();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor vecmat tests passed\n";
  return EXIT_SUCCESS;
}
