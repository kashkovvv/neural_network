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

template <typename MatrixTensor, typename VectorTensor>
concept CanMatvec = requires(MatrixTensor&& matrix, VectorTensor&& vector) {
  std::forward<MatrixTensor>(matrix).matvec(std::forward<VectorTensor>(vector));
};

static_assert(std::same_as<decltype(std::declval<const Tensor&>().matvec(
                               std::declval<const Tensor&>())),
                           Tensor>);
static_assert(CanMatvec<Tensor&, Tensor&>);
static_assert(CanMatvec<const Tensor&, const Tensor&>);
static_assert(CanMatvec<Tensor&&, Tensor&>);
static_assert(CanMatvec<const Tensor&&, const Tensor&>);
static_assert(CanMatvec<Tensor&, Tensor&&>);
static_assert(CanMatvec<DoubleTensor&, const DoubleTensor&>);
static_assert(!CanMatvec<IntegerTensor&, const IntegerTensor&>);
static_assert(!CanMatvec<Tensor&, const DoubleTensor&>);
static_assert(!noexcept(
    std::declval<const Tensor&>().matvec(std::declval<const Tensor&>())));

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

void test_matvec_multiplies_rectangular_matrix_without_changing_operands() {
  const Tensor matrix =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor vector = Tensor::from_data({3}, {2.0F, -1.0F, 0.5F});
  const Tensor::shape_type matrix_shape{2, 3};
  const Tensor::strides_type matrix_strides{3, 1};
  const Tensor::storage_type matrix_elements{1.0F, 2.0F, 3.0F,
                                             4.0F, 5.0F, 6.0F};
  const Tensor::shape_type vector_shape{3};
  const Tensor::strides_type vector_strides{1};
  const Tensor::storage_type vector_elements{2.0F, -1.0F, 0.5F};

  const Tensor result = matrix.matvec(vector);

  expect_tensor(result, {2}, {1}, {1.5F, 6.0F},
                "matvec produced an incorrect result");
  expect_tensor(matrix, matrix_shape, matrix_strides, matrix_elements,
                "matvec changed the matrix operand");
  expect_tensor(vector, vector_shape, vector_strides, vector_elements,
                "matvec changed the vector operand");
}

void test_matvec_supports_singleton_dimensions() {
  const Tensor one_row =
      Tensor::from_data({1, 3}, {1.0F, -2.0F, 4.0F})
          .matvec(Tensor::from_data({3}, {3.0F, 0.5F, 2.0F}));
  const Tensor one_column = Tensor::from_data({3, 1}, {2.0F, -3.0F, 4.0F})
                                .matvec(Tensor::from_data({1}, {2.5F}));

  expect_tensor(one_row, {1}, {1}, {10.0F},
                "matvec produced an incorrect one-row result");
  expect_tensor(one_column, {3}, {1}, {5.0F, -7.5F, 10.0F},
                "matvec produced an incorrect one-column result");
}

void test_matvec_supports_zero_extent_dimensions() {
  const Tensor empty_inner = Tensor({2, 0}).matvec(Tensor({0}));
  const Tensor zero_rows =
      Tensor({0, 3}).matvec(Tensor::from_data({3}, {1.0F, 2.0F, 3.0F}));
  const Tensor fully_empty = Tensor({0, 0}).matvec(Tensor({0}));

  expect_tensor(empty_inner, {2}, {1}, {0.0F, 0.0F},
                "matvec did not use zero for an empty inner dimension");
  expect_tensor(zero_rows, {0}, {1}, {},
                "matvec produced an incorrect zero-row result");
  expect_tensor(fully_empty, {0}, {1}, {},
                "matvec produced an incorrect fully empty result");
}

void test_matvec_uses_native_floating_point_behavior() {
  const Tensor infinity_result =
      Tensor::from_data({1, 1}, {std::numeric_limits<float>::infinity()})
          .matvec(Tensor::from_data({1}, {2.0F}));
  const Tensor nan_result =
      Tensor::from_data({1, 1}, {std::numeric_limits<float>::infinity()})
          .matvec(Tensor::from_data({1}, {0.0F}));

  expect(std::isinf(infinity_result.at(0)) && infinity_result.at(0) > 0.0F,
         "matvec did not preserve native floating-point infinity behavior");
  expect(std::isnan(nan_result.at(0)),
         "matvec did not preserve native floating-point NaN behavior");
}

void test_matvec_accepts_rvalues_without_consuming_operands() {
  Tensor matrix = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  Tensor vector = Tensor::from_data({2}, {5.0F, 6.0F});
  const Tensor::storage_type expected_matrix{1.0F, 2.0F, 3.0F, 4.0F};
  const Tensor::storage_type expected_vector{5.0F, 6.0F};

  const Tensor result = std::move(matrix).matvec(std::move(vector));

  expect_tensor(result, {2}, {1}, {17.0F, 39.0F},
                "rvalue matvec produced an incorrect result");
  expect(std::ranges::equal(matrix.elements(), expected_matrix),
         "matvec consumed its rvalue matrix operand");
  expect(std::ranges::equal(vector.elements(), expected_vector),
         "matvec consumed its rvalue vector operand");
}

void test_matvec_rejects_invalid_operand_ranks() {
  const Tensor valid_matrix = Tensor::from_data({1, 2}, {1.0F, 2.0F});
  const Tensor valid_vector = Tensor::from_data({2}, {3.0F, 4.0F});
  const Tensor scalar = Tensor::scalar(1.0F);
  const Tensor rank_one = Tensor::from_data({2}, {1.0F, 2.0F});
  const Tensor rank_three = Tensor({1, 1, 2});
  const Tensor rank_two = Tensor::from_data({2, 1}, {1.0F, 2.0F});

  expect_throws<std::invalid_argument>(
      [&scalar, &valid_vector] {
        static_cast<void>(scalar.matvec(valid_vector));
      },
      "matvec accepted a rank-zero matrix operand",
      "matvec with a rank-zero matrix produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&rank_one, &valid_vector] {
        static_cast<void>(rank_one.matvec(valid_vector));
      },
      "matvec accepted a rank-one matrix operand",
      "matvec with a rank-one matrix produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&rank_three, &valid_vector] {
        static_cast<void>(rank_three.matvec(valid_vector));
      },
      "matvec accepted a rank-three matrix operand",
      "matvec with a rank-three matrix produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&valid_matrix, &scalar] {
        static_cast<void>(valid_matrix.matvec(scalar));
      },
      "matvec accepted a rank-zero vector operand",
      "matvec with a rank-zero vector produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&valid_matrix, &rank_two] {
        static_cast<void>(valid_matrix.matvec(rank_two));
      },
      "matvec accepted a rank-two vector operand",
      "matvec with a rank-two vector produced the wrong exception type");
}

void test_matvec_rejects_inner_dimension_mismatch_without_changes() {
  const Tensor matrix = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  const Tensor vector = Tensor::from_data({3}, {5.0F, 6.0F, 7.0F});
  const Tensor zero_row_matrix = Tensor({0, 2});
  const Tensor::storage_type expected_matrix{1.0F, 2.0F, 3.0F, 4.0F};
  const Tensor::storage_type expected_vector{5.0F, 6.0F, 7.0F};

  expect_throws<std::invalid_argument>(
      [&matrix, &vector] { static_cast<void>(matrix.matvec(vector)); },
      "matvec accepted incompatible inner dimensions",
      "matvec dimension mismatch produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&zero_row_matrix, &vector] {
        static_cast<void>(zero_row_matrix.matvec(vector));
      },
      "zero-row matvec accepted incompatible inner dimensions",
      "zero-row matvec mismatch produced the wrong exception type");
  expect(std::ranges::equal(matrix.elements(), expected_matrix),
         "failed matvec changed the matrix operand");
  expect(std::ranges::equal(vector.elements(), expected_vector),
         "failed matvec changed the vector operand");
}

void test_matvec_rejects_moved_from_sentinel_on_each_side() {
  Tensor matrix_source = Tensor::from_data({1, 2}, {1.0F, 2.0F});
  Tensor matrix_owner(std::move(matrix_source));
  Tensor vector_source = Tensor::from_data({2}, {3.0F, 4.0F});
  Tensor vector_owner(std::move(vector_source));
  const Tensor valid_matrix = Tensor::from_data({1, 2}, {5.0F, 6.0F});
  const Tensor valid_vector = Tensor::from_data({2}, {7.0F, 8.0F});
  static_cast<void>(matrix_owner);
  static_cast<void>(vector_owner);

  expect_throws<std::invalid_argument>(
      [&matrix_source, &valid_vector] {
        static_cast<void>(matrix_source.matvec(valid_vector));
      },
      "matvec accepted a moved-from matrix operand",
      "matvec with a moved-from matrix produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&vector_source, &valid_matrix] {
        static_cast<void>(valid_matrix.matvec(vector_source));
      },
      "matvec accepted a moved-from vector operand",
      "matvec with a moved-from vector produced the wrong exception type");
  expect_empty_sentinel(matrix_source);
  expect_empty_sentinel(vector_source);
}

}  // namespace

int main() {
  try {
    test_matvec_multiplies_rectangular_matrix_without_changing_operands();
    test_matvec_supports_singleton_dimensions();
    test_matvec_supports_zero_extent_dimensions();
    test_matvec_uses_native_floating_point_behavior();
    test_matvec_accepts_rvalues_without_consuming_operands();
    test_matvec_rejects_invalid_operand_ranks();
    test_matvec_rejects_inner_dimension_mismatch_without_changes();
    test_matvec_rejects_moved_from_sentinel_on_each_side();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor matvec tests passed\n";
  return EXIT_SUCCESS;
}
