#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

#include "nn/tensor.hpp"

namespace {

using Tensor = nn::Tensor<float>;

struct MatmulCase {
  Tensor::shape_type left_shape;
  Tensor::shape_type right_shape;
  Tensor::shape_type expected_shape;
};

void expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

Tensor::size_type compute_element_count(
    std::span<const Tensor::size_type> shape) {
  Tensor::size_type element_count = 1;

  for (const Tensor::size_type extent : shape) {
    element_count *= extent;
  }

  return element_count;
}

Tensor::strides_type compute_row_major_strides(
    std::span<const Tensor::size_type> shape) {
  Tensor::strides_type strides(shape.size());
  Tensor::size_type running_stride = 1;

  for (Tensor::size_type remaining_axes = shape.size(); remaining_axes != 0;
       --remaining_axes) {
    const Tensor::size_type axis = remaining_axes - 1;

    strides[axis] = running_stride;
    running_stride *= shape[axis];
  }

  return strides;
}

Tensor::storage_type make_elements(const Tensor::shape_type& shape,
                                   Tensor::size_type seed) {
  constexpr std::array<float, 7> pattern{-3.0F, -2.0F, -1.0F, 0.0F,
                                         1.0F,  2.0F,  3.0F};
  const Tensor::size_type element_count = compute_element_count(shape);

  Tensor::storage_type elements;
  elements.reserve(element_count);

  for (Tensor::size_type offset = 0; offset < element_count; ++offset) {
    elements.push_back(pattern[(offset + seed) % pattern.size()]);
  }

  return elements;
}

void decode_row_major_offset(Tensor::size_type offset,
                             std::span<const Tensor::size_type> shape,
                             std::span<Tensor::size_type> indices) {
  for (Tensor::size_type remaining_axes = shape.size(); remaining_axes != 0;
       --remaining_axes) {
    const Tensor::size_type axis = remaining_axes - 1;

    indices[axis] = offset % shape[axis];
    offset /= shape[axis];
  }
}

void map_batch_indices(const Tensor& operand,
                       std::span<const Tensor::size_type> result_indices,
                       Tensor::size_type result_batch_rank,
                       std::span<Tensor::size_type> operand_indices) {
  const Tensor::size_type operand_batch_rank = operand.rank() - 2;
  const Tensor::size_type rank_difference =
      result_batch_rank - operand_batch_rank;

  for (Tensor::size_type operand_axis = 0; operand_axis < operand_batch_rank;
       ++operand_axis) {
    const Tensor::size_type result_axis = rank_difference + operand_axis;

    operand_indices[operand_axis] = operand.shape()[operand_axis] == 1
                                        ? Tensor::size_type{0}
                                        : result_indices[result_axis];
  }
}

Tensor::storage_type reference_matmul(
    const Tensor& left, const Tensor& right,
    const Tensor::shape_type& expected_shape) {
  const Tensor::size_type result_element_count =
      compute_element_count(expected_shape);
  Tensor::storage_type expected_elements(result_element_count, 0.0F);

  if (result_element_count == 0) {
    return expected_elements;
  }

  const Tensor::size_type result_batch_rank = expected_shape.size() - 2;
  const Tensor::size_type left_row_axis = left.rank() - 2;
  const Tensor::size_type right_row_axis = right.rank() - 2;
  const Tensor::size_type inner_extent = left.shape()[left_row_axis + 1];

  std::vector<Tensor::size_type> result_indices(expected_shape.size());
  std::vector<Tensor::size_type> left_indices(left.rank());
  std::vector<Tensor::size_type> right_indices(right.rank());

  for (Tensor::size_type result_offset = 0;
       result_offset < result_element_count; ++result_offset) {
    decode_row_major_offset(result_offset, expected_shape, result_indices);
    map_batch_indices(left, result_indices, result_batch_rank, left_indices);
    map_batch_indices(right, result_indices, result_batch_rank, right_indices);

    left_indices[left_row_axis] = result_indices[result_batch_rank];
    right_indices[right_row_axis + 1] = result_indices[result_batch_rank + 1];

    for (Tensor::size_type inner_index = 0; inner_index < inner_extent;
         ++inner_index) {
      left_indices[left_row_axis + 1] = inner_index;
      right_indices[right_row_axis] = inner_index;

      expected_elements[result_offset] +=
          left.at(std::span<const Tensor::size_type>{left_indices}) *
          right.at(std::span<const Tensor::size_type>{right_indices});
    }
  }

  return expected_elements;
}

void test_matmul_matches_independent_reference() {
  const std::vector<MatmulCase> test_cases{
      {{2, 3}, {3, 4}, {2, 4}},
      {{2, 2, 3}, {2, 3, 4}, {2, 2, 4}},
      {{2, 3}, {4, 3, 2}, {4, 2, 2}},
      {{2, 3}, {2, 4, 3, 2}, {2, 4, 2, 2}},
      {{3, 2, 3}, {3, 4}, {3, 2, 4}},
      {{2, 1, 2, 3}, {3, 3, 2}, {2, 3, 2, 2}},
      {{1, 4, 1, 3}, {2, 1, 3, 2}, {2, 4, 1, 2}},
      {{2, 1, 4, 1, 1}, {1, 3, 1, 1, 1}, {2, 3, 4, 1, 1}},
      {{3, 1, 2}, {1, 2, 1}, {3, 1, 1}},
      {{2, 2, 0}, {1, 0, 3}, {2, 2, 3}},
      {{0, 2, 3}, {1, 3, 4}, {0, 2, 4}},
      {{1, 0, 2, 3}, {4, 1, 3, 2}, {4, 0, 2, 2}},
  };

  for (Tensor::size_type case_index = 0; case_index < test_cases.size();
       ++case_index) {
    const MatmulCase& test_case = test_cases[case_index];

    for (Tensor::size_type seed = 0; seed < 5; ++seed) {
      const Tensor left = Tensor::from_data(
          test_case.left_shape, make_elements(test_case.left_shape, seed));
      const Tensor right = Tensor::from_data(
          test_case.right_shape,
          make_elements(test_case.right_shape, case_index + seed + 1));
      const Tensor result = left.matmul(right);
      const Tensor::storage_type expected_elements =
          reference_matmul(left, right, test_case.expected_shape);
      const Tensor::strides_type expected_strides =
          compute_row_major_strides(test_case.expected_shape);

      expect(std::ranges::equal(result.shape(), test_case.expected_shape),
             "matmul property test produced an incorrect shape");
      expect(std::ranges::equal(result.strides(), expected_strides),
             "matmul property test produced incorrect strides");
      expect(std::ranges::equal(result.elements(), expected_elements),
             "matmul did not match the independent reference");
    }
  }
}

}  // namespace

int main() {
  try {
    test_matmul_matches_independent_reference();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor matmul property tests passed\n";
  return EXIT_SUCCESS;
}
