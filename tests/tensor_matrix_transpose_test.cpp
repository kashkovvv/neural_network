#include <algorithm>
#include <concepts>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "nn/tensor.hpp"

namespace {

using Tensor = nn::Tensor<float>;

struct CopyConstructibleOnly {
  explicit CopyConstructibleOnly(int initial_value) : value(initial_value) {}

  CopyConstructibleOnly(const CopyConstructibleOnly&) = default;
  CopyConstructibleOnly(CopyConstructibleOnly&&) noexcept = default;
  CopyConstructibleOnly& operator=(const CopyConstructibleOnly&) = delete;
  CopyConstructibleOnly& operator=(CopyConstructibleOnly&&) = delete;

  int value;
};

struct MoveOnly {
  MoveOnly() = default;
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly(MoveOnly&&) = default;
  MoveOnly& operator=(const MoveOnly&) = delete;
  MoveOnly& operator=(MoveOnly&&) = default;
};

template <typename TensorType>
concept CanMatrixTranspose = requires(TensorType&& tensor) {
  std::forward<TensorType>(tensor).matrix_transpose();
};

static_assert(std::same_as<
              decltype(std::declval<const Tensor&>().matrix_transpose()),
              Tensor>);
static_assert(CanMatrixTranspose<Tensor&>);
static_assert(CanMatrixTranspose<const Tensor&>);
static_assert(CanMatrixTranspose<Tensor&&>);
static_assert(CanMatrixTranspose<const Tensor&&>);
static_assert(CanMatrixTranspose<nn::Tensor<int>&>);
static_assert(CanMatrixTranspose<nn::Tensor<CopyConstructibleOnly>&>);
static_assert(!CanMatrixTranspose<nn::Tensor<MoveOnly>&>);
static_assert(
    !noexcept(std::declval<const Tensor&>().matrix_transpose()));

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

void test_matrix_transpose_transposes_rank_two_tensor() {
  const Tensor tensor =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});

  const Tensor result = tensor.matrix_transpose();

  const Tensor::shape_type expected_shape{3, 2};
  const Tensor::strides_type expected_strides{2, 1};
  const Tensor::storage_type expected_elements{1.0F, 4.0F, 2.0F,
                                                5.0F, 3.0F, 6.0F};

  expect(std::ranges::equal(result.shape(), expected_shape),
         "matrix transpose produced an incorrect shape");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "matrix transpose produced incorrect row-major strides");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "matrix transpose produced an incorrect element order");
  expect(result[2, 1] == tensor[1, 2],
         "matrix transpose produced incorrect index mapping");
}

void test_matrix_transpose_transposes_batched_matrices() {
  const Tensor tensor = Tensor::from_data(
      {2, 2, 3},
      {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F,
       11.0F, 12.0F});

  const Tensor result = tensor.matrix_transpose();

  const Tensor::shape_type expected_shape{2, 3, 2};
  const Tensor::strides_type expected_strides{6, 2, 1};
  const Tensor::storage_type expected_elements{1.0F, 4.0F,  2.0F, 5.0F,
                                                3.0F, 6.0F,  7.0F, 10.0F,
                                                8.0F, 11.0F, 9.0F, 12.0F};

  expect(std::ranges::equal(result.shape(), expected_shape),
         "batched matrix transpose produced an incorrect shape");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "batched matrix transpose produced incorrect strides");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "batched matrix transpose produced an incorrect element order");
  expect(result[1, 2, 1] == tensor[1, 1, 2],
         "batched matrix transpose changed a batch axis");
}

void test_matrix_transpose_returns_independent_tensor() {
  Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});

  Tensor result = tensor.matrix_transpose();

  expect(result.elements().data() != tensor.elements().data(),
         "matrix transpose shares storage with its source");

  result[0, 1] = -3.0F;

  expect(tensor[1, 0] == 3.0F,
         "mutation of matrix transpose result changed its source");
}

void test_matrix_transpose_singleton_matrix_axes() {
  const Tensor tensor =
      Tensor::from_data({2, 1, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});

  const Tensor result = tensor.matrix_transpose();

  const Tensor::shape_type expected_shape{2, 3, 1};
  const Tensor::strides_type expected_strides{3, 1, 1};
  const Tensor::storage_type expected_elements{1.0F, 2.0F, 3.0F,
                                                4.0F, 5.0F, 6.0F};

  expect(std::ranges::equal(result.shape(), expected_shape),
         "singleton matrix transpose produced an incorrect shape");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "singleton matrix transpose produced incorrect strides");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "singleton matrix transpose changed the element order");
}

void test_matrix_transpose_zero_extent_matrix_axes() {
  const Tensor zero_rows({2, 0, 3});

  const Tensor transposed_zero_rows = zero_rows.matrix_transpose();

  const Tensor::shape_type zero_columns_shape{2, 3, 0};
  const Tensor::strides_type zero_columns_strides{0, 0, 1};

  expect(std::ranges::equal(transposed_zero_rows.shape(), zero_columns_shape),
         "zero-row matrix transpose produced an incorrect shape");
  expect(std::ranges::equal(transposed_zero_rows.strides(),
                            zero_columns_strides),
         "zero-row matrix transpose produced incorrect strides");
  expect(transposed_zero_rows.elements().empty(),
         "zero-row matrix transpose produced nonempty storage");

  const Tensor zero_columns({2, 3, 0});

  const Tensor transposed_zero_columns = zero_columns.matrix_transpose();

  const Tensor::shape_type zero_rows_shape{2, 0, 3};
  const Tensor::strides_type zero_rows_strides{0, 3, 1};

  expect(std::ranges::equal(transposed_zero_columns.shape(), zero_rows_shape),
         "zero-column matrix transpose produced an incorrect shape");
  expect(std::ranges::equal(transposed_zero_columns.strides(),
                            zero_rows_strides),
         "zero-column matrix transpose produced incorrect strides");
  expect(transposed_zero_columns.elements().empty(),
         "zero-column matrix transpose produced nonempty storage");
}

void test_matrix_transpose_copy_constructible_nonassignable_elements() {
  using RestrictedTensor = nn::Tensor<CopyConstructibleOnly>;

  RestrictedTensor::storage_type data;
  data.emplace_back(1);
  data.emplace_back(2);
  data.emplace_back(3);
  data.emplace_back(4);
  data.emplace_back(5);
  data.emplace_back(6);
  const RestrictedTensor tensor =
      RestrictedTensor::from_data({2, 3}, std::move(data));

  const RestrictedTensor result = tensor.matrix_transpose();

  expect(result.elements()[0].value == 1,
         "restricted matrix transpose changed the first element");
  expect(result.elements()[1].value == 4,
         "restricted matrix transpose produced an incorrect second element");
  expect(result.elements()[2].value == 2,
         "restricted matrix transpose produced an incorrect third element");
  expect(result.elements()[3].value == 5,
         "restricted matrix transpose produced an incorrect fourth element");
  expect(result.elements()[4].value == 3,
         "restricted matrix transpose produced an incorrect fifth element");
  expect(result.elements()[5].value == 6,
         "restricted matrix transpose changed the final element");
}

void test_matrix_transpose_rejects_scalar() {
  const Tensor tensor = Tensor::scalar(7.0F);

  expect_throws<std::invalid_argument>(
      [&tensor] { static_cast<void>(tensor.matrix_transpose()); },
      "scalar matrix transpose did not throw",
      "scalar matrix transpose produced the wrong exception type");

  expect(tensor.rank() == 0, "failed scalar matrix transpose changed rank");
  expect(tensor.numel() == 1,
         "failed scalar matrix transpose changed element count");
  expect(tensor.at() == 7.0F,
         "failed scalar matrix transpose changed the element value");
}

void test_matrix_transpose_rejects_rank_one_tensor() {
  const Tensor tensor =
      Tensor::from_data({3}, {1.0F, 2.0F, 3.0F});

  expect_throws<std::invalid_argument>(
      [&tensor] { static_cast<void>(tensor.matrix_transpose()); },
      "rank-1 matrix transpose did not throw",
      "rank-1 matrix transpose produced the wrong exception type");

  const Tensor::shape_type expected_shape{3};
  const Tensor::strides_type expected_strides{1};
  const Tensor::storage_type expected_elements{1.0F, 2.0F, 3.0F};

  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "failed rank-1 matrix transpose changed shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "failed rank-1 matrix transpose changed strides");
  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "failed rank-1 matrix transpose changed elements");
}

void test_matrix_transpose_rejects_moved_from_sentinel() {
  Tensor source = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  Tensor destination(std::move(source));
  static_cast<void>(destination);

  expect_throws<std::invalid_argument>(
      [&source] { static_cast<void>(source.matrix_transpose()); },
      "matrix transpose of a moved-from sentinel did not throw",
      "matrix transpose of a moved-from sentinel produced the wrong exception "
      "type");

  expect(source.rank() == 0,
         "failed sentinel matrix transpose changed rank");
  expect(source.numel() == 0,
         "failed sentinel matrix transpose changed element count");
  expect(source.shape().empty(),
         "failed sentinel matrix transpose changed shape");
  expect(source.strides().empty(),
         "failed sentinel matrix transpose changed strides");
  expect(source.elements().empty(),
         "failed sentinel matrix transpose changed elements");
}

class CopyFailure final : public std::exception {};

struct ThrowingValue {
  ThrowingValue() = default;
  explicit ThrowingValue(int initial_value) : value(initial_value) {}

  ThrowingValue(const ThrowingValue& other) : value(other.value) {
    if (copy_failure_enabled) {
      throw CopyFailure{};
    }
  }

  ThrowingValue(ThrowingValue&&) noexcept = default;

  ThrowingValue& operator=(const ThrowingValue& other) {
    if (copy_failure_enabled) {
      throw CopyFailure{};
    }

    value = other.value;
    return *this;
  }

  ThrowingValue& operator=(ThrowingValue&&) noexcept = default;

  int value{};
  inline static bool copy_failure_enabled{};
};

void test_element_copy_failure_preserves_source() {
  using ThrowingTensor = nn::Tensor<ThrowingValue>;

  ThrowingTensor::storage_type data;
  data.emplace_back(1);
  data.emplace_back(2);
  data.emplace_back(3);
  data.emplace_back(4);
  const ThrowingTensor tensor =
      ThrowingTensor::from_data({2, 2}, std::move(data));

  ThrowingValue::copy_failure_enabled = true;

  expect_throws<CopyFailure>(
      [&tensor] { static_cast<void>(tensor.matrix_transpose()); },
      "element copy failure did not propagate from matrix transpose",
      "element copy failure produced the wrong exception type");

  ThrowingValue::copy_failure_enabled = false;

  const ThrowingTensor::shape_type expected_shape{2, 2};
  const ThrowingTensor::strides_type expected_strides{2, 1};

  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "element copy failure changed source shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "element copy failure changed source strides");
  expect(tensor.elements()[0].value == 1,
         "element copy failure changed the first source element");
  expect(tensor.elements()[1].value == 2,
         "element copy failure changed the second source element");
  expect(tensor.elements()[2].value == 3,
         "element copy failure changed the third source element");
  expect(tensor.elements()[3].value == 4,
         "element copy failure changed the fourth source element");
}

}  // namespace

int main() {
  try {
    test_matrix_transpose_transposes_rank_two_tensor();
    test_matrix_transpose_transposes_batched_matrices();
    test_matrix_transpose_returns_independent_tensor();
    test_matrix_transpose_singleton_matrix_axes();
    test_matrix_transpose_zero_extent_matrix_axes();
    test_matrix_transpose_copy_constructible_nonassignable_elements();
    test_matrix_transpose_rejects_scalar();
    test_matrix_transpose_rejects_rank_one_tensor();
    test_matrix_transpose_rejects_moved_from_sentinel();
    test_element_copy_failure_preserves_source();
  } catch (const std::exception& exception) {
    ThrowingValue::copy_failure_enabled = false;
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor matrix transpose tests passed\n";
  return EXIT_SUCCESS;
}
