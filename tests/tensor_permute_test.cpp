#include <algorithm>
#include <concepts>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

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
concept CanPermute = requires(TensorType&& tensor) {
  std::forward<TensorType>(tensor).permute(
      typename std::remove_cvref_t<TensorType>::axes_type{});
};

static_assert(
    std::same_as<Tensor::axes_type, std::vector<Tensor::size_type>>);
static_assert(std::same_as<
              decltype(std::declval<const Tensor&>().permute(
                  std::declval<const Tensor::axes_type&>())),
              Tensor>);
static_assert(CanPermute<Tensor&>);
static_assert(CanPermute<const Tensor&>);
static_assert(CanPermute<Tensor&&>);
static_assert(CanPermute<const Tensor&&>);
static_assert(CanPermute<nn::Tensor<int>&>);
static_assert(CanPermute<nn::Tensor<CopyConstructibleOnly>&>);
static_assert(!CanPermute<nn::Tensor<MoveOnly>&>);
static_assert(!noexcept(std::declval<const Tensor&>().permute(
    std::declval<const Tensor::axes_type&>())));

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

void test_permute_reorders_rank_three_tensor() {
  const Tensor tensor = Tensor::from_data(
      {2, 3, 2},
      {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F,
       10.0F, 11.0F});

  const Tensor result = tensor.permute({1, 2, 0});

  const Tensor::shape_type expected_shape{3, 2, 2};
  const Tensor::strides_type expected_strides{4, 2, 1};
  const Tensor::storage_type expected_elements{0.0F, 6.0F,  1.0F, 7.0F,
                                                2.0F, 8.0F,  3.0F, 9.0F,
                                                4.0F, 10.0F, 5.0F, 11.0F};

  expect(std::ranges::equal(result.shape(), expected_shape),
         "rank-3 permutation produced an incorrect shape");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "rank-3 permutation produced incorrect row-major strides");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "rank-3 permutation produced an incorrect element order");
  expect(result[2, 1, 1] == tensor[1, 2, 1],
         "rank-3 permutation produced incorrect index mapping");
}

void test_permute_transposes_matrix() {
  const Tensor tensor =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});

  const Tensor result = tensor.permute({1, 0});

  const Tensor::shape_type expected_shape{3, 2};
  const Tensor::strides_type expected_strides{2, 1};
  const Tensor::storage_type expected_elements{1.0F, 4.0F, 2.0F,
                                                5.0F, 3.0F, 6.0F};

  expect(std::ranges::equal(result.shape(), expected_shape),
         "matrix permutation produced an incorrect shape");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "matrix permutation produced incorrect strides");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "matrix permutation produced an incorrect element order");
}

void test_identity_permutation_returns_independent_tensor() {
  Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});

  Tensor result = tensor.permute({0, 1});

  const Tensor::shape_type expected_shape{2, 2};
  const Tensor::strides_type expected_strides{2, 1};
  const Tensor::storage_type expected_elements{1.0F, 2.0F, 3.0F, 4.0F};

  expect(std::ranges::equal(result.shape(), expected_shape),
         "identity permutation changed shape");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "identity permutation changed strides");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "identity permutation changed elements");
  expect(result.elements().data() != tensor.elements().data(),
         "identity permutation shares storage with its source");

  result[0, 0] = -1.0F;

  expect(tensor[0, 0] == 1.0F,
         "mutation of permutation result changed its source");
}

void test_permute_scalar() {
  const Tensor tensor = Tensor::scalar(7.0F);

  const Tensor result = tensor.permute({});

  expect(result.rank() == 0, "scalar permutation changed rank");
  expect(result.numel() == 1, "scalar permutation changed element count");
  expect(result.shape().empty(), "scalar permutation produced a shape");
  expect(result.strides().empty(), "scalar permutation produced strides");
  expect(result.at() == 7.0F, "scalar permutation changed the element value");
}

void test_permute_rank_one_tensor() {
  const Tensor result =
      Tensor::from_data({3}, {1.0F, 2.0F, 3.0F}).permute({0});
  const Tensor::shape_type expected_shape{3};
  const Tensor::strides_type expected_strides{1};
  const Tensor::storage_type expected_elements{1.0F, 2.0F, 3.0F};

  expect(std::ranges::equal(result.shape(), expected_shape),
         "rank-1 permutation changed shape");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "rank-1 permutation changed strides");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "rank-1 permutation changed elements");
}

void test_permute_singleton_axes() {
  const Tensor tensor =
      Tensor::from_data({2, 1, 2}, {1.0F, 2.0F, 3.0F, 4.0F});

  const Tensor result = tensor.permute({1, 2, 0});

  const Tensor::shape_type expected_shape{1, 2, 2};
  const Tensor::strides_type expected_strides{4, 2, 1};
  const Tensor::storage_type expected_elements{1.0F, 3.0F, 2.0F, 4.0F};

  expect(std::ranges::equal(result.shape(), expected_shape),
         "singleton-axis permutation produced an incorrect shape");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "singleton-axis permutation produced incorrect strides");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "singleton-axis permutation produced an incorrect element order");
}

void test_permute_zero_extent_tensor() {
  const Tensor tensor({2, 0, 3});

  const Tensor result = tensor.permute({2, 0, 1});

  const Tensor::shape_type expected_shape{3, 2, 0};
  const Tensor::strides_type expected_strides{0, 0, 1};

  expect(result.numel() == 0,
         "zero-extent permutation changed element count");
  expect(std::ranges::equal(result.shape(), expected_shape),
         "zero-extent permutation produced an incorrect shape");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "zero-extent permutation produced incorrect strides");
  expect(result.elements().empty(),
         "zero-extent permutation produced nonempty storage");
}

void test_permute_copy_constructible_nonassignable_elements() {
  using RestrictedTensor = nn::Tensor<CopyConstructibleOnly>;

  RestrictedTensor::storage_type data;
  data.emplace_back(1);
  data.emplace_back(2);
  data.emplace_back(3);
  data.emplace_back(4);
  const RestrictedTensor tensor =
      RestrictedTensor::from_data({2, 2}, std::move(data));

  const RestrictedTensor result = tensor.permute({1, 0});

  expect(result.elements()[0].value == 1,
         "restricted permutation changed the first element");
  expect(result.elements()[1].value == 3,
         "restricted permutation produced an incorrect second element");
  expect(result.elements()[2].value == 2,
         "restricted permutation produced an incorrect third element");
  expect(result.elements()[3].value == 4,
         "restricted permutation changed the fourth element");
}

void expect_invalid_permutation_preserves_source(
    const Tensor::axes_type& axes, const char* missing_exception_message,
    const char* wrong_exception_message) {
  const Tensor tensor =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});

  expect_throws<std::invalid_argument>(
      [&tensor, &axes] { static_cast<void>(tensor.permute(axes)); },
      missing_exception_message, wrong_exception_message);

  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_elements{1.0F, 2.0F, 3.0F,
                                                4.0F, 5.0F, 6.0F};

  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "invalid permutation changed source shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "invalid permutation changed source strides");
  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "invalid permutation changed source elements");
}

void test_permute_rejects_too_few_axes() {
  expect_invalid_permutation_preserves_source(
      {0}, "permutation with too few axes did not throw",
      "permutation with too few axes produced the wrong exception type");
}

void test_permute_rejects_too_many_axes() {
  expect_invalid_permutation_preserves_source(
      {0, 1, 2}, "permutation with too many axes did not throw",
      "permutation with too many axes produced the wrong exception type");
}

void test_permute_rejects_duplicate_axis() {
  expect_invalid_permutation_preserves_source(
      {0, 0}, "permutation with a duplicate axis did not throw",
      "permutation with a duplicate axis produced the wrong exception type");
}

void test_permute_rejects_out_of_range_axis() {
  const Tensor tensor =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});

  expect_throws<std::out_of_range>(
      [&tensor] { static_cast<void>(tensor.permute({0, 2})); },
      "permutation with an out-of-range axis did not throw",
      "permutation with an out-of-range axis produced the wrong exception "
      "type");

  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_elements{1.0F, 2.0F, 3.0F,
                                                4.0F, 5.0F, 6.0F};

  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "out-of-range permutation changed source shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "out-of-range permutation changed source strides");
  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "out-of-range permutation changed source elements");
}

void test_permute_rejects_moved_from_sentinel() {
  Tensor source = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor destination(std::move(source));
  static_cast<void>(destination);

  expect_throws<std::invalid_argument>(
      [&source] { static_cast<void>(source.permute({})); },
      "permutation of a moved-from sentinel did not throw",
      "permutation of a moved-from sentinel produced the wrong exception type");

  expect(source.rank() == 0, "failed sentinel permutation changed rank");
  expect(source.numel() == 0,
         "failed sentinel permutation changed element count");
  expect(source.shape().empty(),
         "failed sentinel permutation changed source shape");
  expect(source.strides().empty(),
         "failed sentinel permutation changed source strides");
  expect(source.elements().empty(),
         "failed sentinel permutation changed source elements");
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
      [&tensor] { static_cast<void>(tensor.permute({1, 0})); },
      "element copy failure did not propagate from permutation",
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
    test_permute_reorders_rank_three_tensor();
    test_permute_transposes_matrix();
    test_identity_permutation_returns_independent_tensor();
    test_permute_scalar();
    test_permute_rank_one_tensor();
    test_permute_singleton_axes();
    test_permute_zero_extent_tensor();
    test_permute_copy_constructible_nonassignable_elements();
    test_permute_rejects_too_few_axes();
    test_permute_rejects_too_many_axes();
    test_permute_rejects_duplicate_axis();
    test_permute_rejects_out_of_range_axis();
    test_permute_rejects_moved_from_sentinel();
    test_element_copy_failure_preserves_source();
  } catch (const std::exception& exception) {
    ThrowingValue::copy_failure_enabled = false;
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor permutation tests passed\n";
  return EXIT_SUCCESS;
}
