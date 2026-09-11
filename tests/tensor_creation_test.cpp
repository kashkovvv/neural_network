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

struct DefaultConstructibleNonAssignable {
  int value{};

  DefaultConstructibleNonAssignable() = default;
  DefaultConstructibleNonAssignable(
      const DefaultConstructibleNonAssignable&) = delete;
  DefaultConstructibleNonAssignable(DefaultConstructibleNonAssignable&&) =
      default;
  DefaultConstructibleNonAssignable& operator=(
      const DefaultConstructibleNonAssignable&) = delete;
  DefaultConstructibleNonAssignable& operator=(
      DefaultConstructibleNonAssignable&&) = delete;
};

struct CopyConstructibleNonDefaultNonAssignable {
  explicit CopyConstructibleNonDefaultNonAssignable(int initial_value)
      : value(initial_value) {}

  CopyConstructibleNonDefaultNonAssignable() = delete;
  CopyConstructibleNonDefaultNonAssignable(
      const CopyConstructibleNonDefaultNonAssignable&) = default;
  CopyConstructibleNonDefaultNonAssignable(
      CopyConstructibleNonDefaultNonAssignable&&) = default;
  CopyConstructibleNonDefaultNonAssignable& operator=(
      const CopyConstructibleNonDefaultNonAssignable&) = delete;
  CopyConstructibleNonDefaultNonAssignable& operator=(
      CopyConstructibleNonDefaultNonAssignable&&) = delete;

  int value;
};

struct MoveOnlyNonDefaultNonAssignable {
  explicit MoveOnlyNonDefaultNonAssignable(int initial_value)
      : value(initial_value) {}

  MoveOnlyNonDefaultNonAssignable() = delete;
  MoveOnlyNonDefaultNonAssignable(const MoveOnlyNonDefaultNonAssignable&) =
      delete;
  MoveOnlyNonDefaultNonAssignable(MoveOnlyNonDefaultNonAssignable&&) noexcept =
      default;
  MoveOnlyNonDefaultNonAssignable& operator=(
      const MoveOnlyNonDefaultNonAssignable&) = delete;
  MoveOnlyNonDefaultNonAssignable& operator=(
      MoveOnlyNonDefaultNonAssignable&&) = delete;

  int value;
};

using DefaultConstructibleTensor =
    nn::Tensor<DefaultConstructibleNonAssignable>;
using CopyConstructibleTensor =
    nn::Tensor<CopyConstructibleNonDefaultNonAssignable>;
using MoveOnlyTensor = nn::Tensor<MoveOnlyNonDefaultNonAssignable>;

static_assert(std::same_as<
              decltype(Tensor::full(Tensor::shape_type{2, 3}, 4.0f)), Tensor>);
static_assert(
    std::same_as<decltype(Tensor::zeros(Tensor::shape_type{2, 3})), Tensor>);
static_assert(std::same_as<decltype(Tensor::scalar(-3.5f)), Tensor>);
static_assert(
    std::same_as<decltype(Tensor::from_data(Tensor::shape_type{},
                                            Tensor::storage_type{0.0f})),
                 Tensor>);

void expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_full_tensor_creation() {
  const Tensor tensor = Tensor::full({2, 3}, 4.0f);

  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};

  expect(tensor.rank() == 2, "full 2x3 tensor rank must be 2");
  expect(tensor.numel() == 6, "full 2x3 tensor numel must be 6");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "full tensor shape is incorrect");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "full tensor strides are incorrect");

  for (Tensor::value_type value : tensor.elements()) {
    expect(value == 4.0f, "full tensor element does not equal the fill value");
  }
}

void test_full_scalar_creation() {
  const Tensor tensor = Tensor::full({}, -2.0f);

  expect(tensor.rank() == 0, "full scalar rank must be 0");
  expect(tensor.numel() == 1, "full scalar numel must be 1");
  expect(tensor.shape().empty(), "full scalar shape must be empty");
  expect(tensor.strides().empty(), "full scalar strides must be empty");
  expect(tensor.elements()[0] == -2.0f,
         "full scalar element does not equal the fill value");
}

void test_full_zero_extent_creation() {
  const Tensor tensor = Tensor::full({2, 0, 4}, 7.0f);

  const Tensor::shape_type expected_shape{2, 0, 4};
  const Tensor::strides_type expected_strides{0, 4, 1};

  expect(tensor.rank() == 3, "full zero-extent tensor rank must be 3");
  expect(tensor.numel() == 0, "full zero-extent tensor numel must be 0");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "full zero-extent tensor shape is incorrect");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "full zero-extent tensor strides are incorrect");
  expect(tensor.elements().empty(),
         "full zero-extent tensor elements must be empty");
}

void test_full_preserves_negative_zero() {
  const Tensor tensor = Tensor::full({2}, -0.0F);

  for (const Tensor::value_type value : tensor.elements()) {
    expect(value == 0.0F, "full negative-zero element is not zero");
    expect(std::signbit(value), "full did not preserve negative-zero sign");
  }
}

void test_full_does_not_require_default_construction_or_assignment() {
  const CopyConstructibleNonDefaultNonAssignable value(7);
  const CopyConstructibleTensor tensor =
      CopyConstructibleTensor::full({2, 2}, value);

  expect(tensor.rank() == 2, "generic full tensor rank must be 2");
  expect(tensor.numel() == 4, "generic full tensor numel must be 4");
  expect(std::ranges::equal(tensor.shape(),
                            CopyConstructibleTensor::shape_type{2, 2}),
         "generic full tensor shape is incorrect");
  expect(std::ranges::equal(tensor.strides(),
                            CopyConstructibleTensor::strides_type{2, 1}),
         "generic full tensor strides are incorrect");

  for (const auto& element : tensor.elements()) {
    expect(element.value == 7, "generic full tensor element is incorrect");
  }
}

void test_full_shape_product_overflow() {
  const Tensor::size_type max_size =
      std::numeric_limits<Tensor::size_type>::max();

  try {
    static_cast<void>(Tensor::full({max_size, 2}, 1.0f));
  } catch (const std::overflow_error&) {
    return;
  } catch (...) {
    throw std::runtime_error(
        "full shape overflow produced the wrong exception type");
  }

  throw std::runtime_error(
      "full shape overflow did not throw std::overflow_error");
}

void test_zeros_tensor_creation() {
  const Tensor tensor = Tensor::zeros({2, 3});

  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};

  expect(tensor.rank() == 2, "zeros 2x3 tensor rank must be 2");
  expect(tensor.numel() == 6, "zeros 2x3 tensor numel must be 6");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "zeros tensor shape is incorrect");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "zeros tensor strides are incorrect");

  for (Tensor::value_type value : tensor.elements()) {
    expect(value == 0.0f, "zeros tensor element is not zero");
  }
}

void test_zeros_scalar_creation() {
  const Tensor tensor = Tensor::zeros({});

  expect(tensor.rank() == 0, "zeros scalar rank must be 0");
  expect(tensor.numel() == 1, "zeros scalar numel must be 1");
  expect(tensor.shape().empty(), "zeros scalar shape must be empty");
  expect(tensor.strides().empty(), "zeros scalar strides must be empty");
  expect(tensor.elements()[0] == 0.0f, "zeros scalar element is not zero");
}

void test_zeros_zero_extent_creation() {
  const Tensor tensor = Tensor::zeros({2, 0, 4});

  const Tensor::shape_type expected_shape{2, 0, 4};
  const Tensor::strides_type expected_strides{0, 4, 1};

  expect(tensor.rank() == 3, "zeros zero-extent tensor rank must be 3");
  expect(tensor.numel() == 0, "zeros zero-extent tensor numel must be 0");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "zeros zero-extent tensor shape is incorrect");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "zeros zero-extent tensor strides are incorrect");
  expect(tensor.elements().empty(),
         "zeros zero-extent tensor elements must be empty");
}

void test_zeros_does_not_require_assignment() {
  const DefaultConstructibleTensor tensor =
      DefaultConstructibleTensor::zeros({2, 2});

  expect(tensor.rank() == 2, "generic zeros tensor rank must be 2");
  expect(tensor.numel() == 4, "generic zeros tensor numel must be 4");
  expect(std::ranges::equal(tensor.shape(),
                            DefaultConstructibleTensor::shape_type{2, 2}),
         "generic zeros tensor shape is incorrect");
  expect(std::ranges::equal(tensor.strides(),
                            DefaultConstructibleTensor::strides_type{2, 1}),
         "generic zeros tensor strides are incorrect");

  for (const auto& element : tensor.elements()) {
    expect(element.value == 0,
           "generic zeros element is not value-initialized");
  }
}

void test_scalar_creation() {
  const Tensor tensor = Tensor::scalar(-3.5f);

  expect(tensor.rank() == 0, "scalar tensor rank must be 0");
  expect(tensor.numel() == 1, "scalar tensor numel must be 1");
  expect(tensor.shape().empty(), "scalar tensor shape must be empty");
  expect(tensor.strides().empty(), "scalar tensor strides must be empty");
  expect(tensor.elements()[0] == -3.5f, "scalar tensor element is incorrect");
}

void test_scalar_supports_move_only_non_default_non_assignable_values() {
  const MoveOnlyTensor tensor =
      MoveOnlyTensor::scalar(MoveOnlyNonDefaultNonAssignable(11));

  expect(tensor.rank() == 0, "generic scalar rank must be 0");
  expect(tensor.numel() == 1, "generic scalar numel must be 1");
  expect(tensor.shape().empty(), "generic scalar shape must be empty");
  expect(tensor.strides().empty(), "generic scalar strides must be empty");
  expect(tensor.elements().front().value == 11,
         "generic scalar element is incorrect");
}

void test_from_data_tensor_creation() {
  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_elements{1.0f, 2.0f, 3.0f,
                                               4.0f, 5.0f, 6.0f};

  const Tensor tensor = Tensor::from_data({2, 3}, expected_elements);

  expect(tensor.rank() == 2, "from_data 2x3 tensor rank must be 2");
  expect(tensor.numel() == 6, "from_data 2x3 tensor numel must be 6");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "from_data tensor shape is incorrect");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "from_data tensor strides are incorrect");
  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "from_data tensor elements are incorrect");
}

void test_from_data_scalar_creation() {
  const Tensor tensor = Tensor::from_data({}, {-2.5f});

  expect(tensor.rank() == 0, "from_data scalar rank must be 0");
  expect(tensor.numel() == 1, "from_data scalar numel must be 1");
  expect(tensor.shape().empty(), "from_data scalar shape must be empty");
  expect(tensor.strides().empty(), "from_data scalar strides must be empty");
  expect(tensor.elements()[0] == -2.5f,
         "from_data scalar element is incorrect");
}

void test_from_data_zero_extent_creation() {
  const Tensor tensor = Tensor::from_data({2, 0, 4}, Tensor::storage_type{});

  const Tensor::shape_type expected_shape{2, 0, 4};
  const Tensor::strides_type expected_strides{0, 4, 1};

  expect(tensor.rank() == 3, "from_data zero-extent tensor rank must be 3");
  expect(tensor.numel() == 0, "from_data zero-extent tensor numel must be 0");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "from_data zero-extent tensor shape is incorrect");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "from_data zero-extent tensor strides are incorrect");
  expect(tensor.elements().empty(),
         "from_data zero-extent tensor elements must be empty");
}

void test_from_data_supports_move_only_non_default_non_assignable_values() {
  MoveOnlyTensor::storage_type data;
  data.emplace_back(13);

  const MoveOnlyTensor tensor =
      MoveOnlyTensor::from_data({1}, std::move(data));

  expect(tensor.rank() == 1, "generic from_data tensor rank must be 1");
  expect(tensor.numel() == 1, "generic from_data tensor numel must be 1");
  expect(std::ranges::equal(tensor.shape(), MoveOnlyTensor::shape_type{1}),
         "generic from_data tensor shape is incorrect");
  expect(std::ranges::equal(tensor.strides(), MoveOnlyTensor::strides_type{1}),
         "generic from_data tensor strides are incorrect");
  expect(tensor.elements().front().value == 13,
         "generic from_data tensor element is incorrect");
}

void test_from_data_rejects_too_few_elements() {
  try {
    static_cast<void>(Tensor::from_data(
        {2, 3}, Tensor::storage_type{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}));
  } catch (const std::invalid_argument&) {
    return;
  } catch (...) {
    throw std::runtime_error(
        "from_data with too few elements produced the wrong exception type");
  }

  throw std::runtime_error(
      "from_data with too few elements did not throw std::invalid_argument");
}

void test_from_data_rejects_too_many_elements() {
  try {
    static_cast<void>(Tensor::from_data(
        {2, 3},
        Tensor::storage_type{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}));
  } catch (const std::invalid_argument&) {
    return;
  } catch (...) {
    throw std::runtime_error(
        "from_data with too many elements produced the wrong exception type");
  }

  throw std::runtime_error(
      "from_data with too many elements did not throw std::invalid_argument");
}

void test_from_data_shape_product_overflow() {
  const Tensor::size_type max_size =
      std::numeric_limits<Tensor::size_type>::max();

  try {
    static_cast<void>(Tensor::from_data({max_size, 2}, Tensor::storage_type{}));
  } catch (const std::overflow_error&) {
    return;
  } catch (...) {
    throw std::runtime_error(
        "from_data shape overflow produced the wrong exception type");
  }

  throw std::runtime_error(
      "from_data shape overflow did not throw std::overflow_error");
}

}  // namespace

int main() {
  try {
    test_full_tensor_creation();
    test_full_scalar_creation();
    test_full_zero_extent_creation();
    test_full_preserves_negative_zero();
    test_full_does_not_require_default_construction_or_assignment();
    test_full_shape_product_overflow();
    test_zeros_tensor_creation();
    test_zeros_scalar_creation();
    test_zeros_zero_extent_creation();
    test_zeros_does_not_require_assignment();
    test_scalar_creation();
    test_scalar_supports_move_only_non_default_non_assignable_values();
    test_from_data_tensor_creation();
    test_from_data_scalar_creation();
    test_from_data_zero_extent_creation();
    test_from_data_supports_move_only_non_default_non_assignable_values();
    test_from_data_rejects_too_few_elements();
    test_from_data_rejects_too_many_elements();
    test_from_data_shape_product_overflow();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor creation tests passed\n";
  return EXIT_SUCCESS;
}
