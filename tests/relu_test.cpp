#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

#include "nn/activations.hpp"

namespace {

using Tensor = nn::Tensor<float>;
using DoubleTensor = nn::Tensor<double>;
using IntegerTensor = nn::Tensor<int>;

template <typename TensorType>
concept CanApplyRelu = requires { nn::relu(std::declval<TensorType>()); };

static_assert(
    std::same_as<decltype(nn::relu(std::declval<const Tensor&>())), Tensor>);
static_assert(CanApplyRelu<Tensor&>);
static_assert(CanApplyRelu<const Tensor&>);
static_assert(CanApplyRelu<Tensor&&>);
static_assert(CanApplyRelu<const Tensor&&>);
static_assert(CanApplyRelu<DoubleTensor&>);
static_assert(!CanApplyRelu<IntegerTensor&>);
static_assert(!noexcept(nn::relu(std::declval<const Tensor&>())));

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

void expect_tensor(const Tensor& tensor,
                   const Tensor::shape_type& expected_shape,
                   const Tensor::strides_type& expected_strides,
                   const Tensor::storage_type& expected_elements,
                   const char* message) {
  expect(std::ranges::equal(tensor.shape(), expected_shape), message);
  expect(std::ranges::equal(tensor.strides(), expected_strides), message);
  expect(std::ranges::equal(tensor.elements(), expected_elements), message);
}

void test_relu_transforms_elements_without_changing_lvalue() {
  const Tensor tensor =
      Tensor::from_data({2, 3}, {-3.0F, -0.5F, 0.0F, 1.0F, 4.5F, -7.0F});
  const Tensor::storage_type expected_source{-3.0F, -0.5F, 0.0F,
                                             1.0F,  4.5F,  -7.0F};

  const Tensor result = nn::relu(tensor);

  expect_tensor(result, {2, 3}, {3, 1}, {0.0F, 0.0F, 0.0F, 1.0F, 4.5F, 0.0F},
                "ReLU produced an incorrect tensor");
  expect(std::ranges::equal(tensor.elements(), expected_source),
         "ReLU changed its tensor lvalue");
  expect(result.data() != tensor.data(),
         "ReLU result aliases its tensor lvalue");
}

void test_relu_handles_special_floating_point_values() {
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const Tensor tensor =
      Tensor::from_data({5}, {-infinity, infinity, -0.0F, 0.0F, nan});

  const Tensor result = nn::relu(tensor);

  expect(result.elements()[0] == 0.0F && !std::signbit(result.elements()[0]),
         "ReLU handled negative infinity incorrectly");
  expect(std::isinf(result.elements()[1]) &&
             !std::signbit(result.elements()[1]),
         "ReLU handled positive infinity incorrectly");
  expect(result.elements()[2] == 0.0F && std::signbit(result.elements()[2]),
         "ReLU changed negative zero");
  expect(result.elements()[3] == 0.0F && !std::signbit(result.elements()[3]),
         "ReLU changed positive zero");
  expect(std::isnan(result.elements()[4]), "ReLU did not propagate NaN");
}

void test_relu_supports_rank_zero_and_zero_extent_tensors() {
  const Tensor rank_zero_result = nn::relu(Tensor::scalar(-2.5F));
  const Tensor zero_extent_result = nn::relu(Tensor({2, 0, 4}));

  expect(rank_zero_result.rank() == 0, "ReLU changed rank-zero tensor rank");
  expect(rank_zero_result.numel() == 1, "ReLU changed rank-zero tensor numel");
  expect(rank_zero_result.at() == 0.0F,
         "ReLU computed an incorrect rank-zero value");
  expect_tensor(zero_extent_result, {2, 0, 4}, {0, 4, 1}, {},
                "ReLU handled a zero-extent tensor incorrectly");
}

void test_relu_reuses_rvalue_storage() {
  Tensor source = Tensor::from_data({3}, {-1.0F, 2.0F, -3.0F});
  const Tensor::value_type* const original_storage = source.data();

  const Tensor result = nn::relu(std::move(source));

  expect(result.data() == original_storage,
         "ReLU did not reuse rvalue storage");
  expect(std::ranges::equal(result.elements(),
                            Tensor::storage_type{0.0F, 2.0F, 0.0F}),
         "rvalue ReLU produced incorrect elements");
  expect_empty_sentinel(source);
}

void test_relu_copies_const_rvalue() {
  const Tensor source = Tensor::from_data({2}, {-1.0F, 2.0F});
  const Tensor::value_type* const original_storage = source.data();

  const Tensor result = nn::relu(std::move(source));

  expect(result.data() != original_storage, "ReLU reused const rvalue storage");
  expect(
      std::ranges::equal(source.elements(), Tensor::storage_type{-1.0F, 2.0F}),
      "ReLU changed its const rvalue argument");
}

void test_relu_rejects_moved_from_sentinel() {
  Tensor source = Tensor::from_data({1}, {-1.0F});
  Tensor owner(std::move(source));
  static_cast<void>(owner);

  expect_throws<std::invalid_argument>(
      [&source] { static_cast<void>(nn::relu(source)); },
      "ReLU accepted an empty sentinel",
      "sentinel ReLU produced the wrong exception type");
  expect_empty_sentinel(source);
}

}  // namespace

int main() {
  try {
    test_relu_transforms_elements_without_changing_lvalue();
    test_relu_handles_special_floating_point_values();
    test_relu_supports_rank_zero_and_zero_extent_tensors();
    test_relu_reuses_rvalue_storage();
    test_relu_copies_const_rvalue();
    test_relu_rejects_moved_from_sentinel();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All ReLU tests passed\n";
  return EXIT_SUCCESS;
}
