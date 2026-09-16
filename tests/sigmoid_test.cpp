#include <algorithm>
#include <cfenv>
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
concept CanApplySigmoid = requires { nn::sigmoid(std::declval<TensorType>()); };

static_assert(
    std::same_as<decltype(nn::sigmoid(std::declval<const Tensor&>())), Tensor>);
static_assert(CanApplySigmoid<Tensor&>);
static_assert(CanApplySigmoid<const Tensor&>);
static_assert(CanApplySigmoid<Tensor&&>);
static_assert(CanApplySigmoid<const Tensor&&>);
static_assert(CanApplySigmoid<DoubleTensor&>);
static_assert(!CanApplySigmoid<IntegerTensor&>);
static_assert(!noexcept(nn::sigmoid(std::declval<const Tensor&>())));

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

void expect_near(float actual, float expected, float tolerance,
                 const char* message) {
  if (std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(message);
  }
}

void expect_empty_sentinel(const Tensor& tensor) {
  expect(tensor.rank() == 0, "sentinel rank is not zero");
  expect(tensor.numel() == 0, "sentinel numel is not zero");
  expect(tensor.shape().empty(), "sentinel shape is not empty");
  expect(tensor.strides().empty(), "sentinel strides are not empty");
  expect(tensor.elements().empty(), "sentinel elements are not empty");
}

void test_sigmoid_transforms_elements_without_changing_lvalue() {
  const Tensor tensor =
      Tensor::from_data({2, 3}, {-2.0F, -1.0F, 0.0F, 1.0F, 2.0F, 4.0F});
  const Tensor::storage_type expected_source{-2.0F, -1.0F, 0.0F,
                                             1.0F,  2.0F,  4.0F};
  const Tensor::storage_type expected_result{
      0.11920292F, 0.26894143F, 0.5F, 0.73105860F, 0.88079708F, 0.98201376F};

  const Tensor result = nn::sigmoid(tensor);

  expect(std::ranges::equal(result.shape(), Tensor::shape_type{2, 3}),
         "sigmoid changed shape");
  expect(std::ranges::equal(result.strides(), Tensor::strides_type{3, 1}),
         "sigmoid changed strides");
  expect(result.numel() == tensor.numel(), "sigmoid changed numel");

  for (Tensor::size_type index = 0; index < result.numel(); ++index) {
    expect_near(result.elements()[index], expected_result[index], 1.0e-6F,
                "sigmoid produced an inaccurate element");
  }

  expect(std::ranges::equal(tensor.elements(), expected_source),
         "sigmoid changed its tensor lvalue");
  expect(result.data() != tensor.data(),
         "sigmoid result aliases its tensor lvalue");
}

void test_sigmoid_handles_special_floating_point_values() {
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const Tensor tensor =
      Tensor::from_data({5}, {-infinity, infinity, -0.0F, 0.0F, nan});

  const Tensor result = nn::sigmoid(tensor);

  expect(result.elements()[0] == 0.0F,
         "sigmoid handled negative infinity incorrectly");
  expect(result.elements()[1] == 1.0F,
         "sigmoid handled positive infinity incorrectly");
  expect(result.elements()[2] == 0.5F,
         "sigmoid handled negative zero incorrectly");
  expect(result.elements()[3] == 0.5F,
         "sigmoid handled positive zero incorrectly");
  expect(std::isnan(result.elements()[4]), "sigmoid did not propagate NaN");
}

void test_sigmoid_avoids_exponential_overflow() {
  const float maximum = std::numeric_limits<float>::max();
  const Tensor tensor = Tensor::from_data({2}, {-maximum, maximum});

  if ((math_errhandling & MATH_ERREXCEPT) != 0) {
    std::feclearexcept(FE_ALL_EXCEPT);
  }

  const Tensor result = nn::sigmoid(tensor);

  expect(result.elements()[0] == 0.0F,
         "sigmoid handled a large negative value incorrectly");
  expect(result.elements()[1] == 1.0F,
         "sigmoid handled a large positive value incorrectly");

  if ((math_errhandling & MATH_ERREXCEPT) != 0) {
    expect((std::fetestexcept(FE_OVERFLOW) & FE_OVERFLOW) == 0,
           "sigmoid overflowed while evaluating a finite value");
  }
}

void test_sigmoid_supports_rank_zero_and_zero_extent_tensors() {
  const Tensor rank_zero_result = nn::sigmoid(Tensor::scalar(0.0F));
  const Tensor zero_extent_result = nn::sigmoid(Tensor({2, 0, 4}));

  expect(rank_zero_result.rank() == 0, "sigmoid changed rank-zero tensor rank");
  expect(rank_zero_result.numel() == 1,
         "sigmoid changed rank-zero tensor numel");
  expect(rank_zero_result.at() == 0.5F,
         "sigmoid computed an incorrect rank-zero value");
  expect(std::ranges::equal(zero_extent_result.shape(),
                            Tensor::shape_type{2, 0, 4}),
         "sigmoid changed zero-extent shape");
  expect(std::ranges::equal(zero_extent_result.strides(),
                            Tensor::strides_type{0, 4, 1}),
         "sigmoid changed zero-extent strides");
  expect(zero_extent_result.elements().empty(),
         "sigmoid created zero-extent elements");
}

void test_sigmoid_reuses_rvalue_storage() {
  Tensor source = Tensor::from_data({3}, {-1.0F, 0.0F, 1.0F});
  const Tensor::value_type* const original_storage = source.data();

  const Tensor result = nn::sigmoid(std::move(source));

  expect(result.data() == original_storage,
         "sigmoid did not reuse rvalue storage");
  expect_near(result.elements()[0], 0.26894143F, 1.0e-6F,
              "rvalue sigmoid produced an inaccurate negative element");
  expect(result.elements()[1] == 0.5F,
         "rvalue sigmoid produced an incorrect zero element");
  expect_near(result.elements()[2], 0.73105860F, 1.0e-6F,
              "rvalue sigmoid produced an inaccurate positive element");
  expect_empty_sentinel(source);
}

void test_sigmoid_copies_const_rvalue() {
  const Tensor source = Tensor::from_data({2}, {-1.0F, 1.0F});
  const Tensor::value_type* const original_storage = source.data();

  const Tensor result = nn::sigmoid(std::move(source));

  expect(result.data() != original_storage,
         "sigmoid reused const rvalue storage");
  expect(
      std::ranges::equal(source.elements(), Tensor::storage_type{-1.0F, 1.0F}),
      "sigmoid changed its const rvalue argument");
}

void test_sigmoid_rejects_moved_from_sentinel() {
  Tensor source = Tensor::from_data({1}, {0.0F});
  Tensor owner(std::move(source));
  static_cast<void>(owner);

  expect_throws<std::invalid_argument>(
      [&source] { static_cast<void>(nn::sigmoid(source)); },
      "sigmoid accepted an empty sentinel",
      "sentinel sigmoid produced the wrong exception type");
  expect_empty_sentinel(source);
}

}  // namespace

int main() {
  try {
    test_sigmoid_transforms_elements_without_changing_lvalue();
    test_sigmoid_handles_special_floating_point_values();
    test_sigmoid_avoids_exponential_overflow();
    test_sigmoid_supports_rank_zero_and_zero_extent_tensors();
    test_sigmoid_reuses_rvalue_storage();
    test_sigmoid_copies_const_rvalue();
    test_sigmoid_rejects_moved_from_sentinel();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All sigmoid tests passed\n";
  return EXIT_SUCCESS;
}
