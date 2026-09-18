#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

#include "nn/losses.hpp"

namespace {

using Tensor = nn::Tensor<float>;
using DoubleTensor = nn::Tensor<double>;
using IntegerTensor = nn::Tensor<int>;
using MseLossFunction = Tensor (*)(const Tensor&, const Tensor&);

template <typename Prediction, typename Target>
concept CanComputeMseLoss = requires {
  nn::mse_loss(std::declval<Prediction>(), std::declval<Target>());
};

static_assert(
    std::same_as<decltype(nn::mse_loss(std::declval<const Tensor&>(),
                                      std::declval<const Tensor&>())),
                 Tensor>);
static_assert(std::same_as<decltype(&nn::mse_loss<float>), MseLossFunction>);
static_assert(CanComputeMseLoss<Tensor&, Tensor&>);
static_assert(CanComputeMseLoss<const Tensor&, const Tensor&>);
static_assert(CanComputeMseLoss<Tensor&&, const Tensor&>);
static_assert(CanComputeMseLoss<const Tensor&&, const Tensor&>);
static_assert(CanComputeMseLoss<Tensor&, Tensor&&>);
static_assert(CanComputeMseLoss<DoubleTensor&, const DoubleTensor&>);
static_assert(!CanComputeMseLoss<IntegerTensor&, const IntegerTensor&>);
static_assert(!CanComputeMseLoss<Tensor&, const DoubleTensor&>);
static_assert(
    !noexcept(nn::mse_loss(std::declval<const Tensor&>(),
                          std::declval<const Tensor&>())));

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

void expect_scalar_metadata(const Tensor& tensor) {
  expect(tensor.rank() == 0, "MSE loss result rank is not zero");
  expect(tensor.numel() == 1, "MSE loss result numel is not one");
  expect(tensor.shape().empty(), "MSE loss result shape is not empty");
  expect(tensor.strides().empty(), "MSE loss result strides are not empty");
}

void expect_scalar_near(const Tensor& tensor, float expected, float tolerance,
                        const char* value_message) {
  expect_scalar_metadata(tensor);

  if (!(std::abs(tensor.at() - expected) <= tolerance)) {
    throw std::runtime_error(value_message);
  }
}

void expect_nan_scalar(const Tensor& tensor, const char* value_message) {
  expect_scalar_metadata(tensor);
  expect(std::isnan(tensor.at()), value_message);
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

void test_mse_loss_computes_mean_squared_error_without_changing_lvalues() {
  const Tensor prediction =
      Tensor::from_data({2, 2}, {1.0F, 2.0F, 4.0F, 8.0F});
  const Tensor target =
      Tensor::from_data({2, 2}, {0.0F, 2.0F, 2.0F, 4.0F});
  const Tensor::shape_type expected_shape{2, 2};
  const Tensor::strides_type expected_strides{2, 1};
  const Tensor::storage_type expected_prediction{1.0F, 2.0F, 4.0F, 8.0F};
  const Tensor::storage_type expected_target{0.0F, 2.0F, 2.0F, 4.0F};

  const Tensor result = nn::mse_loss(prediction, target);

  expect_scalar_near(result, 5.25F, 1.0e-6F,
                     "MSE loss produced an inaccurate value");
  expect_state(prediction, expected_shape, expected_strides,
               expected_prediction, "MSE loss changed prediction shape",
               "MSE loss changed prediction strides",
               "MSE loss changed prediction elements");
  expect_state(target, expected_shape, expected_strides, expected_target,
               "MSE loss changed target shape",
               "MSE loss changed target strides",
               "MSE loss changed target elements");
}

void test_mse_loss_supports_rank_zero_singleton_and_aliased_tensors() {
  const Tensor scalar_result =
      nn::mse_loss(Tensor::scalar(4.0F), Tensor::scalar(1.0F));
  const Tensor singleton_result = nn::mse_loss(
      Tensor::from_data({1, 1}, {-2.0F}),
      Tensor::from_data({1, 1}, {3.0F}));
  const Tensor tensor = Tensor::from_data({3}, {1.0F, -2.0F, 4.0F});
  const Tensor aliased_result = nn::mse_loss(tensor, tensor);

  expect_scalar_near(scalar_result, 9.0F, 0.0F,
                     "MSE loss computed an incorrect rank-zero value");
  expect_scalar_near(singleton_result, 25.0F, 0.0F,
                     "MSE loss computed an incorrect singleton value");
  expect_scalar_near(aliased_result, 0.0F, 0.0F,
                     "MSE loss computed an incorrect aliased value");
  expect(std::ranges::equal(tensor.elements(),
                            Tensor::storage_type{1.0F, -2.0F, 4.0F}),
         "aliased MSE loss changed its source");
}

void test_mse_loss_rejects_broadcasting_and_different_shapes() {
  const Tensor prediction = Tensor::from_data(
      {2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor broadcastable_target =
      Tensor::from_data({3}, {1.0F, 2.0F, 3.0F});
  const Tensor transposed_shape_target = Tensor::from_data(
      {3, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});

  expect_throws<std::invalid_argument>(
      [&prediction, &broadcastable_target] {
        static_cast<void>(nn::mse_loss(prediction, broadcastable_target));
      },
      "MSE loss accepted broadcast-compatible shapes",
      "broadcast-compatible MSE loss produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&prediction, &transposed_shape_target] {
        static_cast<void>(nn::mse_loss(prediction, transposed_shape_target));
      },
      "MSE loss accepted different shapes with equal numel",
      "different-shape MSE loss produced the wrong exception type");

  expect(std::ranges::equal(
             prediction.elements(),
             Tensor::storage_type{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}),
         "failed MSE loss changed prediction");
  expect(std::ranges::equal(broadcastable_target.elements(),
                            Tensor::storage_type{1.0F, 2.0F, 3.0F}),
         "failed MSE loss changed broadcast-compatible target");
  expect(std::ranges::equal(
             transposed_shape_target.elements(),
             Tensor::storage_type{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}),
         "failed MSE loss changed different-shape target");
}

void test_mse_loss_returns_nan_for_equal_zero_extent_shapes() {
  const Tensor vector_result = nn::mse_loss(Tensor({0}), Tensor({0}));
  const Tensor tensor_result =
      nn::mse_loss(Tensor({2, 0, 3}), Tensor({2, 0, 3}));

  expect_nan_scalar(vector_result,
                    "MSE loss of empty vectors did not produce NaN");
  expect_nan_scalar(tensor_result,
                    "MSE loss of empty tensors did not produce NaN");
}

void test_mse_loss_uses_native_floating_point_behavior() {
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float maximum = std::numeric_limits<float>::max();

  const Tensor nan_input_result = nn::mse_loss(
      Tensor::from_data({2}, {nan, 1.0F}),
      Tensor::from_data({2}, {0.0F, 1.0F}));
  const Tensor indeterminate_result = nn::mse_loss(
      Tensor::from_data({1}, {infinity}),
      Tensor::from_data({1}, {infinity}));
  const Tensor infinity_result = nn::mse_loss(
      Tensor::from_data({1}, {infinity}), Tensor::from_data({1}, {0.0F}));
  const Tensor overflow_result = nn::mse_loss(
      Tensor::from_data({1}, {maximum}),
      Tensor::from_data({1}, {-maximum}));

  expect_nan_scalar(nan_input_result, "MSE loss did not propagate NaN");
  expect_nan_scalar(indeterminate_result,
                    "MSE loss did not preserve infinity subtraction behavior");
  expect(std::isinf(infinity_result.at()) && infinity_result.at() > 0.0F,
         "MSE loss did not preserve positive infinity");
  expect(std::isinf(overflow_result.at()) && overflow_result.at() > 0.0F,
         "MSE loss did not preserve native floating-point overflow");
}

void test_mse_loss_uses_compensated_accumulation() {
  const Tensor result = nn::mse_loss(
      Tensor::from_data({5}, {4096.0F, 1.0F, 1.0F, 1.0F, 1.0F}),
      Tensor::zeros({5}));

  expect_scalar_near(result, 3355444.0F, 0.0F,
                     "MSE loss did not compensate its loss accumulation");
}

void test_mse_loss_does_not_consume_rvalue_prediction() {
  Tensor prediction = Tensor::from_data({2}, {3.0F, 5.0F});
  const Tensor target = Tensor::from_data({2}, {1.0F, 1.0F});

  const Tensor result = nn::mse_loss(std::move(prediction), target);

  expect_scalar_near(result, 10.0F, 1.0e-6F,
                     "rvalue-prediction MSE loss produced an inaccurate value");
  expect(std::ranges::equal(prediction.elements(),
                            Tensor::storage_type{3.0F, 5.0F}),
         "MSE loss consumed its rvalue prediction");
}

void test_mse_loss_does_not_consume_const_rvalue_prediction() {
  const Tensor prediction = Tensor::from_data({2}, {3.0F, 5.0F});
  const Tensor target = Tensor::from_data({2}, {1.0F, 1.0F});

  const Tensor result = nn::mse_loss(std::move(prediction), target);

  expect_scalar_near(result, 10.0F, 1.0e-6F,
                     "const-rvalue MSE loss produced an inaccurate value");
  expect(std::ranges::equal(prediction.elements(),
                            Tensor::storage_type{3.0F, 5.0F}),
         "MSE loss changed its const rvalue prediction");
}

void test_mse_loss_does_not_consume_rvalue_target() {
  const Tensor prediction = Tensor::from_data({2}, {3.0F, 5.0F});
  Tensor target = Tensor::from_data({2}, {1.0F, 1.0F});

  const Tensor result = nn::mse_loss(prediction, std::move(target));

  expect_scalar_near(result, 10.0F, 1.0e-6F,
                     "rvalue-target MSE loss produced an inaccurate value");
  expect(std::ranges::equal(target.elements(),
                            Tensor::storage_type{1.0F, 1.0F}),
         "MSE loss consumed its rvalue target");
}

void test_mse_loss_rejects_moved_from_sentinels() {
  Tensor prediction_source = Tensor::from_data({1}, {1.0F});
  Tensor prediction_owner(std::move(prediction_source));
  Tensor target_source = Tensor::from_data({1}, {1.0F});
  Tensor target_owner(std::move(target_source));
  const Tensor valid = Tensor::from_data({1}, {1.0F});
  static_cast<void>(prediction_owner);
  static_cast<void>(target_owner);

  expect_throws<std::invalid_argument>(
      [&prediction_source, &valid] {
        static_cast<void>(nn::mse_loss(prediction_source, valid));
      },
      "MSE loss accepted a moved-from prediction",
      "moved-from prediction produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&target_source, &valid] {
        static_cast<void>(nn::mse_loss(valid, target_source));
      },
      "MSE loss accepted a moved-from target",
      "moved-from target produced the wrong exception type");
  expect_empty_sentinel(prediction_source);
  expect_empty_sentinel(target_source);
  expect(std::ranges::equal(valid.elements(), Tensor::storage_type{1.0F}),
         "failed sentinel MSE loss changed its valid operand");
}

}  // namespace

int main() {
  try {
    test_mse_loss_computes_mean_squared_error_without_changing_lvalues();
    test_mse_loss_supports_rank_zero_singleton_and_aliased_tensors();
    test_mse_loss_rejects_broadcasting_and_different_shapes();
    test_mse_loss_returns_nan_for_equal_zero_extent_shapes();
    test_mse_loss_uses_native_floating_point_behavior();
    test_mse_loss_uses_compensated_accumulation();
    test_mse_loss_does_not_consume_rvalue_prediction();
    test_mse_loss_does_not_consume_const_rvalue_prediction();
    test_mse_loss_does_not_consume_rvalue_target();
    test_mse_loss_rejects_moved_from_sentinels();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All MSE loss tests passed\n";
  return EXIT_SUCCESS;
}
