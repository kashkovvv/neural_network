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
using BinaryCrossEntropyWithLogitsFunction =
    Tensor (*)(const Tensor&, const Tensor&);

template <typename Logits, typename Target>
concept CanComputeBinaryCrossEntropyWithLogits = requires {
  nn::binary_cross_entropy_with_logits(std::declval<Logits>(),
                                       std::declval<Target>());
};

static_assert(std::same_as<
              decltype(nn::binary_cross_entropy_with_logits(
                  std::declval<const Tensor&>(),
                  std::declval<const Tensor&>())),
              Tensor>);
static_assert(std::same_as<
              decltype(&nn::binary_cross_entropy_with_logits<float>),
              BinaryCrossEntropyWithLogitsFunction>);
static_assert(CanComputeBinaryCrossEntropyWithLogits<Tensor&, Tensor&>);
static_assert(
    CanComputeBinaryCrossEntropyWithLogits<const Tensor&, const Tensor&>);
static_assert(CanComputeBinaryCrossEntropyWithLogits<Tensor&&, const Tensor&>);
static_assert(
    CanComputeBinaryCrossEntropyWithLogits<const Tensor&&, const Tensor&>);
static_assert(CanComputeBinaryCrossEntropyWithLogits<Tensor&, Tensor&&>);
static_assert(CanComputeBinaryCrossEntropyWithLogits<DoubleTensor&,
                                                       const DoubleTensor&>);
static_assert(!CanComputeBinaryCrossEntropyWithLogits<IntegerTensor&,
                                                        const IntegerTensor&>);
static_assert(
    !CanComputeBinaryCrossEntropyWithLogits<Tensor&, const DoubleTensor&>);
static_assert(!noexcept(nn::binary_cross_entropy_with_logits(
    std::declval<const Tensor&>(), std::declval<const Tensor&>())));

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
  expect(tensor.rank() == 0, "BCE-with-logits result rank is not zero");
  expect(tensor.numel() == 1, "BCE-with-logits result numel is not one");
  expect(tensor.shape().empty(), "BCE-with-logits result shape is not empty");
  expect(tensor.strides().empty(),
         "BCE-with-logits result strides are not empty");
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

void test_binary_cross_entropy_with_logits_computes_mean_loss() {
  const Tensor logits =
      Tensor::from_data({2, 2}, {0.0F, 0.0F, 2.0F, -2.0F});
  const Tensor target =
      Tensor::from_data({2, 2}, {0.0F, 1.0F, 1.0F, 0.0F});
  const Tensor::shape_type expected_shape{2, 2};
  const Tensor::strides_type expected_strides{2, 1};
  const Tensor::storage_type expected_logits{0.0F, 0.0F, 2.0F, -2.0F};
  const Tensor::storage_type expected_target{0.0F, 1.0F, 1.0F, 0.0F};

  const Tensor result = nn::binary_cross_entropy_with_logits(logits, target);

  expect_scalar_near(result, 0.4100376F, 1.0e-6F,
                     "BCE-with-logits produced an inaccurate value");
  expect_state(logits, expected_shape, expected_strides, expected_logits,
               "BCE-with-logits changed logits shape",
               "BCE-with-logits changed logits strides",
               "BCE-with-logits changed logits elements");
  expect_state(target, expected_shape, expected_strides, expected_target,
               "BCE-with-logits changed target shape",
               "BCE-with-logits changed target strides",
               "BCE-with-logits changed target elements");
}

void test_binary_cross_entropy_with_logits_supports_soft_targets() {
  const Tensor result = nn::binary_cross_entropy_with_logits(
      Tensor::from_data({3}, {2.0F, -2.0F, 0.0F}),
      Tensor::from_data({3}, {0.25F, 0.75F, 0.5F}));

  expect_scalar_near(result, 1.3156677F, 1.0e-6F,
                     "BCE-with-logits mishandled soft targets");
}

void test_binary_cross_entropy_with_logits_is_stable_for_large_logits() {
  const Tensor positive_match = nn::binary_cross_entropy_with_logits(
      Tensor::scalar(100.0F), Tensor::scalar(1.0F));
  const Tensor negative_match = nn::binary_cross_entropy_with_logits(
      Tensor::scalar(-100.0F), Tensor::scalar(0.0F));
  const Tensor positive_mismatch = nn::binary_cross_entropy_with_logits(
      Tensor::scalar(100.0F), Tensor::scalar(0.0F));
  const Tensor negative_mismatch = nn::binary_cross_entropy_with_logits(
      Tensor::scalar(-100.0F), Tensor::scalar(1.0F));

  expect(std::isfinite(positive_match.at()) && positive_match.at() >= 0.0F &&
             positive_match.at() < 1.0e-30F,
         "BCE-with-logits is unstable for a large positive matched logit");
  expect(std::isfinite(negative_match.at()) && negative_match.at() >= 0.0F &&
             negative_match.at() < 1.0e-30F,
         "BCE-with-logits is unstable for a large negative matched logit");
  expect_scalar_near(positive_mismatch, 100.0F, 0.0F,
                     "BCE-with-logits mishandled a positive mismatch");
  expect_scalar_near(negative_mismatch, 100.0F, 0.0F,
                     "BCE-with-logits mishandled a negative mismatch");
}

void test_binary_cross_entropy_with_logits_handles_special_logits() {
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();

  const Tensor positive_match = nn::binary_cross_entropy_with_logits(
      Tensor::scalar(infinity), Tensor::scalar(1.0F));
  const Tensor negative_match = nn::binary_cross_entropy_with_logits(
      Tensor::scalar(-infinity), Tensor::scalar(0.0F));
  const Tensor positive_mismatch = nn::binary_cross_entropy_with_logits(
      Tensor::scalar(infinity), Tensor::scalar(0.0F));
  const Tensor negative_soft_target = nn::binary_cross_entropy_with_logits(
      Tensor::scalar(-infinity), Tensor::scalar(0.5F));
  const Tensor nan_result = nn::binary_cross_entropy_with_logits(
      Tensor::scalar(nan), Tensor::scalar(0.5F));

  expect_scalar_near(positive_match, 0.0F, 0.0F,
                     "BCE-with-logits mishandled positive infinity");
  expect_scalar_near(negative_match, 0.0F, 0.0F,
                     "BCE-with-logits mishandled negative infinity");
  expect(std::isinf(positive_mismatch.at()) &&
             positive_mismatch.at() > 0.0F,
         "BCE-with-logits did not produce infinity for a positive mismatch");
  expect(std::isinf(negative_soft_target.at()) &&
             negative_soft_target.at() > 0.0F,
         "BCE-with-logits did not produce infinity for a soft-target mismatch");
  expect_nan_scalar(nan_result, "BCE-with-logits did not propagate NaN logits");
}

void test_binary_cross_entropy_with_logits_supports_shapes_and_aliasing() {
  const Tensor scalar_result = nn::binary_cross_entropy_with_logits(
      Tensor::scalar(0.0F), Tensor::scalar(0.0F));
  const Tensor singleton_result = nn::binary_cross_entropy_with_logits(
      Tensor::from_data({1, 1}, {0.0F}),
      Tensor::from_data({1, 1}, {1.0F}));
  const Tensor tensor = Tensor::from_data({2}, {0.0F, 1.0F});
  const Tensor aliased_result =
      nn::binary_cross_entropy_with_logits(tensor, tensor);

  expect_scalar_near(scalar_result, std::log(2.0F), 1.0e-6F,
                     "BCE-with-logits mishandled a scalar");
  expect_scalar_near(singleton_result, std::log(2.0F), 1.0e-6F,
                     "BCE-with-logits mishandled a singleton tensor");
  expect_scalar_near(aliased_result, 0.5032044F, 1.0e-6F,
                     "BCE-with-logits mishandled aliased operands");
  expect(std::ranges::equal(tensor.elements(),
                            Tensor::storage_type{0.0F, 1.0F}),
         "aliased BCE-with-logits changed its source");
}

void test_binary_cross_entropy_with_logits_rejects_invalid_targets() {
  const Tensor logits = Tensor::from_data({2}, {1.0F, -1.0F});
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();

  for (const float invalid_target : {-0.1F, 1.1F, nan, infinity, -infinity}) {
    const Tensor target =
        Tensor::from_data({2}, {0.5F, invalid_target});

    expect_throws<std::domain_error>(
        [&logits, &target] {
          static_cast<void>(
              nn::binary_cross_entropy_with_logits(logits, target));
        },
        "BCE-with-logits accepted an invalid target",
        "invalid BCE-with-logits target produced the wrong exception type");
  }

  expect(std::ranges::equal(logits.elements(),
                            Tensor::storage_type{1.0F, -1.0F}),
         "failed BCE-with-logits changed logits");
}

void test_binary_cross_entropy_with_logits_rejects_different_shapes() {
  const Tensor logits = Tensor::from_data(
      {2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor broadcastable_target =
      Tensor::from_data({3}, {0.0F, 1.0F, 0.0F});
  const Tensor different_shape_target =
      Tensor::from_data({3, 2}, {0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 1.0F});

  expect_throws<std::invalid_argument>(
      [&logits, &broadcastable_target] {
        static_cast<void>(nn::binary_cross_entropy_with_logits(
            logits, broadcastable_target));
      },
      "BCE-with-logits accepted broadcast-compatible shapes",
      "broadcast-compatible BCE-with-logits produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&logits, &different_shape_target] {
        static_cast<void>(nn::binary_cross_entropy_with_logits(
            logits, different_shape_target));
      },
      "BCE-with-logits accepted different shapes with equal numel",
      "different-shape BCE-with-logits produced the wrong exception type");
}

void test_binary_cross_entropy_with_logits_returns_nan_for_empty_tensors() {
  const Tensor vector_result =
      nn::binary_cross_entropy_with_logits(Tensor({0}), Tensor({0}));
  const Tensor tensor_result =
      nn::binary_cross_entropy_with_logits(Tensor({2, 0, 3}),
                                           Tensor({2, 0, 3}));

  expect_nan_scalar(vector_result,
                    "BCE-with-logits of empty vectors did not produce NaN");
  expect_nan_scalar(tensor_result,
                    "BCE-with-logits of empty tensors did not produce NaN");
}

void test_binary_cross_entropy_with_logits_uses_compensated_accumulation() {
  const Tensor result = nn::binary_cross_entropy_with_logits(
      Tensor::from_data(
          {5}, {16777216.0F, 128.0F, 128.0F, 128.0F, 128.0F}),
      Tensor::from_data(
          {5}, {0.0F, 0.9921875F, 0.9921875F, 0.9921875F, 0.9921875F}));

  expect_scalar_near(result, 3355444.0F, 0.0F,
                     "BCE-with-logits did not compensate its loss accumulation");
}

void test_binary_cross_entropy_with_logits_does_not_consume_rvalues() {
  Tensor rvalue_logits = Tensor::from_data({2}, {0.0F, 0.0F});
  const Tensor target = Tensor::from_data({2}, {0.0F, 1.0F});

  const Tensor rvalue_result = nn::binary_cross_entropy_with_logits(
      std::move(rvalue_logits), target);

  expect_scalar_near(rvalue_result, std::log(2.0F), 1.0e-6F,
                     "rvalue BCE-with-logits produced an inaccurate value");
  expect(std::ranges::equal(rvalue_logits.elements(),
                            Tensor::storage_type{0.0F, 0.0F}),
         "BCE-with-logits consumed its rvalue logits");

  const Tensor const_logits = Tensor::from_data({2}, {0.0F, 0.0F});
  Tensor rvalue_target = Tensor::from_data({2}, {0.0F, 1.0F});

  const Tensor const_rvalue_result = nn::binary_cross_entropy_with_logits(
      std::move(const_logits), std::move(rvalue_target));

  expect_scalar_near(
      const_rvalue_result, std::log(2.0F), 1.0e-6F,
      "const-rvalue BCE-with-logits produced an inaccurate value");
  expect(std::ranges::equal(const_logits.elements(),
                            Tensor::storage_type{0.0F, 0.0F}),
         "BCE-with-logits changed const rvalue logits");
  expect(std::ranges::equal(rvalue_target.elements(),
                            Tensor::storage_type{0.0F, 1.0F}),
         "BCE-with-logits consumed its rvalue target");
}

void test_binary_cross_entropy_with_logits_rejects_sentinels() {
  Tensor logits_source = Tensor::from_data({1}, {0.0F});
  Tensor logits_owner(std::move(logits_source));
  Tensor target_source = Tensor::from_data({1}, {1.0F});
  Tensor target_owner(std::move(target_source));
  const Tensor valid_logits = Tensor::from_data({1}, {0.0F});
  const Tensor valid_target = Tensor::from_data({1}, {1.0F});
  static_cast<void>(logits_owner);
  static_cast<void>(target_owner);

  expect_throws<std::invalid_argument>(
      [&logits_source, &valid_target] {
        static_cast<void>(nn::binary_cross_entropy_with_logits(
            logits_source, valid_target));
      },
      "BCE-with-logits accepted moved-from logits",
      "moved-from logits produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&valid_logits, &target_source] {
        static_cast<void>(nn::binary_cross_entropy_with_logits(
            valid_logits, target_source));
      },
      "BCE-with-logits accepted a moved-from target",
      "moved-from target produced the wrong exception type");
  expect_empty_sentinel(logits_source);
  expect_empty_sentinel(target_source);
}

}  // namespace

int main() {
  try {
    test_binary_cross_entropy_with_logits_computes_mean_loss();
    test_binary_cross_entropy_with_logits_supports_soft_targets();
    test_binary_cross_entropy_with_logits_is_stable_for_large_logits();
    test_binary_cross_entropy_with_logits_handles_special_logits();
    test_binary_cross_entropy_with_logits_supports_shapes_and_aliasing();
    test_binary_cross_entropy_with_logits_rejects_invalid_targets();
    test_binary_cross_entropy_with_logits_rejects_different_shapes();
    test_binary_cross_entropy_with_logits_returns_nan_for_empty_tensors();
    test_binary_cross_entropy_with_logits_uses_compensated_accumulation();
    test_binary_cross_entropy_with_logits_does_not_consume_rvalues();
    test_binary_cross_entropy_with_logits_rejects_sentinels();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All BCE-with-logits tests passed\n";
  return EXIT_SUCCESS;
}
