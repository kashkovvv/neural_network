#include <algorithm>
#include <concepts>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "nn/linear.hpp"

namespace {

using Tensor = nn::Tensor<float>;
using DoubleTensor = nn::Tensor<double>;
using Linear = nn::Linear<float>;
using DoubleLinear = nn::Linear<double>;

template <typename Layer, typename Input>
concept CanApplyLinear = requires(Layer&& layer, Input&& input) {
  std::forward<Layer>(layer)(std::forward<Input>(input));
};

static_assert(std::same_as<decltype(std::declval<const Linear&>()(
                               std::declval<const Tensor&>())),
                           Tensor>);
static_assert(CanApplyLinear<Linear&, Tensor&>);
static_assert(CanApplyLinear<const Linear&, const Tensor&>);
static_assert(CanApplyLinear<Linear&&, Tensor&>);
static_assert(CanApplyLinear<const Linear&&, const Tensor&>);
static_assert(CanApplyLinear<Linear&, Tensor&&>);
static_assert(CanApplyLinear<Linear&&, Tensor&&>);
static_assert(CanApplyLinear<DoubleLinear&, DoubleTensor&>);
static_assert(!CanApplyLinear<Linear&, DoubleTensor&>);
static_assert(
    !noexcept(std::declval<const Linear&>()(std::declval<const Tensor&>())));

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

void expect_empty_linear_sentinel(const Linear& layer) {
  expect(layer.in_features() == 0, "moved-from Linear in_features is not zero");
  expect(layer.out_features() == 0,
         "moved-from Linear out_features is not zero");
  expect(layer.weights().rank() == 0,
         "moved-from Linear weights rank is not zero");
  expect(layer.weights().numel() == 0,
         "moved-from Linear weights numel is not zero");
  expect(!layer.bias().has_value(),
         "moved-from Linear still contains a bias object");
}

void test_linear_applies_rank_one_input_with_bias() {
  const Tensor input = Tensor::from_data({3}, {2.0F, -1.0F, 0.5F});
  const Linear layer(
      Tensor::from_data({3, 2}, {1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F}),
      Tensor::from_data({2}, {0.5F, -2.0F}));
  const Tensor::storage_type expected_input{2.0F, -1.0F, 0.5F};
  const Tensor::storage_type expected_weights{1.0F, 4.0F, 2.0F,
                                              5.0F, 3.0F, 6.0F};
  const Tensor::storage_type expected_bias{0.5F, -2.0F};

  const Tensor result = layer(input);

  expect_tensor(result, {2}, {1}, {2.0F, 4.0F},
                "Linear produced an incorrect rank-one result");
  expect(std::ranges::equal(input.elements(), expected_input),
         "Linear changed its rank-one input");
  expect(std::ranges::equal(layer.weights().elements(), expected_weights),
         "Linear changed its weights");
  expect(layer.bias().has_value(), "Linear lost its bias");
  expect(std::ranges::equal(layer.bias()->elements(), expected_bias),
         "Linear changed its bias");
}

void test_linear_applies_rank_two_input_without_bias() {
  const Tensor input =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Linear layer(
      Tensor::from_data({3, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}));

  const Tensor result = layer(input);

  expect_tensor(result, {2, 2}, {2, 1}, {22.0F, 28.0F, 49.0F, 64.0F},
                "bias-free Linear produced an incorrect rank-two result");
}

void test_linear_applies_batched_input_and_broadcasts_bias() {
  const Tensor input = Tensor::from_data(
      {2, 2, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
  const Linear layer(Tensor::from_data({2, 1}, {2.0F, -1.0F}),
                     Tensor::from_data({1}, {0.5F}));

  const Tensor result = layer(input);

  expect_tensor(result, {2, 2, 1}, {2, 1, 1}, {0.5F, 2.5F, 4.5F, 6.5F},
                "Linear produced an incorrect batched result");
}

void test_linear_supports_zero_extent_inputs_and_features() {
  const Linear zero_inner_with_bias(Tensor({0, 2}),
                                    Tensor::from_data({2}, {3.0F, -1.0F}));
  const Linear zero_outputs(Tensor({2, 0}), Tensor({0}));
  const Linear empty_batch(Tensor({2, 3}),
                           Tensor::from_data({3}, {1.0F, 2.0F, 3.0F}));

  const Tensor vector_result = zero_inner_with_bias(Tensor({0}));
  const Tensor matrix_result = zero_inner_with_bias(Tensor({2, 0}));
  const Tensor zero_output_result =
      zero_outputs(Tensor::from_data({2}, {4.0F, 5.0F}));
  const Tensor empty_batch_result = empty_batch(Tensor({0, 2}));

  expect_tensor(vector_result, {2}, {1}, {3.0F, -1.0F},
                "Linear handled an empty vector inner dimension incorrectly");
  expect_tensor(matrix_result, {2, 2}, {2, 1}, {3.0F, -1.0F, 3.0F, -1.0F},
                "Linear handled an empty matrix inner dimension incorrectly");
  expect_tensor(zero_output_result, {0}, {1}, {},
                "Linear produced elements for zero output features");
  expect_tensor(empty_batch_result, {0, 3}, {3, 1}, {},
                "Linear produced elements for an empty batch");
}

void test_linear_rejects_rank_zero_and_incompatible_inputs_without_changes() {
  const Linear layer(Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}),
                     Tensor::from_data({2}, {5.0F, 6.0F}));
  const Tensor scalar = Tensor::scalar(1.0F);
  const Tensor vector = Tensor::from_data({3}, {7.0F, 8.0F, 9.0F});
  const Tensor matrix = Tensor::from_data({1, 3}, {10.0F, 11.0F, 12.0F});
  const Tensor::storage_type expected_weights{1.0F, 2.0F, 3.0F, 4.0F};
  const Tensor::storage_type expected_bias{5.0F, 6.0F};
  const Tensor::storage_type expected_vector{7.0F, 8.0F, 9.0F};
  const Tensor::storage_type expected_matrix{10.0F, 11.0F, 12.0F};

  expect_throws<std::invalid_argument>(
      [&layer, &scalar] { static_cast<void>(layer(scalar)); },
      "Linear accepted a rank-zero input",
      "rank-zero Linear input produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&layer, &vector] { static_cast<void>(layer(vector)); },
      "Linear accepted an incompatible rank-one input",
      "incompatible rank-one input produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&layer, &matrix] { static_cast<void>(layer(matrix)); },
      "Linear accepted an incompatible rank-two input",
      "incompatible rank-two input produced the wrong exception type");
  expect(std::ranges::equal(layer.weights().elements(), expected_weights),
         "failed Linear call changed weights");
  expect(layer.bias().has_value(), "failed Linear call removed bias");
  expect(std::ranges::equal(layer.bias()->elements(), expected_bias),
         "failed Linear call changed bias");
  expect(std::ranges::equal(vector.elements(), expected_vector),
         "failed Linear call changed its vector input");
  expect(std::ranges::equal(matrix.elements(), expected_matrix),
         "failed Linear call changed its matrix input");
}

void test_linear_rejects_moved_from_input_and_layer() {
  Tensor input_source = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor input_owner(std::move(input_source));
  Linear layer_source(Tensor::from_data({2, 1}, {3.0F, 4.0F}));
  Linear layer_owner(std::move(layer_source));
  const Tensor valid_input = Tensor::from_data({2}, {5.0F, 6.0F});
  const Linear valid_layer(Tensor::from_data({2, 1}, {7.0F, 8.0F}));
  static_cast<void>(input_owner);
  static_cast<void>(layer_owner);

  expect_throws<std::invalid_argument>(
      [&valid_layer, &input_source] {
        static_cast<void>(valid_layer(input_source));
      },
      "Linear accepted a moved-from input",
      "moved-from Linear input produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&layer_source, &valid_input] {
        static_cast<void>(layer_source(valid_input));
      },
      "moved-from Linear accepted an input",
      "moved-from Linear produced the wrong exception type");
  expect(input_source.rank() == 0 && input_source.numel() == 0,
         "failed Linear call changed the moved-from input sentinel");
  expect_empty_linear_sentinel(layer_source);
}

void test_linear_accepts_rvalues_without_consuming_them() {
  Linear layer(Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}),
               Tensor::from_data({2}, {5.0F, 6.0F}));
  Tensor input = Tensor::from_data({2}, {7.0F, 8.0F});
  const Tensor::storage_type expected_input{7.0F, 8.0F};
  const Tensor::storage_type expected_weights{1.0F, 2.0F, 3.0F, 4.0F};
  const Tensor::storage_type expected_bias{5.0F, 6.0F};

  const Tensor result = std::move(layer)(std::move(input));

  expect_tensor(result, {2}, {1}, {36.0F, 52.0F},
                "rvalue Linear call produced an incorrect result");
  expect(std::ranges::equal(input.elements(), expected_input),
         "Linear consumed its rvalue input");
  expect(std::ranges::equal(layer.weights().elements(), expected_weights),
         "rvalue call consumed Linear weights");
  expect(layer.bias().has_value(), "rvalue call removed Linear bias");
  expect(std::ranges::equal(layer.bias()->elements(), expected_bias),
         "rvalue call consumed Linear bias");
}

}  // namespace

int main() {
  try {
    test_linear_applies_rank_one_input_with_bias();
    test_linear_applies_rank_two_input_without_bias();
    test_linear_applies_batched_input_and_broadcasts_bias();
    test_linear_supports_zero_extent_inputs_and_features();
    test_linear_rejects_rank_zero_and_incompatible_inputs_without_changes();
    test_linear_rejects_moved_from_input_and_layer();
    test_linear_accepts_rvalues_without_consuming_them();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All Linear forward tests passed\n";
  return EXIT_SUCCESS;
}
