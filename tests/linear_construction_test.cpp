#include <algorithm>
#include <concepts>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "nn/linear.hpp"

namespace {

using Tensor = nn::Tensor<float>;
using Linear = nn::Linear<float>;
using DoubleLinear = nn::Linear<double>;

template <typename Element>
concept CanInstantiateLinear = requires { typename nn::Linear<Element>; };

template <typename Layer>
concept HasRvalueWeights =
    requires(Layer&& layer) { std::move(layer).weights(); };

template <typename Layer>
concept HasRvalueBias = requires(Layer&& layer) { std::move(layer).bias(); };

static_assert(std::same_as<typename Linear::value_type, float>);
static_assert(std::same_as<typename Linear::tensor_type, Tensor>);
static_assert(std::same_as<typename Linear::size_type, Tensor::size_type>);
static_assert(std::same_as<typename DoubleLinear::value_type, double>);
static_assert(CanInstantiateLinear<float>);
static_assert(CanInstantiateLinear<double>);
static_assert(!CanInstantiateLinear<int>);
static_assert(!CanInstantiateLinear<const float>);
static_assert(!CanInstantiateLinear<float&>);

static_assert(!std::default_initializable<Linear>);
static_assert(std::constructible_from<Linear, Tensor>);
static_assert(std::constructible_from<Linear, Tensor, std::optional<Tensor>>);
static_assert(!std::convertible_to<Tensor, Linear>);
static_assert(std::copy_constructible<Linear>);
static_assert(std::move_constructible<Linear>);
static_assert(std::is_copy_assignable_v<Linear>);
static_assert(std::is_move_assignable_v<Linear>);
static_assert(std::is_nothrow_move_constructible_v<Linear>);
static_assert(std::is_nothrow_move_assignable_v<Linear>);
static_assert(std::is_nothrow_swappable_v<Linear>);
static_assert(
    std::same_as<
        decltype(std::declval<Linear&>().swap(std::declval<Linear&>())), void>);
static_assert(noexcept(std::declval<Linear&>().swap(std::declval<Linear&>())));
static_assert(std::same_as<decltype(nn::swap(std::declval<Linear&>(),
                                             std::declval<Linear&>())),
                           void>);
static_assert(noexcept(nn::swap(std::declval<Linear&>(),
                                std::declval<Linear&>())));

static_assert(std::same_as<decltype(std::declval<const Linear&>().weights()),
                           const Tensor&>);
static_assert(std::same_as<decltype(std::declval<const Linear&>().bias()),
                           const std::optional<Tensor>&>);
static_assert(
    std::same_as<decltype(std::declval<const Linear&>().in_features()),
                 Linear::size_type>);
static_assert(
    std::same_as<decltype(std::declval<const Linear&>().out_features()),
                 Linear::size_type>);
static_assert(!HasRvalueWeights<Linear>);
static_assert(!HasRvalueBias<Linear>);

static_assert(noexcept(std::declval<const Linear&>().weights()));
static_assert(noexcept(std::declval<const Linear&>().bias()));
static_assert(noexcept(std::declval<const Linear&>().in_features()));
static_assert(noexcept(std::declval<const Linear&>().out_features()));
static_assert(!noexcept(Linear(std::declval<Tensor>())));

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
                   const Tensor::storage_type& expected_elements,
                   const char* message) {
  expect(std::ranges::equal(tensor.shape(), expected_shape), message);
  expect(std::ranges::equal(tensor.elements(), expected_elements), message);
}

void expect_empty_sentinel(const Linear& layer) {
  expect(layer.in_features() == 0, "moved-from Linear in_features is not zero");
  expect(layer.out_features() == 0,
         "moved-from Linear out_features is not zero");
  expect(layer.weights().rank() == 0,
         "moved-from Linear weights rank is not zero");
  expect(layer.weights().numel() == 0,
         "moved-from Linear weights numel is not zero");
  expect(layer.weights().shape().empty(),
         "moved-from Linear weights shape is not empty");
  expect(layer.weights().strides().empty(),
         "moved-from Linear weights strides are not empty");
  expect(layer.weights().elements().empty(),
         "moved-from Linear weights storage is not empty");
  expect(!layer.bias().has_value(),
         "moved-from Linear still contains a bias object");
}

void test_linear_owns_weights_and_bias() {
  Tensor weights =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  Tensor bias = Tensor::from_data({3}, {7.0F, 8.0F, 9.0F});

  const Linear layer(weights, bias);

  weights.at(0, 0) = -1.0F;
  bias.at(0) = -7.0F;

  expect(layer.in_features() == 2, "Linear reported incorrect in_features");
  expect(layer.out_features() == 3, "Linear reported incorrect out_features");
  expect_tensor(layer.weights(), {2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
                "Linear did not own an independent weights tensor");
  expect(layer.bias().has_value(), "Linear lost its bias tensor");
  expect_tensor(*layer.bias(), {3}, {7.0F, 8.0F, 9.0F},
                "Linear did not own an independent bias tensor");
}

void test_linear_supports_disabled_bias() {
  const Linear layer(
      Tensor::from_data({3, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}));

  expect(layer.in_features() == 3,
         "bias-free Linear reported incorrect in_features");
  expect(layer.out_features() == 2,
         "bias-free Linear reported incorrect out_features");
  expect(!layer.bias().has_value(),
         "bias-free Linear unexpectedly contains a bias tensor");
}

void test_linear_accepts_zero_extent_feature_dimensions() {
  const Linear zero_inputs(Tensor({0, 3}), Tensor({3}));
  const Linear zero_outputs(Tensor({2, 0}), Tensor({0}));
  const Linear fully_empty(Tensor({0, 0}));

  expect(zero_inputs.in_features() == 0 && zero_inputs.out_features() == 3,
         "Linear rejected or changed a zero input-feature dimension");
  expect(zero_outputs.in_features() == 2 && zero_outputs.out_features() == 0,
         "Linear rejected or changed a zero output-feature dimension");
  expect(fully_empty.in_features() == 0 && fully_empty.out_features() == 0,
         "Linear rejected or changed two zero feature dimensions");
}

void test_linear_rejects_invalid_weights_rank() {
  const Tensor scalar = Tensor::scalar(1.0F);
  const Tensor vector = Tensor::from_data({2}, {1.0F, 2.0F});
  const Tensor rank_three = Tensor({1, 2, 3});

  expect_throws<std::invalid_argument>(
      [&scalar] { static_cast<void>(Linear(scalar)); },
      "Linear accepted rank-zero weights",
      "rank-zero weights produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&vector] { static_cast<void>(Linear(vector)); },
      "Linear accepted rank-one weights",
      "rank-one weights produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&rank_three] { static_cast<void>(Linear(rank_three)); },
      "Linear accepted rank-three weights",
      "rank-three weights produced the wrong exception type");
}

void test_linear_rejects_invalid_bias_shape() {
  const Tensor weights = Tensor({2, 3});
  const Tensor scalar_bias = Tensor::scalar(1.0F);
  const Tensor matrix_bias = Tensor({1, 3});
  const Tensor wrong_size_bias = Tensor({2});

  expect_throws<std::invalid_argument>(
      [&weights, &scalar_bias] {
        static_cast<void>(Linear(weights, scalar_bias));
      },
      "Linear accepted rank-zero bias",
      "rank-zero bias produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&weights, &matrix_bias] {
        static_cast<void>(Linear(weights, matrix_bias));
      },
      "Linear accepted rank-two bias",
      "rank-two bias produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&weights, &wrong_size_bias] {
        static_cast<void>(Linear(weights, wrong_size_bias));
      },
      "Linear accepted bias with an incompatible extent",
      "incompatible bias extent produced the wrong exception type");
}

void test_linear_rejects_moved_from_parameters() {
  Tensor weights_source = Tensor({2, 3});
  Tensor weights_owner(std::move(weights_source));
  Tensor bias_source = Tensor({3});
  Tensor bias_owner(std::move(bias_source));
  const Tensor valid_weights = Tensor({2, 3});
  static_cast<void>(weights_owner);
  static_cast<void>(bias_owner);

  expect_throws<std::invalid_argument>(
      [&weights_source] { static_cast<void>(Linear(weights_source)); },
      "Linear accepted moved-from weights",
      "moved-from weights produced the wrong exception type");
  expect_throws<std::invalid_argument>(
      [&valid_weights, &bias_source] {
        static_cast<void>(Linear(valid_weights, bias_source));
      },
      "Linear accepted moved-from bias",
      "moved-from bias produced the wrong exception type");
}

void test_linear_copy_semantics_preserve_independent_valid_state() {
  const Linear source(
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}),
      Tensor::from_data({3}, {7.0F, 8.0F, 9.0F}));
  const Linear copy(source);
  Linear assigned(Tensor({4, 1}));

  assigned = source;

  expect_tensor(copy.weights(), {2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
                "Linear copy construction changed weights");
  expect(copy.bias().has_value(), "Linear copy construction lost bias");
  expect_tensor(*copy.bias(), {3}, {7.0F, 8.0F, 9.0F},
                "Linear copy construction changed bias");
  expect_tensor(assigned.weights(), {2, 3},
                {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
                "Linear copy assignment changed weights");
  expect(assigned.bias().has_value(), "Linear copy assignment lost bias");
  expect_tensor(*assigned.bias(), {3}, {7.0F, 8.0F, 9.0F},
                "Linear copy assignment changed bias");

  assigned = assigned;

  expect(assigned.in_features() == 2 && assigned.out_features() == 3,
         "Linear self-copy assignment changed feature counts");
  expect(assigned.bias().has_value(),
         "Linear self-copy assignment changed bias state");
}

void test_linear_move_semantics_create_empty_sentinel() {
  Linear constructor_source(Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}),
                            Tensor::from_data({2}, {5.0F, 6.0F}));

  const Linear moved(std::move(constructor_source));

  expect_tensor(moved.weights(), {2, 2}, {1.0F, 2.0F, 3.0F, 4.0F},
                "Linear move construction changed weights");
  expect(moved.bias().has_value(), "Linear move construction lost bias");
  expect_tensor(*moved.bias(), {2}, {5.0F, 6.0F},
                "Linear move construction changed bias");
  expect_empty_sentinel(constructor_source);

  Linear assignment_source(Tensor::from_data({3, 1}, {7.0F, 8.0F, 9.0F}));
  Linear assigned(Tensor({1, 4}), Tensor({4}));

  assigned = std::move(assignment_source);

  expect_tensor(assigned.weights(), {3, 1}, {7.0F, 8.0F, 9.0F},
                "Linear move assignment changed weights");
  expect(!assigned.bias().has_value(),
         "Linear move assignment did not remove the old bias");
  expect_empty_sentinel(assignment_source);

  Linear& assigned_alias = assigned;
  assigned = std::move(assigned_alias);

  expect_tensor(assigned.weights(), {3, 1}, {7.0F, 8.0F, 9.0F},
                "Linear self-move assignment changed weights");
  expect(!assigned.bias().has_value(),
         "Linear self-move assignment changed bias state");
}

void test_linear_empty_sentinel_supports_value_semantics() {
  Linear source(Tensor({2, 1}), Tensor({1}));
  Linear owner(std::move(source));
  static_cast<void>(owner);

  const Linear copy(source);
  Linear assigned(Tensor({1, 2}), Tensor({2}));

  assigned = source;

  expect_empty_sentinel(copy);
  expect_empty_sentinel(assigned);

  source =
      Linear(Tensor::from_data({1, 1}, {3.0F}), Tensor::from_data({1}, {4.0F}));

  expect(source.in_features() == 1 && source.out_features() == 1,
         "reassigned Linear sentinel has incorrect feature counts");
  expect_tensor(source.weights(), {1, 1}, {3.0F},
                "reassigned Linear sentinel has incorrect weights");
  expect(source.bias().has_value(), "reassigned Linear sentinel lost its bias");
  expect_tensor(*source.bias(), {1}, {4.0F},
                "reassigned Linear sentinel has incorrect bias");
}

void test_linear_swap_exchanges_complete_states() {
  Linear first(Tensor::from_data({1, 2}, {1.0F, 2.0F}),
               Tensor::from_data({2}, {3.0F, 4.0F}));
  Linear second(Tensor::from_data({3, 1}, {5.0F, 6.0F, 7.0F}));

  using std::swap;
  swap(first, second);

  expect_tensor(first.weights(), {3, 1}, {5.0F, 6.0F, 7.0F},
                "Linear swap produced incorrect first weights");
  expect(!first.bias().has_value(),
         "Linear swap produced incorrect first bias state");
  expect_tensor(second.weights(), {1, 2}, {1.0F, 2.0F},
                "Linear swap produced incorrect second weights");
  expect(second.bias().has_value(),
         "Linear swap produced incorrect second bias state");
  expect_tensor(*second.bias(), {2}, {3.0F, 4.0F},
                "Linear swap produced incorrect second bias");
}

}  // namespace

int main() {
  try {
    test_linear_owns_weights_and_bias();
    test_linear_supports_disabled_bias();
    test_linear_accepts_zero_extent_feature_dimensions();
    test_linear_rejects_invalid_weights_rank();
    test_linear_rejects_invalid_bias_shape();
    test_linear_rejects_moved_from_parameters();
    test_linear_copy_semantics_preserve_independent_valid_state();
    test_linear_move_semantics_create_empty_sentinel();
    test_linear_empty_sentinel_supports_value_semantics();
    test_linear_swap_exchanges_complete_states();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All Linear construction tests passed\n";
  return EXIT_SUCCESS;
}
