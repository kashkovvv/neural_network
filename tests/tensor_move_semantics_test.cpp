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

static_assert(std::is_nothrow_move_constructible_v<Tensor>);
static_assert(std::is_nothrow_move_assignable_v<Tensor>);
static_assert(std::swappable<Tensor>);
static_assert(std::is_nothrow_swappable_v<Tensor>);
static_assert(std::same_as<
              decltype(std::declval<Tensor&>().swap(std::declval<Tensor&>())),
              void>);
static_assert(noexcept(
    std::declval<Tensor&>().swap(std::declval<Tensor&>())));
static_assert(std::same_as<
              decltype(nn::swap(std::declval<Tensor&>(),
                                std::declval<Tensor&>())),
              void>);
static_assert(noexcept(
    nn::swap(std::declval<Tensor&>(), std::declval<Tensor&>())));

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
  expect(tensor.rank() == 0, "moved-from sentinel rank is not zero");
  expect(tensor.numel() == 0, "moved-from sentinel numel is not zero");
  expect(tensor.shape().empty(), "moved-from sentinel shape is not empty");
  expect(tensor.strides().empty(),
         "moved-from sentinel strides are not empty");
  expect(tensor.elements().empty(),
         "moved-from sentinel elements are not empty");

  expect_throws<std::out_of_range>(
      [&tensor] { static_cast<void>(tensor.at()); },
      "empty sentinel at() did not throw std::out_of_range",
      "empty sentinel at() produced the wrong exception type");

  expect_throws<std::invalid_argument>(
      [&tensor] { static_cast<void>(tensor.at(0)); },
      "nonempty sentinel indexing did not throw std::invalid_argument",
      "nonempty sentinel indexing produced the wrong exception type");
}

void test_move_construction_transfers_complete_state() {
  const Tensor::shape_type expected_shape{2, 2};
  const Tensor::strides_type expected_strides{2, 1};
  const Tensor::storage_type expected_elements{1.0F, 2.0F, 3.0F, 4.0F};

  Tensor source = Tensor::from_data(expected_shape, expected_elements);
  Tensor destination(std::move(source));

  expect(std::ranges::equal(destination.shape(), expected_shape),
         "move construction did not transfer shape");
  expect(std::ranges::equal(destination.strides(), expected_strides),
         "move construction did not transfer strides");
  expect(std::ranges::equal(destination.elements(), expected_elements),
         "move construction did not transfer elements");
  expect_empty_sentinel(source);
}

void test_move_assignment_transfers_complete_state() {
  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_elements{1.0F, 2.0F, 3.0F,
                                                4.0F, 5.0F, 6.0F};

  Tensor source = Tensor::from_data(expected_shape, expected_elements);
  Tensor destination = Tensor::scalar(-1.0F);

  destination = std::move(source);

  expect(std::ranges::equal(destination.shape(), expected_shape),
         "move assignment did not transfer shape");
  expect(std::ranges::equal(destination.strides(), expected_strides),
         "move assignment did not transfer strides");
  expect(std::ranges::equal(destination.elements(), expected_elements),
         "move assignment did not transfer elements");
  expect_empty_sentinel(source);
}

void test_self_move_assignment_preserves_state() {
  Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  const Tensor::shape_type expected_shape(tensor.shape().begin(),
                                           tensor.shape().end());
  const Tensor::strides_type expected_strides(tensor.strides().begin(),
                                               tensor.strides().end());
  const Tensor::storage_type expected_elements(tensor.elements().begin(),
                                                tensor.elements().end());
  Tensor& alias = tensor;

  tensor = std::move(alias);

  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "self-move assignment changed shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "self-move assignment changed strides");
  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "self-move assignment changed elements");
}

void test_moved_from_sentinel_can_be_reassigned() {
  Tensor source = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor destination(std::move(source));
  static_cast<void>(destination);

  source = Tensor::scalar(7.0F);

  expect(source.rank() == 0, "reassigned sentinel has an incorrect rank");
  expect(source.numel() == 1, "reassigned sentinel has an incorrect numel");
  expect(source.at() == 7.0F, "reassigned sentinel has an incorrect value");
}

void test_moved_from_sentinel_can_be_copied_and_moved() {
  Tensor source = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor destination(std::move(source));
  static_cast<void>(destination);

  const Tensor copy(source);
  expect_empty_sentinel(copy);

  Tensor second_destination(std::move(source));
  expect_empty_sentinel(second_destination);
  expect_empty_sentinel(source);
}

void test_swap_exchanges_complete_states() {
  Tensor left = Tensor::scalar(-1.0F);
  Tensor right = Tensor::from_data({2}, {3.0F, 4.0F});
  const auto existing_right_shape = right.shape();
  const auto existing_right_strides = right.strides();
  const auto existing_right_elements = right.elements();
  Tensor::value_type* const existing_right_data = right.elements().data();
  Tensor::value_type& existing_right_reference = right.elements()[1];

  using std::swap;
  swap(left, right);

  const Tensor::shape_type expected_left_shape{2};
  const Tensor::strides_type expected_left_strides{1};
  const Tensor::storage_type expected_left_elements{3.0F, 4.0F};

  expect(std::ranges::equal(left.shape(), expected_left_shape),
         "swap did not transfer shape to the left tensor");
  expect(std::ranges::equal(left.strides(), expected_left_strides),
         "swap did not transfer strides to the left tensor");
  expect(std::ranges::equal(left.elements(), expected_left_elements),
         "swap did not transfer elements to the left tensor");
  expect(std::ranges::equal(existing_right_shape, expected_left_shape),
         "swap invalidated an existing shape view");
  expect(std::ranges::equal(existing_right_strides, expected_left_strides),
         "swap invalidated an existing strides view");
  expect(std::ranges::equal(existing_right_elements, expected_left_elements),
         "swap invalidated an existing elements view");
  expect(left.elements().data() == existing_right_data,
         "swap changed the transferred storage address");
  expect(&left.elements()[1] == &existing_right_reference,
         "swap invalidated an existing element reference");

  expect(right.rank() == 0, "swap did not transfer scalar rank");
  expect(right.numel() == 1, "swap did not transfer scalar numel");
  expect(right.at() == -1.0F, "swap did not transfer scalar value");
}

void test_member_swap_exchanges_complete_states() {
  Tensor left = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor right = Tensor::from_data({1, 2}, {3.0F, 4.0F});

  left.swap(right);

  const Tensor::shape_type expected_left_shape{1, 2};
  const Tensor::strides_type expected_left_strides{2, 1};
  const Tensor::storage_type expected_left_elements{3.0F, 4.0F};
  const Tensor::shape_type expected_right_shape{2};
  const Tensor::strides_type expected_right_strides{1};
  const Tensor::storage_type expected_right_elements{1.0F, 2.0F};

  expect(std::ranges::equal(left.shape(), expected_left_shape),
         "member swap did not transfer shape to the left tensor");
  expect(std::ranges::equal(left.strides(), expected_left_strides),
         "member swap did not transfer strides to the left tensor");
  expect(std::ranges::equal(left.elements(), expected_left_elements),
         "member swap did not transfer elements to the left tensor");
  expect(std::ranges::equal(right.shape(), expected_right_shape),
         "member swap did not transfer shape to the right tensor");
  expect(std::ranges::equal(right.strides(), expected_right_strides),
         "member swap did not transfer strides to the right tensor");
  expect(std::ranges::equal(right.elements(), expected_right_elements),
         "member swap did not transfer elements to the right tensor");
}

void test_self_swap_preserves_state() {
  Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  const Tensor::shape_type expected_shape(tensor.shape().begin(),
                                           tensor.shape().end());
  const Tensor::strides_type expected_strides(tensor.strides().begin(),
                                               tensor.strides().end());
  const Tensor::storage_type expected_elements(tensor.elements().begin(),
                                                tensor.elements().end());

  tensor.swap(tensor);

  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "self-swap changed shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "self-swap changed strides");
  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "self-swap changed elements");
}

void test_swap_exchanges_empty_sentinel() {
  Tensor source = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor destination(std::move(source));
  static_cast<void>(destination);
  Tensor tensor = Tensor::scalar(7.0F);

  source.swap(tensor);

  expect(source.rank() == 0, "sentinel swap produced an incorrect rank");
  expect(source.numel() == 1,
         "sentinel swap produced an incorrect element count");
  expect(source.at() == 7.0F, "sentinel swap produced an incorrect value");
  expect_empty_sentinel(tensor);
}

}  // namespace

int main() {
  try {
    test_move_construction_transfers_complete_state();
    test_move_assignment_transfers_complete_state();
    test_self_move_assignment_preserves_state();
    test_moved_from_sentinel_can_be_reassigned();
    test_moved_from_sentinel_can_be_copied_and_moved();
    test_swap_exchanges_complete_states();
    test_member_swap_exchanges_complete_states();
    test_self_swap_preserves_state();
    test_swap_exchanges_empty_sentinel();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor move semantics tests passed\n";
  return EXIT_SUCCESS;
}
