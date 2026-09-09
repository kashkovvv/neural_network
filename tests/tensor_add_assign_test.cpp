#include <algorithm>
#include <concepts>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "nn/tensor.hpp"

namespace {

using Tensor = nn::Tensor<float>;
using DoubleTensor = nn::Tensor<double>;
using IntegerTensor = nn::Tensor<int>;

template <typename Left, typename Right>
concept CanAddAssign = requires {
  std::declval<Left>() += std::declval<Right>();
};

static_assert(std::same_as<decltype(std::declval<Tensor&>() +=
                                    std::declval<const Tensor&>()),
                           Tensor&>);
static_assert(CanAddAssign<DoubleTensor&, const DoubleTensor&>);
static_assert(!CanAddAssign<const Tensor&, const Tensor&>);
static_assert(!CanAddAssign<Tensor&&, const Tensor&>);
static_assert(!CanAddAssign<const Tensor&&, const Tensor&>);
static_assert(!CanAddAssign<IntegerTensor&, const IntegerTensor&>);
static_assert(!CanAddAssign<Tensor&, const DoubleTensor&>);
static_assert(
    !noexcept(std::declval<Tensor&>() += std::declval<const Tensor&>()));

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

void expect_state(const Tensor& tensor, const Tensor::shape_type& shape,
                  const Tensor::strides_type& strides,
                  const Tensor::storage_type& elements,
                  const char* shape_message, const char* strides_message,
                  const char* elements_message) {
  expect(std::ranges::equal(tensor.shape(), shape), shape_message);
  expect(std::ranges::equal(tensor.strides(), strides), strides_message);
  expect(std::ranges::equal(tensor.elements(), elements), elements_message);
}

void expect_add_assign_rejected_without_left_change(Tensor& left,
                                                    const Tensor& right) {
  const Tensor::shape_type original_shape(left.shape().begin(),
                                           left.shape().end());
  const Tensor::strides_type original_strides(left.strides().begin(),
                                               left.strides().end());
  const Tensor::storage_type original_elements(left.elements().begin(),
                                                left.elements().end());

  expect_throws<std::invalid_argument>(
      [&left, &right] { left += right; },
      "invalid tensor addition assignment did not throw std::invalid_argument",
      "invalid tensor addition assignment produced the wrong exception type");

  expect_state(left, original_shape, original_strides, original_elements,
               "failed addition assignment changed the left shape",
               "failed addition assignment changed the left strides",
               "failed addition assignment changed the left elements");
}

void test_add_assign_computes_elementwise_and_returns_left_operand() {
  Tensor left =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor right =
      Tensor::from_data({2, 3}, {6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F});
  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_elements{7.0F, 7.0F, 7.0F,
                                                7.0F, 7.0F, 7.0F};
  const Tensor::storage_type expected_right_elements{6.0F, 5.0F, 4.0F,
                                                      3.0F, 2.0F, 1.0F};

  Tensor& result = (left += right);

  expect(&result == &left, "addition assignment returned the wrong object");
  expect_state(left, expected_shape, expected_strides, expected_elements,
               "addition assignment changed the left shape",
               "addition assignment changed the left strides",
               "addition assignment computed incorrect elements");
  expect(std::ranges::equal(right.elements(), expected_right_elements),
         "addition assignment changed the right elements");
}

void test_add_assign_supports_scalars() {
  Tensor left = Tensor::scalar(2.5F);
  const Tensor right = Tensor::scalar(-0.5F);

  left += right;

  expect(left.rank() == 0, "scalar addition assignment changed rank");
  expect(left.numel() == 1, "scalar addition assignment changed numel");
  expect(left.at() == 2.0F,
         "scalar addition assignment computed an incorrect value");
}

void test_add_assign_broadcasts_lower_rank_right_operand() {
  Tensor left =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor right = Tensor::from_data({3}, {10.0F, 20.0F, 30.0F});
  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_elements{11.0F, 22.0F, 33.0F,
                                                14.0F, 25.0F, 36.0F};
  const Tensor::shape_type expected_right_shape{3};
  const Tensor::strides_type expected_right_strides{1};
  const Tensor::storage_type expected_right_elements{10.0F, 20.0F, 30.0F};

  left += right;

  expect_state(left, expected_shape, expected_strides, expected_elements,
               "broadcast addition assignment changed the left shape",
               "broadcast addition assignment changed the left strides",
               "broadcast addition assignment computed incorrect elements");
  expect_state(right, expected_right_shape, expected_right_strides,
               expected_right_elements,
               "broadcast addition assignment changed the right shape",
               "broadcast addition assignment changed the right strides",
               "broadcast addition assignment changed the right elements");
}

void test_add_assign_broadcasts_singleton_axes() {
  Tensor left = Tensor::from_data(
      {2, 3, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
                  7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
  const Tensor right = Tensor::from_data({2, 1, 2},
                                         {10.0F, 20.0F, 100.0F, 200.0F});
  const Tensor::storage_type expected_elements{
      11.0F,  22.0F,  13.0F,  24.0F,  15.0F,  26.0F,
      107.0F, 208.0F, 109.0F, 210.0F, 111.0F, 212.0F};

  left += right;

  expect(std::ranges::equal(left.elements(), expected_elements),
         "addition assignment broadcast singleton axes incorrectly");
}

void test_add_assign_broadcasts_trailing_singleton_axis() {
  Tensor left =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor right = Tensor::from_data({2, 1}, {10.0F, 20.0F});
  const Tensor::storage_type expected_elements{11.0F, 12.0F, 13.0F,
                                                24.0F, 25.0F, 26.0F};

  left += right;

  expect(std::ranges::equal(left.elements(), expected_elements),
         "addition assignment broadcast a trailing singleton axis "
         "incorrectly");
}

void test_add_assign_broadcasts_missing_and_singleton_axes() {
  Tensor left = Tensor::from_data(
      {2, 2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
                  7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
  const Tensor right = Tensor::from_data({1, 3}, {10.0F, 20.0F, 30.0F});
  const Tensor::storage_type expected_elements{
      11.0F, 22.0F, 33.0F, 14.0F, 25.0F, 36.0F,
      17.0F, 28.0F, 39.0F, 20.0F, 31.0F, 42.0F};

  left += right;

  expect(std::ranges::equal(left.elements(), expected_elements),
         "addition assignment broadcast missing axes incorrectly");
}

void test_add_assign_broadcasts_rank_zero_right_operand() {
  Tensor left = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  const Tensor right = Tensor::scalar(2.5F);
  const Tensor::storage_type expected_elements{3.5F, 4.5F, 5.5F, 6.5F};

  left += right;

  expect(std::ranges::equal(left.elements(), expected_elements),
         "addition assignment broadcast a rank-zero tensor incorrectly");
}

void test_add_assign_supports_matching_zero_extent_tensors() {
  Tensor left({2, 0, 4});
  const Tensor right({2, 0, 4});
  const Tensor::shape_type expected_shape{2, 0, 4};
  const Tensor::strides_type expected_strides{0, 4, 1};

  left += right;

  expect_state(left, expected_shape, expected_strides, {},
               "zero-extent addition assignment changed shape",
               "zero-extent addition assignment changed strides",
               "zero-extent addition assignment created elements");
}

void test_add_assign_broadcasts_over_zero_extent_axes() {
  Tensor left({2, 0, 3});
  const Tensor lower_rank_right =
      Tensor::from_data({1, 3}, {1.0F, 2.0F, 3.0F});
  const Tensor singleton_axis_right = Tensor::from_data(
      {2, 1, 3}, {4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F});
  const Tensor::shape_type expected_shape{2, 0, 3};
  const Tensor::strides_type expected_strides{0, 3, 1};

  left += lower_rank_right;
  left += singleton_axis_right;

  expect_state(left, expected_shape, expected_strides, {},
               "zero-extent broadcast changed the left shape",
               "zero-extent broadcast changed the left strides",
               "zero-extent broadcast created elements");
}

void test_add_assign_supports_self_aliasing() {
  Tensor tensor = Tensor::from_data({2, 2}, {1.0F, -2.0F, 3.5F, 0.0F});
  const Tensor::storage_type expected_elements{2.0F, -4.0F, 7.0F, 0.0F};

  tensor += tensor;

  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "self addition assignment computed incorrect elements");
}

void test_add_assign_rejects_different_shapes() {
  Tensor left =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor permuted =
      Tensor::from_data({3, 2}, {6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F});
  const Tensor incompatible_trailing_axis =
      Tensor::from_data({2}, {1.0F, 2.0F});

  expect_add_assign_rejected_without_left_change(left, permuted);
  expect_add_assign_rejected_without_left_change(left,
                                                  incompatible_trailing_axis);
}

void test_add_assign_rejects_broadcast_result_that_would_expand_left() {
  Tensor left = Tensor::from_data({2, 1, 3},
                                  {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor zero_extent_right({2, 0, 3});

  expect_add_assign_rejected_without_left_change(left, zero_extent_right);

  Tensor lower_rank_left =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor higher_rank_right = Tensor::from_data(
      {1, 2, 3}, {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});

  expect_add_assign_rejected_without_left_change(lower_rank_left,
                                                  higher_rank_right);
}

void test_add_assign_rejects_moved_from_sentinels() {
  Tensor right_source = Tensor::from_data({1}, {1.0F});
  Tensor right_owner(std::move(right_source));
  static_cast<void>(right_owner);

  Tensor ordinary_left = Tensor::from_data({1}, {2.0F});
  expect_add_assign_rejected_without_left_change(ordinary_left, right_source);

  Tensor left_source = Tensor::from_data({1}, {3.0F});
  Tensor left_owner(std::move(left_source));
  static_cast<void>(left_owner);
  const Tensor ordinary_right = Tensor::from_data({1}, {4.0F});
  expect_add_assign_rejected_without_left_change(left_source, ordinary_right);

  Tensor second_source = Tensor::from_data({1}, {5.0F});
  Tensor second_owner(std::move(second_source));
  static_cast<void>(second_owner);
  expect_add_assign_rejected_without_left_change(left_source, second_source);
}

}  // namespace

int main() {
  try {
    test_add_assign_computes_elementwise_and_returns_left_operand();
    test_add_assign_supports_scalars();
    test_add_assign_broadcasts_lower_rank_right_operand();
    test_add_assign_broadcasts_singleton_axes();
    test_add_assign_broadcasts_trailing_singleton_axis();
    test_add_assign_broadcasts_missing_and_singleton_axes();
    test_add_assign_broadcasts_rank_zero_right_operand();
    test_add_assign_supports_matching_zero_extent_tensors();
    test_add_assign_broadcasts_over_zero_extent_axes();
    test_add_assign_supports_self_aliasing();
    test_add_assign_rejects_different_shapes();
    test_add_assign_rejects_broadcast_result_that_would_expand_left();
    test_add_assign_rejects_moved_from_sentinels();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor addition assignment tests passed\n";
  return EXIT_SUCCESS;
}
