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
concept CanApplySoftmax = requires {
  nn::softmax(std::declval<TensorType>(), std::declval<Tensor::size_type>());
};

template <typename TensorType>
concept CanApplySoftmaxWithoutAxis =
    requires { nn::softmax(std::declval<TensorType>()); };

static_assert(
    std::same_as<decltype(nn::softmax(std::declval<const Tensor&>(),
                                      std::declval<Tensor::size_type>())),
                 Tensor>);
static_assert(CanApplySoftmax<Tensor&>);
static_assert(CanApplySoftmax<const Tensor&>);
static_assert(CanApplySoftmax<Tensor&&>);
static_assert(CanApplySoftmax<const Tensor&&>);
static_assert(CanApplySoftmax<DoubleTensor&>);
static_assert(!CanApplySoftmax<IntegerTensor&>);
static_assert(!CanApplySoftmaxWithoutAxis<Tensor&>);
static_assert(!noexcept(nn::softmax(std::declval<const Tensor&>(),
                                    std::declval<Tensor::size_type>())));

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
  if (!(std::abs(actual - expected) <= tolerance)) {
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

void test_softmax_normalizes_last_axis_without_changing_lvalue() {
  const Tensor tensor =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 3.0F, 2.0F, 1.0F});
  const Tensor::storage_type expected_source{1.0F, 2.0F, 3.0F,
                                             3.0F, 2.0F, 1.0F};
  const Tensor::storage_type expected_result{0.09003057F, 0.24472847F,
                                             0.66524096F, 0.66524096F,
                                             0.24472847F, 0.09003057F};

  const Tensor result = nn::softmax(tensor, 1);

  expect(std::ranges::equal(result.shape(), Tensor::shape_type{2, 3}),
         "softmax changed shape");
  expect(std::ranges::equal(result.strides(), Tensor::strides_type{3, 1}),
         "softmax changed strides");
  expect(result.numel() == tensor.numel(), "softmax changed numel");

  for (Tensor::size_type index = 0; index < result.numel(); ++index) {
    expect_near(result.elements()[index], expected_result[index], 1.0e-6F,
                "last-axis softmax produced an inaccurate element");
  }

  expect_near(result.elements()[0] + result.elements()[1] +
                  result.elements()[2],
              1.0F, 1.0e-6F, "first softmax slice is not normalized");
  expect_near(result.elements()[3] + result.elements()[4] +
                  result.elements()[5],
              1.0F, 1.0e-6F, "second softmax slice is not normalized");
  expect(std::ranges::equal(tensor.elements(), expected_source),
         "softmax changed its tensor lvalue");
  expect(result.data() != tensor.data(),
         "softmax result aliases its tensor lvalue");
}

void test_softmax_normalizes_first_axis() {
  const Tensor tensor =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor::storage_type expected_result{0.04742587F, 0.04742587F,
                                             0.04742587F, 0.95257413F,
                                             0.95257413F, 0.95257413F};

  const Tensor result = nn::softmax(tensor, 0);

  for (Tensor::size_type index = 0; index < result.numel(); ++index) {
    expect_near(result.elements()[index], expected_result[index], 1.0e-6F,
                "first-axis softmax produced an inaccurate element");
  }

  for (Tensor::size_type column = 0; column < 3; ++column) {
    expect_near(result.elements()[column] + result.elements()[3 + column], 1.0F,
                1.0e-6F, "first-axis softmax slice is not normalized");
  }
}

void test_softmax_normalizes_middle_axis() {
  const Tensor tensor =
      Tensor::from_data({2, 3, 2}, {0.0F, 10.0F, 1.0F, 11.0F, 2.0F, 12.0F, 3.0F,
                                    13.0F, 4.0F, 14.0F, 5.0F, 15.0F});
  const Tensor::storage_type expected_result{
      0.09003057F, 0.09003057F, 0.24472847F, 0.24472847F,
      0.66524096F, 0.66524096F, 0.09003057F, 0.09003057F,
      0.24472847F, 0.24472847F, 0.66524096F, 0.66524096F};

  const Tensor result = nn::softmax(tensor, 1);

  expect(std::ranges::equal(result.shape(), Tensor::shape_type{2, 3, 2}),
         "middle-axis softmax changed shape");
  expect(std::ranges::equal(result.strides(), Tensor::strides_type{6, 2, 1}),
         "middle-axis softmax changed strides");

  for (Tensor::size_type index = 0; index < result.numel(); ++index) {
    expect_near(result.elements()[index], expected_result[index], 1.0e-6F,
                "middle-axis softmax produced an inaccurate element");
  }
}

void test_softmax_maps_singleton_axis_to_one() {
  const Tensor tensor =
      Tensor::from_data({2, 1, 2}, {-4.0F, 2.0F, 8.0F, -3.0F});

  const Tensor result = nn::softmax(tensor, 1);

  expect(std::ranges::equal(result.shape(), Tensor::shape_type{2, 1, 2}),
         "singleton-axis softmax changed shape");
  expect(std::ranges::equal(result.strides(), Tensor::strides_type{2, 2, 1}),
         "singleton-axis softmax changed strides");
  expect(std::ranges::all_of(result.elements(),
                             [](float element) { return element == 1.0F; }),
         "singleton-axis softmax did not produce ones");
}

void test_softmax_avoids_exponential_overflow() {
  const float maximum = std::numeric_limits<float>::max();
  const Tensor tensor = Tensor::from_data({2}, {maximum, maximum});

  if ((math_errhandling & MATH_ERREXCEPT) != 0) {
    std::feclearexcept(FE_ALL_EXCEPT);
  }

  const Tensor result = nn::softmax(tensor, 0);

  expect(result.elements()[0] == 0.5F && result.elements()[1] == 0.5F,
         "softmax handled large equal finite values incorrectly");

  if ((math_errhandling & MATH_ERREXCEPT) != 0) {
    expect((std::fetestexcept(FE_OVERFLOW) & FE_OVERFLOW) == 0,
           "softmax overflowed while evaluating finite values");
  }
}

void test_softmax_handles_non_finite_slices_independently() {
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const Tensor tensor = Tensor::from_data(
      {4, 3}, {nan, 0.0F, 1.0F, infinity, 0.0F, -1.0F, -infinity, -infinity,
               -infinity, -infinity, 0.0F, -infinity});

  const Tensor result = nn::softmax(tensor, 1);

  for (Tensor::size_type index = 0; index < 9; ++index) {
    expect(std::isnan(result.elements()[index]),
           "softmax did not map a degenerate slice entirely to NaN");
  }

  expect(result.elements()[9] == 0.0F,
         "softmax handled negative infinity before a finite value incorrectly");
  expect(result.elements()[10] == 1.0F,
         "softmax handled an isolated finite value incorrectly");
  expect(result.elements()[11] == 0.0F,
         "softmax handled negative infinity after a finite value incorrectly");
}

void test_softmax_preserves_zero_extent_tensors() {
  const Tensor selected_axis_empty({2, 0, 3});
  const Tensor other_axis_empty({0, 2, 3});

  const Tensor selected_axis_result = nn::softmax(selected_axis_empty, 1);
  const Tensor other_axis_result = nn::softmax(other_axis_empty, 1);

  expect(std::ranges::equal(selected_axis_result.shape(),
                            Tensor::shape_type{2, 0, 3}),
         "softmax changed a selected zero-extent axis");
  expect(std::ranges::equal(selected_axis_result.strides(),
                            Tensor::strides_type{0, 3, 1}),
         "softmax changed selected-zero-extent strides");
  expect(selected_axis_result.elements().empty(),
         "softmax created elements for a selected zero-extent axis");
  expect(std::ranges::equal(other_axis_result.shape(),
                            Tensor::shape_type{0, 2, 3}),
         "softmax changed an unselected zero-extent axis");
  expect(std::ranges::equal(other_axis_result.strides(),
                            Tensor::strides_type{6, 3, 1}),
         "softmax changed unselected-zero-extent strides");
  expect(other_axis_result.elements().empty(),
         "softmax created elements for an unselected zero-extent axis");
}

void test_softmax_rejects_invalid_axis_without_changing_source() {
  const Tensor rank_zero = Tensor::scalar(2.0F);
  const Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});

  expect_throws<std::out_of_range>(
      [&rank_zero] { static_cast<void>(nn::softmax(rank_zero, 0)); },
      "softmax accepted an axis for a rank-zero tensor",
      "rank-zero softmax produced the wrong exception type");
  expect_throws<std::out_of_range>(
      [&tensor] { static_cast<void>(nn::softmax(tensor, 2)); },
      "softmax accepted an out-of-range axis",
      "invalid-axis softmax produced the wrong exception type");

  expect(rank_zero.at() == 2.0F, "failed rank-zero softmax changed its source");
  expect(std::ranges::equal(tensor.elements(),
                            Tensor::storage_type{1.0F, 2.0F, 3.0F, 4.0F}),
         "failed invalid-axis softmax changed its source");
}

void test_softmax_reuses_rvalue_storage() {
  Tensor source = Tensor::from_data({3}, {1.0F, 2.0F, 3.0F});
  const Tensor::value_type* const original_storage = source.data();

  const Tensor result = nn::softmax(std::move(source), 0);

  expect(result.data() == original_storage,
         "softmax did not reuse rvalue storage");
  expect_near(result.elements()[0], 0.09003057F, 1.0e-6F,
              "rvalue softmax produced an inaccurate first element");
  expect_near(result.elements()[1], 0.24472847F, 1.0e-6F,
              "rvalue softmax produced an inaccurate second element");
  expect_near(result.elements()[2], 0.66524096F, 1.0e-6F,
              "rvalue softmax produced an inaccurate third element");
  expect_empty_sentinel(source);
}

void test_softmax_copies_const_rvalue() {
  const Tensor source = Tensor::from_data({2}, {-1.0F, 1.0F});
  const Tensor::value_type* const original_storage = source.data();

  const Tensor result = nn::softmax(std::move(source), 0);

  expect(result.data() != original_storage,
         "softmax reused const rvalue storage");
  expect(
      std::ranges::equal(source.elements(), Tensor::storage_type{-1.0F, 1.0F}),
      "softmax changed its const rvalue argument");
}

void test_softmax_rejects_moved_from_sentinel_before_validating_axis() {
  Tensor source = Tensor::from_data({1}, {0.0F});
  Tensor owner(std::move(source));
  static_cast<void>(owner);

  expect_throws<std::invalid_argument>(
      [&source] { static_cast<void>(nn::softmax(source, 0)); },
      "softmax accepted an empty sentinel",
      "sentinel softmax produced the wrong exception type");
  expect_empty_sentinel(source);
}

}  // namespace

int main() {
  try {
    test_softmax_normalizes_last_axis_without_changing_lvalue();
    test_softmax_normalizes_first_axis();
    test_softmax_normalizes_middle_axis();
    test_softmax_maps_singleton_axis_to_one();
    test_softmax_avoids_exponential_overflow();
    test_softmax_handles_non_finite_slices_independently();
    test_softmax_preserves_zero_extent_tensors();
    test_softmax_rejects_invalid_axis_without_changing_source();
    test_softmax_reuses_rvalue_storage();
    test_softmax_copies_const_rvalue();
    test_softmax_rejects_moved_from_sentinel_before_validating_axis();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All softmax tests passed\n";
  return EXIT_SUCCESS;
}
