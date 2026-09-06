#include <algorithm>
#include <concepts>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "nn/tensor.hpp"

namespace {

using Tensor = nn::Tensor<float>;

static_assert(std::copy_constructible<Tensor>);
static_assert(std::is_copy_assignable_v<Tensor>);

void expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_copy_construction_preserves_value_and_ownership() {
  const Tensor::shape_type expected_shape{2, 2};
  const Tensor::strides_type expected_strides{2, 1};
  const Tensor::storage_type expected_elements{1.0F, 2.0F, 3.0F, 4.0F};

  Tensor source = Tensor::from_data(expected_shape, expected_elements);
  Tensor copy(source);

  expect(std::ranges::equal(copy.shape(), expected_shape),
         "copy construction did not preserve shape");
  expect(std::ranges::equal(copy.strides(), expected_strides),
         "copy construction did not preserve strides");
  expect(std::ranges::equal(copy.elements(), expected_elements),
         "copy construction did not preserve elements");

  copy.at(0, 0) = 9.0F;

  expect(source.at(0, 0) == 1.0F,
         "copy construction shared mutable element storage");
}

void test_copy_construction_preserves_scalar_and_zero_extent_states() {
  const Tensor scalar = Tensor::scalar(3.0F);
  const Tensor scalar_copy(scalar);

  expect(scalar_copy.rank() == 0, "scalar copy has an incorrect rank");
  expect(scalar_copy.numel() == 1, "scalar copy has an incorrect numel");
  expect(scalar_copy.at() == 3.0F, "scalar copy has an incorrect value");

  const Tensor zero_extent({2, 0, 4});
  const Tensor zero_extent_copy(zero_extent);
  const Tensor::shape_type expected_shape{2, 0, 4};
  const Tensor::strides_type expected_strides{0, 4, 1};

  expect(std::ranges::equal(zero_extent_copy.shape(), expected_shape),
         "zero-extent copy has an incorrect shape");
  expect(std::ranges::equal(zero_extent_copy.strides(), expected_strides),
         "zero-extent copy has incorrect strides");
  expect(zero_extent_copy.numel() == 0,
         "zero-extent copy has an incorrect numel");
}

void test_copy_assignment_replaces_complete_state_and_owns_storage() {
  const Tensor::shape_type expected_shape{2, 2};
  const Tensor::strides_type expected_strides{2, 1};
  const Tensor::storage_type expected_elements{5.0F, 6.0F, 7.0F, 8.0F};

  Tensor source = Tensor::from_data(expected_shape, expected_elements);
  Tensor target = Tensor::scalar(-1.0F);

  target = source;

  expect(std::ranges::equal(target.shape(), expected_shape),
         "copy assignment did not replace shape");
  expect(std::ranges::equal(target.strides(), expected_strides),
         "copy assignment did not replace strides");
  expect(std::ranges::equal(target.elements(), expected_elements),
         "copy assignment did not replace elements");

  target.at(1, 1) = 42.0F;

  expect(source.at(1, 1) == 8.0F,
         "copy assignment shared mutable element storage");
}

void test_self_copy_assignment_preserves_state() {
  Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  const Tensor::shape_type expected_shape(tensor.shape().begin(),
                                           tensor.shape().end());
  const Tensor::strides_type expected_strides(tensor.strides().begin(),
                                               tensor.strides().end());
  const Tensor::storage_type expected_elements(tensor.elements().begin(),
                                                tensor.elements().end());
  const Tensor& alias = tensor;

  tensor = alias;

  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "self-copy assignment changed shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "self-copy assignment changed strides");
  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "self-copy assignment changed elements");
}

class CopyFailure : public std::exception {
 public:
  [[nodiscard]] const char* what() const noexcept override {
    return "test copy failure";
  }
};

class ThrowingValue {
 public:
  ThrowingValue() = default;
  explicit ThrowingValue(int value) : value_(value) {}

  ThrowingValue(const ThrowingValue& other) : value_(other.value_) {
    throw_if_copying_is_enabled();
  }

  ThrowingValue& operator=(const ThrowingValue& other) {
    throw_if_copying_is_enabled();
    value_ = other.value_;
    return *this;
  }

  ThrowingValue(ThrowingValue&&) noexcept = default;
  ThrowingValue& operator=(ThrowingValue&&) noexcept = default;

  [[nodiscard]] int value() const noexcept { return value_; }

  static void enable_copy_failure() noexcept { copy_failure_enabled_ = true; }

  static void disable_copy_failure() noexcept { copy_failure_enabled_ = false; }

 private:
  static void throw_if_copying_is_enabled() {
    if (copy_failure_enabled_) {
      throw CopyFailure{};
    }
  }

  int value_{0};
  inline static bool copy_failure_enabled_{false};
};

void test_copy_assignment_has_strong_exception_guarantee() {
  using ThrowingTensor = nn::Tensor<ThrowingValue>;

  ThrowingTensor source({2, 2});
  source.elements()[0] = ThrowingValue{1};
  source.elements()[1] = ThrowingValue{2};
  source.elements()[2] = ThrowingValue{3};
  source.elements()[3] = ThrowingValue{4};

  ThrowingTensor target({1});
  target.elements()[0] = ThrowingValue{9};

  const ThrowingTensor::shape_type expected_shape{1};
  const ThrowingTensor::strides_type expected_strides{1};

  bool did_throw = false;
  ThrowingValue::enable_copy_failure();

  try {
    target = source;
  } catch (const CopyFailure&) {
    did_throw = true;
  } catch (...) {
    ThrowingValue::disable_copy_failure();
    throw;
  }

  ThrowingValue::disable_copy_failure();

  expect(did_throw, "copy assignment did not propagate element copy failure");
  expect(std::ranges::equal(target.shape(), expected_shape),
         "failed copy assignment changed target shape");
  expect(std::ranges::equal(target.strides(), expected_strides),
         "failed copy assignment changed target strides");
  expect(target.numel() == 1,
         "failed copy assignment changed target element count");
  expect(target.elements()[0].value() == 9,
         "failed copy assignment changed target elements");
}

}  // namespace

int main() {
  try {
    test_copy_construction_preserves_value_and_ownership();
    test_copy_construction_preserves_scalar_and_zero_extent_states();
    test_copy_assignment_replaces_complete_state_and_owns_storage();
    test_self_copy_assignment_preserves_state();
    test_copy_assignment_has_strong_exception_guarantee();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor copy semantics tests passed\n";
  return EXIT_SUCCESS;
}
