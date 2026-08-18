#include <algorithm>
#include <concepts>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "nn/tensor.hpp"

namespace {

using Tensor = nn::Tensor<float>;

static_assert(std::same_as<
              decltype(Tensor::full(Tensor::shape_type{2, 3}, 4.0f)), Tensor>);

void expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_full_tensor_creation() {
  const Tensor tensor = Tensor::full({2, 3}, 4.0f);

  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};

  expect(tensor.rank() == 2, "full 2x3 tensor rank must be 2");
  expect(tensor.numel() == 6, "full 2x3 tensor numel must be 6");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "full tensor shape is incorrect");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "full tensor strides are incorrect");

  for (Tensor::value_type value : tensor.elements()) {
    expect(value == 4.0f, "full tensor element does not equal the fill value");
  }
}

void test_full_scalar_creation() {
  const Tensor tensor = Tensor::full({}, -2.0f);

  expect(tensor.rank() == 0, "full scalar rank must be 0");
  expect(tensor.numel() == 1, "full scalar numel must be 1");
  expect(tensor.shape().empty(), "full scalar shape must be empty");
  expect(tensor.strides().empty(), "full scalar strides must be empty");
  expect(tensor.elements()[0] == -2.0f,
         "full scalar element does not equal the fill value");
}

void test_full_zero_extent_creation() {
  const Tensor tensor = Tensor::full({2, 0, 4}, 7.0f);

  const Tensor::shape_type expected_shape{2, 0, 4};
  const Tensor::strides_type expected_strides{0, 4, 1};

  expect(tensor.rank() == 3, "full zero-extent tensor rank must be 3");
  expect(tensor.numel() == 0, "full zero-extent tensor numel must be 0");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "full zero-extent tensor shape is incorrect");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "full zero-extent tensor strides are incorrect");
  expect(tensor.elements().empty(),
         "full zero-extent tensor elements must be empty");
}

void test_full_shape_product_overflow() {
  const Tensor::size_type max_size =
      std::numeric_limits<Tensor::size_type>::max();

  try {
    static_cast<void>(Tensor::full({max_size, 2}, 1.0f));
  } catch (const std::overflow_error&) {
    return;
  } catch (...) {
    throw std::runtime_error(
        "full shape overflow produced the wrong exception type");
  }

  throw std::runtime_error(
      "full shape overflow did not throw std::overflow_error");
}

}  // namespace

int main() {
  try {
    test_full_tensor_creation();
    test_full_scalar_creation();
    test_full_zero_extent_creation();
    test_full_shape_product_overflow();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor creation tests passed\n";
  return EXIT_SUCCESS;
}