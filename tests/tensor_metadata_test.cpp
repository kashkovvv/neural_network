#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

#include "nn/tensor.hpp"

namespace {

using Tensor = nn::Tensor<float>;

static_assert(noexcept(std::declval<const Tensor&>().rank()));
static_assert(noexcept(std::declval<const Tensor&>().numel()));
static_assert(noexcept(std::declval<const Tensor&>().shape()));
static_assert(noexcept(std::declval<const Tensor&>().strides()));

void expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_scalar_metadata() {
  const Tensor tensor({});

  expect(tensor.rank() == 0, "scalar rank must be 0");
  expect(tensor.numel() == 1, "scalar numel must be 1");
  expect(tensor.shape().empty(), "scalar shape must be empty");
  expect(tensor.strides().empty(), "scalar strides must be empty");
}

void test_single_element_vector_metadata() {
  const Tensor tensor({1});

  const Tensor::shape_type expected_shape{1};
  const Tensor::strides_type expected_strides{1};

  expect(tensor.rank() == 1, "single-element vector rank must be 1");
  expect(tensor.numel() == 1, "single-element vector numel must be 1");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "single-element vector shape is incorrect");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "single-element vector strides are incorrect");
}

void test_vector_metadata() {
  const Tensor tensor({5});

  const Tensor::shape_type expected_shape{5};
  const Tensor::strides_type expected_strides{1};

  expect(tensor.rank() == 1, "vector rank must be 1");
  expect(tensor.numel() == 5, "vector numel must be 5");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "vector shape is incorrect");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "vector strides are incorrect");
}

void test_three_dimensional_metadata() {
  const Tensor tensor({2, 3, 4});

  const Tensor::shape_type expected_shape{2, 3, 4};
  const Tensor::strides_type expected_strides{12, 4, 1};

  expect(tensor.rank() == 3, "3D tensor rank must be 3");
  expect(tensor.numel() == 24, "3D tensor numel must be 24");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "3D tensor shape is incorrect");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "3D tensor strides are incorrect");
}

void test_singleton_axis_metadata() {
  const Tensor tensor({2, 1, 4});

  const Tensor::shape_type expected_shape{2, 1, 4};
  const Tensor::strides_type expected_strides{4, 4, 1};

  expect(tensor.rank() == 3, "singleton-axis tensor rank must be 3");
  expect(tensor.numel() == 8, "singleton-axis tensor numel must be 8");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "singleton-axis tensor shape is incorrect");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "singleton-axis tensor strides are incorrect");
}

void test_zero_length_vector_metadata() {
  const Tensor tensor({0});

  const Tensor::shape_type expected_shape{0};
  const Tensor::strides_type expected_strides{1};

  expect(tensor.rank() == 1, "zero-length vector rank must be 1");
  expect(tensor.numel() == 0, "zero-length vector numel must be 0");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "zero-length vector shape is incorrect");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "zero-length vector strides are incorrect");
}

void test_zero_extent_metadata() {
  const Tensor tensor({2, 0, 4});

  const Tensor::shape_type expected_shape{2, 0, 4};
  const Tensor::strides_type expected_strides{0, 4, 1};

  expect(tensor.rank() == 3, "zero-extent tensor rank must be 3");
  expect(tensor.numel() == 0, "zero-extent tensor numel must be 0");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "zero-extent tensor shape is incorrect");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "zero-extent tensor strides are incorrect");
}

void test_shape_ownership() {
  const Tensor::shape_type expected_shape{2, 3};
  Tensor::shape_type external_shape = expected_shape;

  const Tensor tensor(external_shape);

  external_shape.clear();

  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "tensor must own an independent copy of its shape");
}

void test_shape_product_overflow() {
  const Tensor::size_type max_size =
      std::numeric_limits<Tensor::size_type>::max();

  try {
    static_cast<void>(Tensor({max_size, 2}));
  } catch (const std::overflow_error&) {
    return;
  } catch (...) {
    throw std::runtime_error("overflow produced the wrong exception type");
  }

  throw std::runtime_error("overflowing shape must throw std::overflow_error");
}

}  // namespace

int main() {
  try {
    test_scalar_metadata();
    test_single_element_vector_metadata();
    test_vector_metadata();
    test_three_dimensional_metadata();
    test_singleton_axis_metadata();
    test_zero_length_vector_metadata();
    test_zero_extent_metadata();
    test_shape_ownership();
    test_shape_product_overflow();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor metadata tests passed\n";
  return EXIT_SUCCESS;
}
