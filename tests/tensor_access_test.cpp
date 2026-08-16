#include <algorithm>
#include <array>
#include <concepts>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <utility>

#include "nn/tensor.hpp"

namespace {

using Tensor = nn::Tensor<float>;
using MetadataSpan = std::span<const Tensor::size_type>;
using MutableElementsSpan = std::span<Tensor::value_type>;
using ConstElementsSpan = std::span<const Tensor::value_type>;

static_assert(
    std::same_as<decltype(std::declval<Tensor&>().shape()), MetadataSpan>);
static_assert(std::same_as<decltype(std::declval<const Tensor&>().shape()),
                           MetadataSpan>);

static_assert(
    std::same_as<decltype(std::declval<Tensor&>().strides()), MetadataSpan>);
static_assert(std::same_as<decltype(std::declval<const Tensor&>().strides()),
                           MetadataSpan>);

static_assert(std::same_as<decltype(std::declval<Tensor&>().elements()),
                           MutableElementsSpan>);
static_assert(std::same_as<decltype(std::declval<const Tensor&>().elements()),
                           ConstElementsSpan>);

static_assert(noexcept(std::declval<const Tensor&>().shape()));
static_assert(noexcept(std::declval<const Tensor&>().strides()));
static_assert(noexcept(std::declval<Tensor&>().elements()));
static_assert(noexcept(std::declval<const Tensor&>().elements()));

template <typename U>
concept CanCallShape = requires { std::declval<U>().shape(); };

static_assert(CanCallShape<Tensor&>);
static_assert(CanCallShape<const Tensor&>);
static_assert(!CanCallShape<Tensor&&>);
static_assert(!CanCallShape<const Tensor&&>);

template <typename U>
concept CanCallStrides = requires { std::declval<U>().strides(); };

static_assert(CanCallStrides<Tensor&>);
static_assert(CanCallStrides<const Tensor&>);
static_assert(!CanCallStrides<Tensor&&>);
static_assert(!CanCallStrides<const Tensor&&>);

template <typename U>
concept CanCallElements = requires { std::declval<U>().elements(); };

static_assert(CanCallElements<Tensor&>);
static_assert(CanCallElements<const Tensor&>);
static_assert(!CanCallElements<Tensor&&>);
static_assert(!CanCallElements<const Tensor&&>);

void expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_mutable_elements_access() {
  Tensor tensor({2, 3});
  auto elements = tensor.elements();

  expect(elements.size() == 6, "2x3 tensor elements size must be 6");
  expect(elements.size() == tensor.numel(),
         "elements size must equal tensor numel");

  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const std::array<float, 6> expected_values{1.0f, 2.0f, 3.0f,
                                             4.0f, 5.0f, 6.0f};

  for (Tensor::size_type index = 0; index < elements.size(); ++index) {
    elements[index] = expected_values[index];
  }

  const Tensor& const_tensor = tensor;
  const auto const_elements = const_tensor.elements();

  expect(const_elements.size() == const_tensor.numel(),
         "const elements size must equal tensor numel");
  expect(std::ranges::equal(const_elements, expected_values),
         "values read through const elements view are incorrect");
  expect(tensor.rank() == 2,
         "tensor rank changed after modifying its elements");
  expect(tensor.numel() == 6,
         "tensor numel changed after modifying its elements");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "tensor shape changed after modifying its elements");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "tensor strides changed after modifying its elements");
}

void test_scalar_elements_access() {
  Tensor tensor({});
  auto elements = tensor.elements();

  expect(tensor.rank() == 0, "scalar rank must be 0");
  expect(tensor.numel() == 1, "scalar numel must be 1");
  expect(tensor.shape().empty(), "scalar shape must be empty");
  expect(tensor.strides().empty(), "scalar strides must be empty");
  expect(elements.size() == tensor.numel(),
         "scalar elements size must equal scalar numel");
  expect(!elements.empty(), "scalar elements view must not be empty");

  elements[0] = 42.0f;

  const Tensor& const_tensor = tensor;
  const auto const_elements = const_tensor.elements();

  expect(const_elements.size() == const_tensor.numel(),
         "const scalar elements size must equal scalar numel");
  expect(const_elements[0] == 42.0f,
         "scalar value written through mutable view was not preserved");
}

void test_zero_extent_elements_access() {
  Tensor tensor({2, 0, 4});
  const auto elements = tensor.elements();

  expect(tensor.rank() == 3, "zero-extent tensor rank must be 3");
  expect(tensor.numel() == 0, "zero-extent tensor numel must be 0");
  expect(tensor.shape().size() == 3, "zero-extent tensor shape size must be 3");
  expect(tensor.strides().size() == 3,
         "zero-extent tensor strides size must be 3");
  expect(elements.size() == tensor.numel(),
         "zero-extent elements size must equal tensor numel");
  expect(elements.empty(), "zero-extent elements view must be empty");

  const Tensor& const_tensor = tensor;
  const auto const_elements = const_tensor.elements();

  expect(const_elements.size() == const_tensor.numel(),
         "const zero-extent elements size must equal tensor numel");
  expect(const_elements.empty(),
         "const zero-extent elements view must be empty");
}

}  // namespace

int main() {
  try {
    test_mutable_elements_access();
    test_scalar_elements_access();
    test_zero_extent_elements_access();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor access tests passed\n";
  return EXIT_SUCCESS;
}