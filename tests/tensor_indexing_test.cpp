#include <array>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <utility>

#include "nn/tensor.hpp"

namespace {

using Tensor = nn::Tensor<float>;

using FixedSignedSpan = std::span<short, 2>;
using DynamicUnsignedSpan = std::span<const unsigned int>;

template <typename TensorType, typename... IndexTypes>
concept CanIndexWith =
    requires { std::declval<TensorType>()[std::declval<IndexTypes>()...]; };

static_assert(
    std::same_as<decltype(std::declval<Tensor&>()[0]), Tensor::value_type&>);

static_assert(std::same_as<decltype(std::declval<const Tensor&>()[0]),
                           const Tensor::value_type&>);

static_assert(
    std::same_as<decltype(std::declval<Tensor&>()[0, 0]), Tensor::value_type&>);

static_assert(std::same_as<decltype(std::declval<const Tensor&>()[0, 0]),
                           const Tensor::value_type&>);

static_assert(
    std::same_as<decltype(std::declval<Tensor&>()[]), Tensor::value_type&>);

static_assert(std::same_as<decltype(std::declval<const Tensor&>()[]),
                           const Tensor::value_type&>);

static_assert(
    std::same_as<
        decltype(std::declval<Tensor&>()[std::declval<FixedSignedSpan>()]),
        Tensor::value_type&>);

static_assert(
    std::same_as<decltype(std::declval<
                          const Tensor&>()[std::declval<FixedSignedSpan>()]),
                 const Tensor::value_type&>);

static_assert(
    std::same_as<
        decltype(std::declval<Tensor&>()[std::declval<DynamicUnsignedSpan>()]),
        Tensor::value_type&>);

static_assert(std::same_as<decltype(std::declval<const Tensor&>()
                                        [std::declval<DynamicUnsignedSpan>()]),
                           const Tensor::value_type&>);

static_assert(noexcept(std::declval<Tensor&>()[0]));
static_assert(noexcept(std::declval<const Tensor&>()[0]));

static_assert(noexcept(std::declval<Tensor&>()[0, 0]));
static_assert(noexcept(std::declval<const Tensor&>()[0, 0]));

static_assert(noexcept(std::declval<Tensor&>()[]));
static_assert(noexcept(std::declval<const Tensor&>()[]));

static_assert(
    noexcept(std::declval<Tensor&>()[std::declval<FixedSignedSpan>()]));

static_assert(
    noexcept(std::declval<const Tensor&>()[std::declval<FixedSignedSpan>()]));

static_assert(
    noexcept(std::declval<Tensor&>()[std::declval<DynamicUnsignedSpan>()]));

static_assert(noexcept(
    std::declval<const Tensor&>()[std::declval<DynamicUnsignedSpan>()]));

static_assert(!CanIndexWith<Tensor&&, int>);
static_assert(!CanIndexWith<const Tensor&&, int>);

static_assert(!CanIndexWith<Tensor&&, int, int>);
static_assert(!CanIndexWith<const Tensor&&, int, int>);

static_assert(!CanIndexWith<Tensor&&>);
static_assert(!CanIndexWith<const Tensor&&>);

static_assert(!CanIndexWith<Tensor&&, FixedSignedSpan>);
static_assert(!CanIndexWith<const Tensor&&, FixedSignedSpan>);

static_assert(!CanIndexWith<Tensor&&, DynamicUnsignedSpan>);
static_assert(!CanIndexWith<const Tensor&&, DynamicUnsignedSpan>);

static_assert(!CanIndexWith<Tensor&, bool>);
static_assert(!CanIndexWith<Tensor&, char>);
static_assert(!CanIndexWith<Tensor&, wchar_t>);
static_assert(!CanIndexWith<Tensor&, char8_t>);
static_assert(!CanIndexWith<Tensor&, char16_t>);
static_assert(!CanIndexWith<Tensor&, char32_t>);
static_assert(!CanIndexWith<Tensor&, int, bool>);
static_assert(!CanIndexWith<Tensor&, float>);
static_assert(!CanIndexWith<Tensor&, int, double>);
static_assert(!CanIndexWith<Tensor&, std::span<const bool, 2>>);
static_assert(!CanIndexWith<Tensor&, std::span<const char, 2>>);
static_assert(!CanIndexWith<Tensor&, std::span<const wchar_t>>);
static_assert(!CanIndexWith<Tensor&, std::span<const float>>);

static_assert(!CanIndexWith<const Tensor&, bool>);
static_assert(!CanIndexWith<const Tensor&, char>);
static_assert(!CanIndexWith<const Tensor&, wchar_t>);
static_assert(!CanIndexWith<const Tensor&, char8_t>);
static_assert(!CanIndexWith<const Tensor&, char16_t>);
static_assert(!CanIndexWith<const Tensor&, char32_t>);
static_assert(!CanIndexWith<const Tensor&, int, bool>);
static_assert(!CanIndexWith<const Tensor&, float>);
static_assert(!CanIndexWith<const Tensor&, int, double>);
static_assert(!CanIndexWith<const Tensor&, std::span<const bool, 2>>);
static_assert(!CanIndexWith<const Tensor&, std::span<const char, 2>>);
static_assert(!CanIndexWith<const Tensor&, std::span<const wchar_t>>);
static_assert(!CanIndexWith<const Tensor&, std::span<const float>>);

void expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void fill_with_linear_offsets(Tensor& tensor) {
  auto elements = tensor.elements();

  for (Tensor::size_type offset = 0; offset < elements.size(); ++offset) {
    elements[offset] = static_cast<Tensor::value_type>(offset);
  }
}

void test_rank_one_indexing() {
  const Tensor tensor = Tensor::from_data({4}, {1.0f, 2.0f, 3.0f, 4.0f});

  expect(tensor[0] == 1.0f, "rank-1 first element is incorrect");
  expect(tensor[3] == 4.0f, "rank-1 last element is incorrect");
}

void test_rank_two_indexing_and_mutation() {
  Tensor tensor =
      Tensor::from_data({2, 3}, {10.0f, 11.0f, 12.0f, 20.0f, 21.0f, 22.0f});

  expect(tensor[0, 0] == 10.0f, "2D element [0, 0] is incorrect");
  expect(tensor[0, 2] == 12.0f, "2D element [0, 2] is incorrect");
  expect(tensor[1, 0] == 20.0f, "2D element [1, 0] is incorrect");
  expect(tensor[1, 2] == 22.0f, "2D element [1, 2] is incorrect");

  expect(tensor[1U, 1L] == 21.0f,
         "2D mixed signed and unsigned indexing is incorrect");

  tensor[1, 1U] = 42.0f;

  expect(tensor.elements()[4] == 42.0f,
         "2D write changed the wrong linear element");

  const Tensor& const_tensor = tensor;

  expect(const_tensor[1, 1] == 42.0f,
         "const 2D access returned an incorrect value");
}

void test_rank_three_row_major_indexing() {
  Tensor tensor({2, 3, 4});
  fill_with_linear_offsets(tensor);

  const Tensor& const_tensor = tensor;

  expect(const_tensor[0, 0, 0] == 0.0f,
         "3D row-major offset for [0, 0, 0] is incorrect");

  expect(const_tensor[0, 2, 3] == 11.0f,
         "3D row-major offset for [0, 2, 3] is incorrect");

  expect(const_tensor[1, 0, 0] == 12.0f,
         "3D row-major offset for [1, 0, 0] is incorrect");

  expect(const_tensor[1, 2, 3] == 23.0f,
         "3D row-major offset for [1, 2, 3] is incorrect");

  expect(const_tensor[1U, 1L, std::size_t{3}] == 19.0f,
         "3D mixed index types produced an incorrect offset");
}

void test_span_indexing_and_mutation() {
  Tensor tensor({2, 3, 4});
  fill_with_linear_offsets(tensor);

  std::array<short, 3> fixed_values{1, 0, 3};
  const std::span<short, 3> fixed_indices{fixed_values};

  expect(tensor[fixed_indices] == 15.0f,
         "fixed signed span indexing is incorrect");

  const std::array<unsigned int, 3> dynamic_values{0U, 2U, 1U};
  const std::span<const unsigned int> dynamic_indices{dynamic_values};

  expect(tensor[dynamic_indices] == 9.0f,
         "dynamic unsigned span indexing is incorrect");

  tensor[dynamic_indices] = 91.0f;

  expect(tensor.elements()[9] == 91.0f,
         "write through a span changed the wrong linear element");

  const Tensor& const_tensor = tensor;

  expect(const_tensor[fixed_indices] == 15.0f,
         "const span access returned an incorrect value");
}

void test_scalar_indexing() {
  Tensor tensor = Tensor::scalar(7.0f);

  expect(tensor[] == 7.0f, "scalar indexing returned an incorrect value");

  tensor[] = -3.0f;

  const Tensor& const_tensor = tensor;

  expect(const_tensor[] == -3.0f,
         "scalar write through empty indexing was not preserved");
}

void test_scalar_empty_span_indexing() {
  Tensor tensor = Tensor::scalar(5.0f);

  const std::span<short, 0> fixed_empty_indices;
  const std::span<const unsigned int> dynamic_empty_indices;

  expect(tensor[fixed_empty_indices] == 5.0f,
         "scalar indexing with a fixed empty span is incorrect");

  tensor[dynamic_empty_indices] = -8.0f;

  const Tensor& const_tensor = tensor;

  expect(const_tensor[dynamic_empty_indices] == -8.0f,
         "scalar write through a dynamic empty span was not preserved");
}

}  // namespace

int main() {
  try {
    test_rank_one_indexing();
    test_rank_two_indexing_and_mutation();
    test_rank_three_row_major_indexing();
    test_span_indexing_and_mutation();
    test_scalar_indexing();
    test_scalar_empty_span_indexing();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor indexing tests passed\n";
  return EXIT_SUCCESS;
}
