#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

#include "nn/tensor.hpp"

namespace {

using Tensor = nn::Tensor<float>;

using SignedByte = signed char;
using UnsignedByte = unsigned char;
using FixedSignedSpan = std::span<short, 3>;
using DynamicUnsignedSpan = std::span<const unsigned int>;

template <typename TensorType, typename... IndexTypes>
concept CanAccessAtWith =
    requires { std::declval<TensorType>().at(std::declval<IndexTypes>()...); };

static_assert(
    std::same_as<decltype(std::declval<Tensor&>().at()), Tensor::value_type&>);

static_assert(std::same_as<decltype(std::declval<const Tensor&>().at()),
                           const Tensor::value_type&>);

static_assert(
    std::same_as<decltype(std::declval<Tensor&>().at(0)), Tensor::value_type&>);

static_assert(std::same_as<decltype(std::declval<const Tensor&>().at(0)),
                           const Tensor::value_type&>);

static_assert(std::same_as<decltype(std::declval<Tensor&>().at(0, 0)),
                           Tensor::value_type&>);

static_assert(std::same_as<decltype(std::declval<const Tensor&>().at(0, 0)),
                           const Tensor::value_type&>);

static_assert(std::same_as<decltype(std::declval<Tensor&>().at(
                               std::declval<FixedSignedSpan>())),
                           Tensor::value_type&>);

static_assert(std::same_as<decltype(std::declval<const Tensor&>().at(
                               std::declval<FixedSignedSpan>())),
                           const Tensor::value_type&>);

static_assert(std::same_as<decltype(std::declval<Tensor&>().at(
                               std::declval<DynamicUnsignedSpan>())),
                           Tensor::value_type&>);

static_assert(std::same_as<decltype(std::declval<const Tensor&>().at(
                               std::declval<DynamicUnsignedSpan>())),
                           const Tensor::value_type&>);

static_assert(!noexcept(std::declval<Tensor&>().at(0)));
static_assert(!noexcept(std::declval<const Tensor&>().at(0)));

static_assert(
    !noexcept(std::declval<Tensor&>().at(std::declval<FixedSignedSpan>())));

static_assert(!noexcept(
    std::declval<const Tensor&>().at(std::declval<DynamicUnsignedSpan>())));

static_assert(
    CanAccessAtWith<Tensor&, signed char, unsigned short, Tensor::size_type>);

static_assert(
    CanAccessAtWith<const Tensor&, unsigned char, short, unsigned long long>);

static_assert(!CanAccessAtWith<Tensor&&>);
static_assert(!CanAccessAtWith<const Tensor&&>);

static_assert(!CanAccessAtWith<Tensor&&, int>);
static_assert(!CanAccessAtWith<const Tensor&&, int>);

static_assert(!CanAccessAtWith<Tensor&&, FixedSignedSpan>);
static_assert(!CanAccessAtWith<const Tensor&&, DynamicUnsignedSpan>);

static_assert(!CanAccessAtWith<Tensor&, bool>);
static_assert(!CanAccessAtWith<Tensor&, char>);
static_assert(!CanAccessAtWith<Tensor&, wchar_t>);
static_assert(!CanAccessAtWith<Tensor&, char8_t>);
static_assert(!CanAccessAtWith<Tensor&, char16_t>);
static_assert(!CanAccessAtWith<Tensor&, char32_t>);
static_assert(!CanAccessAtWith<Tensor&, int, bool>);
static_assert(!CanAccessAtWith<Tensor&, float>);
static_assert(!CanAccessAtWith<Tensor&, int, double>);
static_assert(!CanAccessAtWith<Tensor&, std::span<const bool, 2>>);
static_assert(!CanAccessAtWith<Tensor&, std::span<const char, 2>>);
static_assert(!CanAccessAtWith<Tensor&, std::span<const wchar_t>>);
static_assert(!CanAccessAtWith<Tensor&, std::span<const float>>);

static_assert(!CanAccessAtWith<const Tensor&, bool>);
static_assert(!CanAccessAtWith<const Tensor&, char>);
static_assert(!CanAccessAtWith<const Tensor&, wchar_t>);
static_assert(!CanAccessAtWith<const Tensor&, char8_t>);
static_assert(!CanAccessAtWith<const Tensor&, char16_t>);
static_assert(!CanAccessAtWith<const Tensor&, char32_t>);
static_assert(!CanAccessAtWith<const Tensor&, int, bool>);
static_assert(!CanAccessAtWith<const Tensor&, float>);
static_assert(!CanAccessAtWith<const Tensor&, int, double>);
static_assert(!CanAccessAtWith<const Tensor&, std::span<const bool, 2>>);
static_assert(!CanAccessAtWith<const Tensor&, std::span<const char, 2>>);
static_assert(!CanAccessAtWith<const Tensor&, std::span<const wchar_t>>);
static_assert(!CanAccessAtWith<const Tensor&, std::span<const float>>);

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

void fill_with_linear_offsets(Tensor& tensor) {
  auto elements = tensor.elements();

  for (Tensor::size_type offset = 0; offset < elements.size(); ++offset) {
    elements[offset] = static_cast<Tensor::value_type>(offset);
  }
}

void test_variadic_checked_access_and_mutation() {
  Tensor tensor({2, 3, 4});
  fill_with_linear_offsets(tensor);

  expect(tensor.at(0, 0, 0) == 0.0F,
         "variadic checked access returned an incorrect first element");

  expect(
      tensor.at(SignedByte{1}, UnsignedByte{1}, Tensor::size_type{3}) == 19.0F,
      "variadic checked access with mixed index types is incorrect");

  tensor.at(1U, 2L, 3) = -7.0F;

  expect(tensor.elements()[23] == -7.0F,
         "variadic checked mutation changed the wrong linear element");

  const Tensor& const_tensor = tensor;

  expect(const_tensor.at(1, 2U, 3L) == -7.0F,
         "const variadic checked access returned an incorrect element");
}

void test_span_checked_access_and_mutation() {
  Tensor tensor({2, 3, 4});
  fill_with_linear_offsets(tensor);

  std::array<short, 3> fixed_values{1, 0, 3};
  const FixedSignedSpan fixed_indices{fixed_values};

  expect(tensor.at(fixed_indices) == 15.0F,
         "fixed span checked access returned an incorrect element");

  const std::array<unsigned int, 3> dynamic_values{0U, 2U, 1U};
  const DynamicUnsignedSpan dynamic_indices{dynamic_values};

  expect(tensor.at(dynamic_indices) == 9.0F,
         "dynamic span checked access returned an incorrect element");

  tensor.at(dynamic_indices) = 91.0F;

  expect(tensor.elements()[9] == 91.0F,
         "span checked mutation changed the wrong linear element");

  const Tensor& const_tensor = tensor;

  expect(const_tensor.at(fixed_indices) == 15.0F,
         "const span checked access returned an incorrect element");
}

void test_scalar_checked_access() {
  Tensor tensor = Tensor::scalar(5.0F);

  expect(tensor.at() == 5.0F,
         "scalar checked access with no indices returned an incorrect value");

  tensor.at() = -3.0F;

  const std::span<short, 0> fixed_empty_indices;
  const std::span<const unsigned int> dynamic_empty_indices;

  expect(tensor.at(fixed_empty_indices) == -3.0F,
         "scalar checked access with a fixed empty span is incorrect");

  tensor.at(dynamic_empty_indices) = 8.0F;

  const Tensor& const_tensor = tensor;

  expect(const_tensor.at() == 8.0F,
         "const scalar checked access with no indices is incorrect");

  expect(const_tensor.at(dynamic_empty_indices) == 8.0F,
         "const scalar checked access with a dynamic empty span is incorrect");
}

void test_rank_mismatch_throws_invalid_argument() {
  Tensor tensor({2, 3});
  const Tensor& const_tensor = tensor;

  expect_throws<std::invalid_argument>(
      [&tensor] { static_cast<void>(tensor.at(-1)); },
      "wrong rank with a negative index did not throw std::invalid_argument",
      "wrong rank with a negative index produced the wrong exception type");

  expect_throws<std::invalid_argument>(
      [&const_tensor] { static_cast<void>(const_tensor.at(0, 0, 0)); },
      "too many variadic indices did not throw std::invalid_argument",
      "too many variadic indices produced the wrong exception type");

  std::array<int, 1> fixed_values{0};
  const std::span<int, 1> fixed_indices{fixed_values};

  expect_throws<std::invalid_argument>(
      [&tensor, fixed_indices] { static_cast<void>(tensor.at(fixed_indices)); },
      "wrong fixed span extent did not throw std::invalid_argument",
      "wrong fixed span extent produced the wrong exception type");

  const std::array<unsigned int, 3> dynamic_values{0U, 0U, 0U};
  const std::span<const unsigned int> dynamic_indices{dynamic_values};

  expect_throws<std::invalid_argument>(
      [&const_tensor, dynamic_indices] {
        static_cast<void>(const_tensor.at(dynamic_indices));
      },
      "wrong dynamic span size did not throw std::invalid_argument",
      "wrong dynamic span size produced the wrong exception type");

  Tensor scalar = Tensor::scalar(1.0F);

  expect_throws<std::invalid_argument>(
      [&scalar] { static_cast<void>(scalar.at(0)); },
      "nonempty scalar indexing did not throw std::invalid_argument",
      "nonempty scalar indexing produced the wrong exception type");
}

void test_out_of_bounds_throws_out_of_range() {
  Tensor tensor({2, 3});
  const Tensor& const_tensor = tensor;

  expect_throws<std::out_of_range>(
      [&tensor] { static_cast<void>(tensor.at(-1, 0)); },
      "negative index did not throw std::out_of_range",
      "negative index produced the wrong exception type");

  expect_throws<std::out_of_range>(
      [&const_tensor] { static_cast<void>(const_tensor.at(2, 0)); },
      "index equal to extent did not throw std::out_of_range",
      "index equal to extent produced the wrong exception type");

  expect_throws<std::out_of_range>(
      [&tensor] { static_cast<void>(tensor.at(0, 4U)); },
      "index greater than extent did not throw std::out_of_range",
      "index greater than extent produced the wrong exception type");

  const Tensor::size_type maximum_index =
      std::numeric_limits<Tensor::size_type>::max();

  expect_throws<std::out_of_range>(
      [&tensor, maximum_index] {
        static_cast<void>(tensor.at(maximum_index, 0));
      },
      "maximum size_type index did not throw std::out_of_range",
      "maximum size_type index produced the wrong exception type");

  std::array<short, 2> negative_values{0, -1};
  const std::span<short, 2> negative_indices{negative_values};

  expect_throws<std::out_of_range>(
      [&tensor, negative_indices] {
        static_cast<void>(tensor.at(negative_indices));
      },
      "negative span index did not throw std::out_of_range",
      "negative span index produced the wrong exception type");

  const std::array<unsigned int, 2> extent_values{0U, 3U};
  const std::span<const unsigned int> extent_indices{extent_values};

  expect_throws<std::out_of_range>(
      [&const_tensor, extent_indices] {
        static_cast<void>(const_tensor.at(extent_indices));
      },
      "span index equal to extent did not throw std::out_of_range",
      "span index equal to extent produced the wrong exception type");

  Tensor zero_extent_tensor({2, 0, 4});

  expect_throws<std::out_of_range>(
      [&zero_extent_tensor] {
        static_cast<void>(zero_extent_tensor.at(0, 0, 0));
      },
      "index into a zero extent did not throw std::out_of_range",
      "index into a zero extent produced the wrong exception type");
}

void test_unrepresentable_index_throws_out_of_range_if_available() {
  if constexpr (std::numeric_limits<std::uintmax_t>::digits >
                std::numeric_limits<Tensor::size_type>::digits) {
    Tensor tensor({1});
    const std::uintmax_t maximum_size_type_value =
        std::numeric_limits<Tensor::size_type>::max();
    const std::uintmax_t index = maximum_size_type_value + std::uintmax_t{1};

    expect_throws<std::out_of_range>(
        [&tensor, index] { static_cast<void>(tensor.at(index)); },
        "index not representable as size_type did not throw std::out_of_range",
        "index not representable as size_type produced the wrong exception "
        "type");
  }
}

void test_failed_checked_access_preserves_state() {
  Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});

  const Tensor::shape_type original_shape(tensor.shape().begin(),
                                          tensor.shape().end());
  const Tensor::strides_type original_strides(tensor.strides().begin(),
                                              tensor.strides().end());
  const Tensor::storage_type original_elements(tensor.elements().begin(),
                                               tensor.elements().end());

  expect_throws<std::invalid_argument>(
      [&tensor] { static_cast<void>(tensor.at(0)); },
      "failed rank check did not throw std::invalid_argument",
      "failed rank check produced the wrong exception type");

  expect_throws<std::out_of_range>(
      [&tensor] { static_cast<void>(tensor.at(0, 2)); },
      "failed bounds check did not throw std::out_of_range",
      "failed bounds check produced the wrong exception type");

  expect(std::ranges::equal(tensor.shape(), original_shape),
         "failed checked access changed the tensor shape");

  expect(std::ranges::equal(tensor.strides(), original_strides),
         "failed checked access changed the tensor strides");

  expect(std::ranges::equal(tensor.elements(), original_elements),
         "failed checked access changed the tensor elements");
}

}  // namespace

int main() {
  try {
    test_variadic_checked_access_and_mutation();
    test_span_checked_access_and_mutation();
    test_scalar_checked_access();
    test_rank_mismatch_throws_invalid_argument();
    test_out_of_bounds_throws_out_of_range();
    test_unrepresentable_index_throws_out_of_range_if_available();
    test_failed_checked_access_preserves_state();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor at tests passed\n";
  return EXIT_SUCCESS;
}
