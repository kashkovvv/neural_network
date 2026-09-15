#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <iterator>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <utility>

#include "nn/tensor.hpp"

namespace {

using Tensor = nn::Tensor<float>;

template <typename SourceType>
concept CanCallBegin = requires { std::declval<SourceType>().begin(); };

template <typename SourceType>
concept CanCallEnd = requires { std::declval<SourceType>().end(); };

template <typename SourceType>
concept CanCallCbegin = requires { std::declval<SourceType>().cbegin(); };

template <typename SourceType>
concept CanCallCend = requires { std::declval<SourceType>().cend(); };

template <typename SourceType>
concept CanCallRbegin = requires { std::declval<SourceType>().rbegin(); };

template <typename SourceType>
concept CanCallRend = requires { std::declval<SourceType>().rend(); };

template <typename SourceType>
concept CanCallCrbegin = requires { std::declval<SourceType>().crbegin(); };

template <typename SourceType>
concept CanCallCrend = requires { std::declval<SourceType>().crend(); };

static_assert(std::same_as<Tensor::difference_type, std::ptrdiff_t>);
static_assert(std::same_as<Tensor::reference, float&>);
static_assert(std::same_as<Tensor::const_reference, const float&>);
static_assert(std::same_as<Tensor::iterator, float*>);
static_assert(std::same_as<Tensor::const_iterator, const float*>);
static_assert(
    std::same_as<Tensor::reverse_iterator, std::reverse_iterator<float*>>);
static_assert(std::same_as<Tensor::const_reverse_iterator,
                           std::reverse_iterator<const float*>>);

static_assert(std::contiguous_iterator<Tensor::iterator>);
static_assert(std::contiguous_iterator<Tensor::const_iterator>);
static_assert(std::random_access_iterator<Tensor::reverse_iterator>);
static_assert(std::random_access_iterator<Tensor::const_reverse_iterator>);

static_assert(std::ranges::range<Tensor>);
static_assert(std::ranges::range<const Tensor>);
static_assert(std::ranges::common_range<Tensor>);
static_assert(std::ranges::common_range<const Tensor>);
static_assert(std::ranges::sized_range<Tensor>);
static_assert(std::ranges::sized_range<const Tensor>);
static_assert(std::ranges::random_access_range<Tensor>);
static_assert(std::ranges::random_access_range<const Tensor>);
static_assert(std::ranges::contiguous_range<Tensor>);
static_assert(std::ranges::contiguous_range<const Tensor>);
static_assert(std::ranges::output_range<Tensor, float>);
static_assert(!std::ranges::output_range<const Tensor, float>);
static_assert(!std::ranges::borrowed_range<Tensor>);
static_assert(!std::ranges::borrowed_range<const Tensor>);

static_assert(CanCallBegin<Tensor&>);
static_assert(CanCallBegin<const Tensor&>);
static_assert(!CanCallBegin<Tensor&&>);
static_assert(!CanCallBegin<const Tensor&&>);
static_assert(CanCallEnd<Tensor&>);
static_assert(CanCallEnd<const Tensor&>);
static_assert(!CanCallEnd<Tensor&&>);
static_assert(!CanCallEnd<const Tensor&&>);

static_assert(CanCallCbegin<Tensor&>);
static_assert(CanCallCbegin<const Tensor&>);
static_assert(!CanCallCbegin<Tensor&&>);
static_assert(!CanCallCbegin<const Tensor&&>);
static_assert(CanCallCend<Tensor&>);
static_assert(CanCallCend<const Tensor&>);
static_assert(!CanCallCend<Tensor&&>);
static_assert(!CanCallCend<const Tensor&&>);

static_assert(CanCallRbegin<Tensor&>);
static_assert(CanCallRbegin<const Tensor&>);
static_assert(!CanCallRbegin<Tensor&&>);
static_assert(!CanCallRbegin<const Tensor&&>);
static_assert(CanCallRend<Tensor&>);
static_assert(CanCallRend<const Tensor&>);
static_assert(!CanCallRend<Tensor&&>);
static_assert(!CanCallRend<const Tensor&&>);

static_assert(CanCallCrbegin<Tensor&>);
static_assert(CanCallCrbegin<const Tensor&>);
static_assert(!CanCallCrbegin<Tensor&&>);
static_assert(!CanCallCrbegin<const Tensor&&>);
static_assert(CanCallCrend<Tensor&>);
static_assert(CanCallCrend<const Tensor&>);
static_assert(!CanCallCrend<Tensor&&>);
static_assert(!CanCallCrend<const Tensor&&>);

static_assert(
    std::same_as<decltype(std::declval<Tensor&>().begin()), Tensor::iterator>);
static_assert(std::same_as<decltype(std::declval<const Tensor&>().begin()),
                           Tensor::const_iterator>);
static_assert(
    std::same_as<decltype(std::declval<Tensor&>().end()), Tensor::iterator>);
static_assert(std::same_as<decltype(std::declval<const Tensor&>().end()),
                           Tensor::const_iterator>);
static_assert(std::same_as<decltype(std::declval<const Tensor&>().cbegin()),
                           Tensor::const_iterator>);
static_assert(std::same_as<decltype(std::declval<const Tensor&>().cend()),
                           Tensor::const_iterator>);
static_assert(std::same_as<decltype(std::declval<Tensor&>().rbegin()),
                           Tensor::reverse_iterator>);
static_assert(std::same_as<decltype(std::declval<const Tensor&>().rbegin()),
                           Tensor::const_reverse_iterator>);
static_assert(std::same_as<decltype(std::declval<Tensor&>().rend()),
                           Tensor::reverse_iterator>);
static_assert(std::same_as<decltype(std::declval<const Tensor&>().rend()),
                           Tensor::const_reverse_iterator>);
static_assert(std::same_as<decltype(std::declval<const Tensor&>().crbegin()),
                           Tensor::const_reverse_iterator>);
static_assert(std::same_as<decltype(std::declval<const Tensor&>().crend()),
                           Tensor::const_reverse_iterator>);

static_assert(noexcept(std::declval<Tensor&>().begin()));
static_assert(noexcept(std::declval<const Tensor&>().begin()));
static_assert(noexcept(std::declval<Tensor&>().end()));
static_assert(noexcept(std::declval<const Tensor&>().end()));
static_assert(noexcept(std::declval<const Tensor&>().cbegin()));
static_assert(noexcept(std::declval<const Tensor&>().cend()));
static_assert(noexcept(std::declval<Tensor&>().rbegin()));
static_assert(noexcept(std::declval<const Tensor&>().rbegin()));
static_assert(noexcept(std::declval<Tensor&>().rend()));
static_assert(noexcept(std::declval<const Tensor&>().rend()));
static_assert(noexcept(std::declval<const Tensor&>().crbegin()));
static_assert(noexcept(std::declval<const Tensor&>().crend()));

void expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_forward_iteration_exposes_contiguous_storage() {
  Tensor tensor =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const std::array<float, 6> expected_values{1.0F, 2.0F, 3.0F,
                                             4.0F, 5.0F, 6.0F};

  expect(tensor.begin() == tensor.data(),
         "tensor begin does not point to its contiguous storage");
  expect(std::to_address(tensor.begin()) == tensor.data(),
         "tensor iterator address does not match tensor data");
  expect(tensor.end() - tensor.begin() ==
             static_cast<Tensor::difference_type>(tensor.numel()),
         "forward iterator distance does not match tensor numel");
  expect(std::ranges::equal(tensor, expected_values),
         "forward iteration does not follow row-major storage order");

  tensor.begin()[2] = 30.0F;

  expect(tensor.at(0, 2) == 30.0F,
         "write through tensor iterator did not change the tensor");
}

void test_const_iteration_exposes_read_only_elements() {
  Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  const Tensor& const_tensor = tensor;
  const std::array<float, 4> expected_values{1.0F, 2.0F, 3.0F, 4.0F};

  expect(const_tensor.begin() == const_tensor.cbegin(),
         "const begin and cbegin do not match");
  expect(const_tensor.end() == const_tensor.cend(),
         "const end and cend do not match");
  expect(std::ranges::data(const_tensor) == const_tensor.data(),
         "contiguous range data does not match tensor data");
  expect(std::ranges::equal(const_tensor, expected_values),
         "const iteration returned incorrect values");
}

void test_random_access_algorithms_mutate_only_elements() {
  Tensor tensor =
      Tensor::from_data({2, 3}, {4.0F, 1.0F, 6.0F, 2.0F, 5.0F, 3.0F});
  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const std::array<float, 6> expected_values{1.0F, 2.0F, 3.0F,
                                             4.0F, 5.0F, 6.0F};

  std::ranges::sort(tensor);

  expect(std::ranges::equal(tensor, expected_values),
         "ranges algorithm did not mutate tensor elements correctly");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "ranges algorithm changed tensor shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "ranges algorithm changed tensor strides");
  expect(*(tensor.begin() + 4) == 5.0F,
         "random-access iterator arithmetic returned an incorrect element");
}

void test_reverse_iteration() {
  Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  const std::array<float, 4> reversed_values{4.0F, 3.0F, 2.0F, 1.0F};

  expect(tensor.rbegin().base() == tensor.end(),
         "reverse begin base does not match forward end");
  expect(tensor.rend().base() == tensor.begin(),
         "reverse end base does not match forward begin");
  expect(
      std::ranges::equal(std::ranges::subrange(tensor.rbegin(), tensor.rend()),
                         reversed_values),
      "mutable reverse iteration returned incorrect values");

  *tensor.rbegin() = 40.0F;

  expect(tensor.at(1, 1) == 40.0F,
         "write through reverse iterator did not change the tensor");

  const Tensor& const_tensor = tensor;
  const std::array<float, 4> expected_const_values{40.0F, 3.0F, 2.0F, 1.0F};

  expect(const_tensor.crbegin().base() == const_tensor.cend(),
         "const reverse begin base does not match const forward end");
  expect(const_tensor.crend().base() == const_tensor.cbegin(),
         "const reverse end base does not match const forward begin");
  expect(std::ranges::equal(std::ranges::subrange(const_tensor.crbegin(),
                                                  const_tensor.crend()),
                            expected_const_values),
         "const reverse iteration returned incorrect values");
}

void test_scalar_is_single_element_range() {
  Tensor tensor = Tensor::scalar(5.0F);

  expect(tensor.end() - tensor.begin() == 1,
         "scalar tensor iterator range must contain one element");
  expect(*tensor.begin() == 5.0F,
         "scalar tensor forward iterator returned an incorrect value");
  expect(*tensor.rbegin() == 5.0F,
         "scalar tensor reverse iterator returned an incorrect value");
}

void test_zero_extent_tensor_is_empty_range() {
  Tensor tensor({2, 0, 4});
  const Tensor& const_tensor = tensor;

  expect(tensor.begin() == tensor.end(),
         "zero-extent tensor forward range must be empty");
  expect(const_tensor.cbegin() == const_tensor.cend(),
         "const zero-extent tensor forward range must be empty");
  expect(tensor.rbegin() == tensor.rend(),
         "zero-extent tensor reverse range must be empty");
  expect(const_tensor.crbegin() == const_tensor.crend(),
         "const zero-extent tensor reverse range must be empty");
  expect(std::ranges::empty(tensor),
         "zero-extent tensor must model an empty range");
  expect(std::ranges::distance(tensor) == 0,
         "zero-extent tensor range distance must be zero");
}

void test_moved_from_tensor_is_empty_range() {
  Tensor source = Tensor::from_data({3}, {1.0F, 2.0F, 3.0F});
  Tensor destination(std::move(source));
  const std::array<float, 3> expected_values{1.0F, 2.0F, 3.0F};

  expect(source.begin() == source.end(),
         "moved-from tensor forward range must be empty");
  expect(source.rbegin() == source.rend(),
         "moved-from tensor reverse range must be empty");
  expect(std::ranges::empty(source),
         "moved-from tensor must model an empty range");
  expect(std::ranges::equal(destination, expected_values),
         "move destination iterator range lost tensor elements");
}

void test_range_for_extends_temporary_tensor_lifetime() {
  float sum = 0.0F;

  for (float value : Tensor::from_data({3}, {1.0F, 2.0F, 3.0F})) {
    sum += value;
  }

  expect(sum == 6.0F,
         "range-for over temporary tensor returned incorrect values");
}

void test_reshape_preserves_iterators() {
  Tensor tensor =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  Tensor::iterator first = tensor.begin();
  Tensor::iterator last = tensor.end();

  tensor.reshape({3, 2});

  expect(tensor.begin() == first,
         "reshape unexpectedly invalidated tensor begin iterator");
  expect(tensor.end() == last,
         "reshape unexpectedly invalidated tensor end iterator");
  expect(*first == 1.0F,
         "iterator preserved by reshape refers to an incorrect element");
}

}  // namespace

int main() {
  try {
    test_forward_iteration_exposes_contiguous_storage();
    test_const_iteration_exposes_read_only_elements();
    test_random_access_algorithms_mutate_only_elements();
    test_reverse_iteration();
    test_scalar_is_single_element_range();
    test_zero_extent_tensor_is_empty_range();
    test_moved_from_tensor_is_empty_range();
    test_range_for_extends_temporary_tensor_lifetime();
    test_reshape_preserves_iterators();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor iterator tests passed\n";
  return EXIT_SUCCESS;
}
