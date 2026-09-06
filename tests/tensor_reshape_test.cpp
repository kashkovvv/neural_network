#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "nn/tensor.hpp"

namespace {

using Tensor = nn::Tensor<float>;

template <typename TensorType>
concept CanReshape = requires(TensorType&& tensor) {
  std::forward<TensorType>(tensor).reshape(
      typename std::remove_cvref_t<TensorType>::shape_type{});
};

static_assert(std::same_as<
              decltype(std::declval<Tensor&>().reshape(Tensor::shape_type{})),
              void>);
static_assert(CanReshape<Tensor&>);
static_assert(CanReshape<nn::Tensor<int>&>);
static_assert(!CanReshape<const Tensor&>);
static_assert(!CanReshape<Tensor&&>);
static_assert(!CanReshape<const Tensor&&>);
static_assert(
    !noexcept(std::declval<Tensor&>().reshape(Tensor::shape_type{})));

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

void test_reshape_changes_shape_and_preserves_storage() {
  Tensor tensor =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  auto existing_elements = tensor.elements();
  Tensor::value_type* const existing_data = existing_elements.data();
  Tensor::value_type& existing_reference = existing_elements[4];

  tensor.reshape({3, 2});

  const Tensor::shape_type expected_shape{3, 2};
  const Tensor::strides_type expected_strides{2, 1};
  const Tensor::storage_type expected_elements{1.0F, 2.0F, 3.0F,
                                                4.0F, 5.0F, 6.0F};

  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "reshape produced an incorrect shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "reshape produced incorrect row-major strides");
  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "reshape changed the linear element order");
  expect(tensor.elements().data() == existing_data,
         "reshape changed the storage address");
  expect(existing_elements.data() == existing_data,
         "reshape invalidated an existing elements view");
  expect(existing_reference == 5.0F,
         "reshape invalidated an existing element reference");
  expect(tensor[2, 0] == 5.0F,
         "reshape produced incorrect multidimensional indexing");

  existing_reference = -5.0F;

  expect(tensor[2, 0] == -5.0F,
         "write through a pre-reshape reference was not preserved");
}

void test_reshape_changes_rank() {
  Tensor tensor({2, 3, 4});

  tensor.reshape({4, 6});

  const Tensor::shape_type expected_shape{4, 6};
  const Tensor::strides_type expected_strides{6, 1};

  expect(tensor.rank() == 2, "reshape did not change rank");
  expect(tensor.numel() == 24, "reshape changed element count");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "rank-changing reshape produced an incorrect shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "rank-changing reshape produced incorrect strides");
}

void test_reshape_scalar_and_singleton_dimensions() {
  Tensor tensor = Tensor::scalar(7.0F);
  Tensor::value_type* const existing_data = tensor.elements().data();

  tensor.reshape({1, 1});

  const Tensor::shape_type matrix_shape{1, 1};
  const Tensor::strides_type matrix_strides{1, 1};

  expect(std::ranges::equal(tensor.shape(), matrix_shape),
         "scalar reshape produced an incorrect shape");
  expect(std::ranges::equal(tensor.strides(), matrix_strides),
         "scalar reshape produced incorrect strides");
  expect(tensor[0, 0] == 7.0F, "scalar reshape changed the element value");
  expect(tensor.elements().data() == existing_data,
         "scalar reshape changed the storage address");

  tensor.reshape({});

  expect(tensor.rank() == 0, "reshape to scalar produced an incorrect rank");
  expect(tensor.shape().empty(), "reshape to scalar produced a nonempty shape");
  expect(tensor.strides().empty(),
         "reshape to scalar produced nonempty strides");
  expect(tensor.at() == 7.0F, "reshape to scalar changed the element value");
}

void test_reshape_zero_extent_tensor() {
  Tensor tensor({2, 0, 4});

  tensor.reshape({3, 0, 2});

  const Tensor::shape_type expected_shape{3, 0, 2};
  const Tensor::strides_type expected_strides{0, 2, 1};

  expect(tensor.numel() == 0, "zero-extent reshape changed element count");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "zero-extent reshape produced an incorrect shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "zero-extent reshape produced incorrect strides");
  expect(tensor.elements().empty(),
         "zero-extent reshape produced nonempty storage");
}

void test_reshape_to_same_shape() {
  Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  Tensor::value_type* const existing_data = tensor.elements().data();

  tensor.reshape({2, 2});

  const Tensor::shape_type expected_shape{2, 2};
  const Tensor::strides_type expected_strides{2, 1};
  const Tensor::storage_type expected_elements{1.0F, 2.0F, 3.0F, 4.0F};

  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "same-shape reshape changed shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "same-shape reshape changed strides");
  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "same-shape reshape changed elements");
  expect(tensor.elements().data() == existing_data,
         "same-shape reshape changed the storage address");
}

void test_reshape_rejects_different_element_count() {
  Tensor tensor =
      Tensor::from_data({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const auto existing_shape = tensor.shape();
  const auto existing_strides = tensor.strides();
  const auto existing_elements = tensor.elements();
  Tensor::value_type* const existing_data = tensor.elements().data();

  expect_throws<std::invalid_argument>(
      [&tensor] { tensor.reshape({2, 2}); },
      "reshape with a different element count did not throw",
      "reshape with a different element count produced the wrong exception "
      "type");

  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_elements{1.0F, 2.0F, 3.0F,
                                                4.0F, 5.0F, 6.0F};

  expect(std::ranges::equal(existing_shape, expected_shape),
         "failed reshape invalidated the existing shape view");
  expect(std::ranges::equal(existing_strides, expected_strides),
         "failed reshape invalidated the existing strides view");
  expect(std::ranges::equal(existing_elements, expected_elements),
         "failed reshape invalidated the existing elements view");
  expect(tensor.elements().data() == existing_data,
         "failed reshape changed the storage address");
  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "failed reshape changed shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "failed reshape changed strides");
  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "failed reshape changed elements");
}

void test_reshape_rejects_shape_product_overflow() {
  Tensor tensor = Tensor::from_data({2}, {1.0F, 2.0F});
  const Tensor::size_type max_size =
      std::numeric_limits<Tensor::size_type>::max();

  expect_throws<std::overflow_error>(
      [&tensor, max_size] { tensor.reshape({max_size, 2}); },
      "reshape with an overflowing shape did not throw",
      "reshape with an overflowing shape produced the wrong exception type");

  const Tensor::shape_type expected_shape{2};
  const Tensor::strides_type expected_strides{1};
  const Tensor::storage_type expected_elements{1.0F, 2.0F};

  expect(std::ranges::equal(tensor.shape(), expected_shape),
         "overflowing reshape changed shape");
  expect(std::ranges::equal(tensor.strides(), expected_strides),
         "overflowing reshape changed strides");
  expect(std::ranges::equal(tensor.elements(), expected_elements),
         "overflowing reshape changed elements");
}

void test_reshape_rejects_shape_larger_than_storage_max_size() {
  Tensor tensor = Tensor::scalar(1.0F);
  const Tensor::size_type storage_max_size = Tensor::storage_type{}.max_size();
  const Tensor::size_type size_type_max =
      std::numeric_limits<Tensor::size_type>::max();

  if (storage_max_size == size_type_max) {
    return;
  }

  expect_throws<std::length_error>(
      [&tensor, storage_max_size] { tensor.reshape({storage_max_size + 1}); },
      "reshape above storage max_size did not throw",
      "reshape above storage max_size produced the wrong exception type");

  expect(tensor.rank() == 0, "oversized reshape changed scalar rank");
  expect(tensor.numel() == 1, "oversized reshape changed element count");
  expect(tensor.at() == 1.0F, "oversized reshape changed the element value");
}

void test_reshape_rejects_moved_from_sentinel() {
  Tensor source = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor destination(std::move(source));
  static_cast<void>(destination);

  expect_throws<std::invalid_argument>(
      [&source] { source.reshape({0}); },
      "reshape of a moved-from sentinel did not throw",
      "reshape of a moved-from sentinel produced the wrong exception type");

  expect(source.rank() == 0, "failed sentinel reshape changed rank");
  expect(source.numel() == 0, "failed sentinel reshape changed element count");
  expect(source.shape().empty(), "failed sentinel reshape changed shape");
  expect(source.strides().empty(), "failed sentinel reshape changed strides");
  expect(source.elements().empty(),
         "failed sentinel reshape changed elements");
}

struct TrackedValue {
  TrackedValue() = default;

  TrackedValue(const TrackedValue& other) : value(other.value) {
    ++operation_count;
  }

  TrackedValue(TrackedValue&& other) noexcept : value(other.value) {
    ++operation_count;
  }

  TrackedValue& operator=(const TrackedValue& other) {
    value = other.value;
    ++operation_count;
    return *this;
  }

  TrackedValue& operator=(TrackedValue&& other) noexcept {
    value = other.value;
    ++operation_count;
    return *this;
  }

  int value{};
  inline static std::size_t operation_count{};
};

void test_reshape_does_not_copy_or_move_elements() {
  nn::Tensor<TrackedValue> tensor({2});
  auto elements = tensor.elements();
  TrackedValue* const existing_data = elements.data();
  TrackedValue& existing_reference = elements[1];
  existing_reference.value = 42;
  TrackedValue::operation_count = 0;

  tensor.reshape({1, 2});

  expect(TrackedValue::operation_count == 0,
         "reshape copied or moved stored elements");
  expect(tensor.elements().data() == existing_data,
         "tracked reshape changed the storage address");
  expect(&tensor.elements()[1] == &existing_reference,
         "tracked reshape invalidated an element reference");
  expect(tensor[0, 1].value == 42,
         "tracked reshape changed an element value");
}

}  // namespace

int main() {
  try {
    test_reshape_changes_shape_and_preserves_storage();
    test_reshape_changes_rank();
    test_reshape_scalar_and_singleton_dimensions();
    test_reshape_zero_extent_tensor();
    test_reshape_to_same_shape();
    test_reshape_rejects_different_element_count();
    test_reshape_rejects_shape_product_overflow();
    test_reshape_rejects_shape_larger_than_storage_max_size();
    test_reshape_rejects_moved_from_sentinel();
    test_reshape_does_not_copy_or_move_elements();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor reshape tests passed\n";
  return EXIT_SUCCESS;
}
