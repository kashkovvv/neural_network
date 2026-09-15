#include <concepts>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "nn/tensor.hpp"

namespace {

using Tensor = nn::Tensor<float>;
using MutableView = nn::TensorView<float>;
using ConstView = nn::TensorView<const float>;

template <typename SourceType>
concept CanCallData = requires { std::declval<SourceType>().data(); };

static_assert(std::same_as<Tensor::pointer, float*>);
static_assert(std::same_as<Tensor::const_pointer, const float*>);
static_assert(std::same_as<MutableView::pointer, float*>);
static_assert(std::same_as<ConstView::pointer, const float*>);

static_assert(CanCallData<Tensor&>);
static_assert(CanCallData<const Tensor&>);
static_assert(!CanCallData<Tensor&&>);
static_assert(!CanCallData<const Tensor&&>);

static_assert(CanCallData<MutableView&>);
static_assert(CanCallData<const MutableView&>);
static_assert(CanCallData<MutableView&&>);
static_assert(CanCallData<const MutableView&&>);
static_assert(CanCallData<ConstView&>);
static_assert(CanCallData<const ConstView&>);
static_assert(CanCallData<ConstView&&>);
static_assert(CanCallData<const ConstView&&>);

static_assert(std::same_as<decltype(std::declval<Tensor&>().data()), float*>);
static_assert(
    std::same_as<decltype(std::declval<const Tensor&>().data()), const float*>);
static_assert(
    std::same_as<decltype(std::declval<MutableView&>().data()), float*>);
static_assert(
    std::same_as<decltype(std::declval<const MutableView&>().data()), float*>);
static_assert(
    std::same_as<decltype(std::declval<ConstView&>().data()), const float*>);
static_assert(std::same_as<decltype(std::declval<const ConstView&>().data()),
                           const float*>);

static_assert(noexcept(std::declval<Tensor&>().data()));
static_assert(noexcept(std::declval<const Tensor&>().data()));
static_assert(noexcept(std::declval<MutableView&>().data()));
static_assert(noexcept(std::declval<const MutableView&>().data()));
static_assert(noexcept(std::declval<MutableView&&>().data()));
static_assert(noexcept(std::declval<const MutableView&&>().data()));
static_assert(noexcept(std::declval<ConstView&>().data()));
static_assert(noexcept(std::declval<const ConstView&>().data()));
static_assert(noexcept(std::declval<ConstView&&>().data()));
static_assert(noexcept(std::declval<const ConstView&&>().data()));

void expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_tensor_data_exposes_contiguous_storage() {
  Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});

  expect(tensor.data() == tensor.elements().data(),
         "mutable tensor data does not point to contiguous storage");

  tensor.data()[2] = 30.0F;

  expect(tensor.at(1, 0) == 30.0F,
         "write through mutable tensor data did not change the tensor");

  const Tensor& const_tensor = tensor;

  expect(const_tensor.data() == tensor.data(),
         "const tensor data does not point to the same storage");
  expect(const_tensor.data()[2] == 30.0F,
         "const tensor data returned an incorrect element");
}

void test_view_data_preserves_element_constness_and_aliasing() {
  Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  MutableView mutable_view = tensor.view();
  const MutableView& const_mutable_handle = mutable_view;
  ConstView const_view = std::as_const(tensor).view();

  expect(mutable_view.data() == tensor.data(),
         "mutable whole view data does not point to tensor storage");
  expect(const_mutable_handle.data() == tensor.data(),
         "const mutable-view handle unexpectedly changed the data pointer");
  expect(const_view.data() == std::as_const(tensor).data(),
         "const view data does not point to tensor storage");

  const_mutable_handle.data()[1] = 20.0F;

  expect(tensor.at(0, 1) == 20.0F,
         "write through const mutable-view handle did not change the tensor");
  expect(const_view.data()[1] == 20.0F,
         "const view did not observe a write through mutable view data");
}

void test_contiguous_subview_data_includes_origin_offset() {
  Tensor tensor =
      Tensor::from_data({3, 4}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F,
                                 8.0F, 9.0F, 10.0F, 11.0F});
  MutableView view = tensor.slice(0, 1, 3);

  expect(view.is_contiguous(), "row subview must be contiguous");
  expect(view.data() == tensor.data() + 4,
         "contiguous subview data ignored its origin offset");
  expect(*view.data() == tensor.at(1, 0),
         "contiguous subview data does not point to its first logical element");
}

void test_non_contiguous_view_data_points_only_to_logical_origin() {
  Tensor tensor =
      Tensor::from_data({3, 4}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F,
                                 8.0F, 9.0F, 10.0F, 11.0F});
  MutableView view = tensor.slice(1, 1, 4, 2);

  expect(!view.is_contiguous(), "strided column view must be non-contiguous");
  expect(view.data() == tensor.data() + 1,
         "non-contiguous view data ignored its origin offset");
  expect(*view.data() == view.at(0, 0),
         "non-contiguous view data does not point to its logical origin");
  expect(
      view.data()[1] != view.at(0, 1),
      "test setup did not demonstrate that view data is not a logical range");
}

void test_nested_view_data_accumulates_origin_offsets() {
  Tensor tensor = Tensor::from_data(
      {4, 6}, {0.0F,  1.0F,  2.0F,  3.0F,  4.0F,  5.0F,  6.0F,  7.0F,
               8.0F,  9.0F,  10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F,
               16.0F, 17.0F, 18.0F, 19.0F, 20.0F, 21.0F, 22.0F, 23.0F});
  MutableView view = tensor.slice(0, 1, 4, 2).slice(1, 2, 6, 3);

  expect(view.data() == tensor.data() + 8,
         "nested view data did not accumulate slicing origin offsets");
  expect(*view.data() == view.at(0, 0),
         "nested view data does not point to its first logical element");
}

void test_temporary_view_data_refers_to_live_tensor_storage() {
  Tensor tensor = Tensor::from_data({3}, {1.0F, 2.0F, 3.0F});

  expect(tensor.view().data() == tensor.data(),
         "temporary mutable view returned an incorrect data pointer");
  expect(std::as_const(tensor).view().data() == std::as_const(tensor).data(),
         "temporary const view returned an incorrect data pointer");
}

void test_scalar_data_points_to_the_scalar_element() {
  Tensor tensor = Tensor::scalar(5.0F);
  MutableView view = tensor.view();

  expect(tensor.data() != nullptr, "scalar tensor data must not be null");
  expect(view.data() == tensor.data(),
         "scalar view data does not point to scalar tensor storage");
  expect(*view.data() == tensor.at(),
         "scalar view data returned an incorrect scalar element");
}

void test_empty_objects_return_null_data() {
  Tensor empty_tensor({2, 0, 4});
  MutableView empty_whole_view = empty_tensor.view();

  expect(empty_tensor.data() == nullptr,
         "zero-extent tensor data must be null");
  expect(empty_whole_view.data() == nullptr,
         "zero-extent whole view data must be null");

  Tensor tensor =
      Tensor::from_data({2, 3}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
  MutableView empty_subview = tensor.slice(1, 2, 2, 2);

  expect(empty_subview.numel() == 0,
         "empty subview test setup produced a non-empty view");
  expect(empty_subview.data() == nullptr,
         "empty subview with non-empty backing storage must return null data");
}

void test_moved_from_objects_return_null_data() {
  Tensor tensor_source = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor tensor_destination(std::move(tensor_source));

  expect(tensor_source.data() == nullptr,
         "moved-from tensor data must be null");
  expect(tensor_destination.data() != nullptr,
         "move destination tensor data must not be null");

  MutableView view_source = tensor_destination.view();
  MutableView view_destination(std::move(view_source));

  expect(view_source.data() == nullptr, "moved-from view data must be null");
  expect(view_destination.data() == tensor_destination.data(),
         "move destination view data does not reference tensor storage");
}

}  // namespace

int main() {
  try {
    test_tensor_data_exposes_contiguous_storage();
    test_view_data_preserves_element_constness_and_aliasing();
    test_contiguous_subview_data_includes_origin_offset();
    test_non_contiguous_view_data_points_only_to_logical_origin();
    test_nested_view_data_accumulates_origin_offsets();
    test_temporary_view_data_refers_to_live_tensor_storage();
    test_scalar_data_points_to_the_scalar_element();
    test_empty_objects_return_null_data();
    test_moved_from_objects_return_null_data();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor data tests passed\n";
  return EXIT_SUCCESS;
}
