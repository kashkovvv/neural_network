#include <algorithm>
#include <array>
#include <concepts>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "nn/tensor.hpp"

namespace {

using Tensor = nn::Tensor<float>;
using MutableView = nn::TensorView<float>;
using ConstView = nn::TensorView<const float>;
using MetadataSpan = std::span<const Tensor::size_type>;

template <typename TensorType>
concept CanCreateView = requires { std::declval<TensorType>().view(); };

template <typename ViewType>
concept CanCallShape = requires { std::declval<ViewType>().shape(); };

template <typename ViewType>
concept CanCallStrides = requires { std::declval<ViewType>().strides(); };

template <typename ViewType, typename... IndexTypes>
concept CanIndexWith =
    requires { std::declval<ViewType>()[std::declval<IndexTypes>()...]; };

template <typename ViewType, typename... IndexTypes>
concept CanAccessAtWith =
    requires { std::declval<ViewType>().at(std::declval<IndexTypes>()...); };

static_assert(
    std::same_as<decltype(std::declval<Tensor&>().view()), MutableView>);
static_assert(
    std::same_as<decltype(std::declval<const Tensor&>().view()), ConstView>);

static_assert(CanCreateView<Tensor&>);
static_assert(CanCreateView<const Tensor&>);
static_assert(!CanCreateView<Tensor&&>);
static_assert(!CanCreateView<const Tensor&&>);

static_assert(!noexcept(std::declval<Tensor&>().view()));
static_assert(!noexcept(std::declval<const Tensor&>().view()));

static_assert(std::same_as<MutableView::element_type, float>);
static_assert(std::same_as<ConstView::element_type, const float>);
static_assert(std::same_as<MutableView::value_type, float>);
static_assert(std::same_as<ConstView::value_type, float>);

static_assert(!std::default_initializable<MutableView>);
static_assert(std::copy_constructible<MutableView>);
static_assert(std::movable<MutableView>);
static_assert(std::swappable<MutableView>);
static_assert(std::is_nothrow_swappable_v<MutableView>);
static_assert(std::convertible_to<MutableView, ConstView>);
static_assert(!std::convertible_to<ConstView, MutableView>);
static_assert(noexcept(MutableView(std::declval<MutableView&&>())));
static_assert(
    noexcept(std::declval<MutableView&>() = std::declval<MutableView&&>()));
static_assert(!noexcept(
    std::declval<MutableView&>() = std::declval<const MutableView&>()));
static_assert(std::same_as<decltype(std::declval<MutableView&>().swap(
                               std::declval<MutableView&>())),
                           void>);
static_assert(
    noexcept(std::declval<MutableView&>().swap(std::declval<MutableView&>())));
static_assert(std::same_as<decltype(nn::swap(std::declval<MutableView&>(),
                                             std::declval<MutableView&>())),
                           void>);
static_assert(noexcept(nn::swap(std::declval<MutableView&>(),
                                std::declval<MutableView&>())));

static_assert(
    std::same_as<decltype(std::declval<MutableView&>().shape()), MetadataSpan>);
static_assert(std::same_as<decltype(std::declval<const MutableView&>().shape()),
                           MetadataSpan>);
static_assert(
    std::same_as<decltype(std::declval<ConstView&>().shape()), MetadataSpan>);

static_assert(std::same_as<decltype(std::declval<MutableView&>().strides()),
                           MetadataSpan>);
static_assert(
    std::same_as<decltype(std::declval<const MutableView&>().strides()),
                 MetadataSpan>);
static_assert(
    std::same_as<decltype(std::declval<ConstView&>().strides()), MetadataSpan>);

static_assert(CanCallShape<MutableView&>);
static_assert(CanCallShape<const MutableView&>);
static_assert(!CanCallShape<MutableView&&>);
static_assert(!CanCallShape<const MutableView&&>);

static_assert(CanCallStrides<MutableView&>);
static_assert(CanCallStrides<const MutableView&>);
static_assert(!CanCallStrides<MutableView&&>);
static_assert(!CanCallStrides<const MutableView&&>);

static_assert(noexcept(std::declval<const MutableView&>().rank()));
static_assert(noexcept(std::declval<const MutableView&>().numel()));
static_assert(noexcept(std::declval<const MutableView&>().shape()));
static_assert(noexcept(std::declval<const MutableView&>().strides()));
static_assert(noexcept(std::declval<const MutableView&>().is_contiguous()));

static_assert(
    std::same_as<decltype(std::declval<MutableView&>()[0, 0]), float&>);
static_assert(
    std::same_as<decltype(std::declval<const MutableView&>()[0, 0]), float&>);
static_assert(
    std::same_as<decltype(std::declval<MutableView>()[0, 0]), float&>);

static_assert(
    std::same_as<decltype(std::declval<ConstView&>()[0, 0]), const float&>);
static_assert(std::same_as<decltype(std::declval<const ConstView&>()[0, 0]),
                           const float&>);
static_assert(
    std::same_as<decltype(std::declval<ConstView>()[0, 0]), const float&>);

static_assert(
    std::same_as<decltype(std::declval<MutableView&>().at(0, 0)), float&>);
static_assert(std::same_as<
              decltype(std::declval<const MutableView&>().at(0, 0)), float&>);
static_assert(
    std::same_as<decltype(std::declval<MutableView>().at(0, 0)), float&>);

static_assert(
    std::same_as<decltype(std::declval<ConstView&>().at(0, 0)), const float&>);
static_assert(std::same_as<decltype(std::declval<const ConstView&>().at(0, 0)),
                           const float&>);
static_assert(
    std::same_as<decltype(std::declval<ConstView>().at(0, 0)), const float&>);

static_assert(noexcept(std::declval<MutableView>()[0, 0]));
static_assert(noexcept(std::declval<ConstView>()[0, 0]));
static_assert(!noexcept(std::declval<MutableView>().at(0, 0)));
static_assert(!noexcept(std::declval<ConstView>().at(0, 0)));

static_assert(CanIndexWith<MutableView, short, unsigned int>);
static_assert(CanAccessAtWith<ConstView, short, unsigned int>);
static_assert(!CanIndexWith<MutableView, bool>);
static_assert(!CanIndexWith<ConstView, char>);
static_assert(!CanAccessAtWith<MutableView, float>);
static_assert(!CanAccessAtWith<ConstView, int, double>);

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

void expect_empty_view_sentinel(const MutableView& view, const char* context) {
  expect(view.rank() == 0, context);
  expect(view.numel() == 0, context);
  expect(view.shape().empty(), context);
  expect(view.strides().empty(), context);
  expect(view.is_contiguous(), context);
}

void test_whole_view_metadata_and_row_major_access() {
  Tensor tensor =
      Tensor::from_data({2, 3, 2}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
                                    7.0F, 8.0F, 9.0F, 10.0F, 11.0F});
  MutableView view = tensor.view();

  const Tensor::shape_type expected_shape{2, 3, 2};
  const Tensor::strides_type expected_strides{6, 2, 1};

  expect(view.rank() == 3, "whole view rank is incorrect");
  expect(view.numel() == 12, "whole view numel is incorrect");
  expect(std::ranges::equal(view.shape(), expected_shape),
         "whole view shape is incorrect");
  expect(std::ranges::equal(view.strides(), expected_strides),
         "whole view strides are incorrect");
  expect(view.is_contiguous(), "whole tensor view must be contiguous");
  expect(view[1, 0, 1] == 7.0F,
         "whole view unchecked row-major access is incorrect");
  expect(view.at(1, 2, 0) == 10.0F,
         "whole view checked row-major access is incorrect");

  const std::array<unsigned int, 3> index_values{0U, 2U, 1U};
  const std::span<const unsigned int> indices{index_values};

  expect(view[indices] == 5.0F, "whole view span indexing is incorrect");
  expect(view.at(indices) == 5.0F,
         "whole view checked span access is incorrect");
}

void test_mutable_view_aliases_tensor_storage() {
  Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  MutableView view = tensor.view();

  view[0, 1] = 20.0F;

  expect(tensor.at(0, 1) == 20.0F,
         "write through mutable view did not change the tensor");

  tensor.at(1, 0) = 30.0F;

  expect(view.at(1, 0) == 30.0F,
         "tensor write is not visible through mutable view");

  const MutableView& const_handle = view;
  const_handle.at(1, 1) = 40.0F;

  expect(tensor.at(1, 1) == 40.0F,
         "const mutable-view handle unexpectedly made elements const");
}

void test_temporary_view_accesses_live_tensor_storage() {
  Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});

  tensor.view()[0, 1] = 20.0F;

  expect(tensor.at(0, 1) == 20.0F,
         "temporary mutable view write did not change the tensor");

  const Tensor& const_tensor = tensor;

  expect(const_tensor.view().at(1, 0) == 3.0F,
         "temporary const view returned an incorrect element");
}

void test_const_view_reads_shared_storage() {
  Tensor tensor = Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
  const Tensor& const_tensor = tensor;
  ConstView view = const_tensor.view();

  expect(view[1, 0] == 3.0F, "const view returned an incorrect element");

  tensor.at(1, 0) = -3.0F;

  expect(view.at(1, 0) == -3.0F,
         "tensor write is not visible through const view");
}

void test_view_copy_and_const_conversion_share_storage() {
  Tensor tensor = Tensor::from_data({3}, {1.0F, 2.0F, 3.0F});
  MutableView original = tensor.view();
  MutableView copy = original;
  ConstView const_view = original;

  copy[1] = 22.0F;

  expect(original[1] == 22.0F,
         "copied view does not share the original storage");
  expect(const_view[1] == 22.0F,
         "converted const view does not share the original storage");
  expect(tensor[1] == 22.0F, "copied view write did not change the tensor");
}

void test_view_copy_assignment_rebinds_storage() {
  Tensor source_tensor = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor previous_target_tensor = Tensor::from_data({1}, {3.0F});
  MutableView source = source_tensor.view();
  MutableView target = previous_target_tensor.view();

  target = source;
  target.at(1) = 20.0F;

  expect(source.at(1) == 20.0F,
         "copy-assigned view does not share source view storage");
  expect(source_tensor.at(1) == 20.0F,
         "copy-assigned view write did not change source tensor");
  expect(previous_target_tensor.at(0) == 3.0F,
         "copy assignment changed previously referenced tensor storage");
}

void test_view_move_semantics() {
  Tensor source_tensor = Tensor::from_data({2}, {1.0F, 2.0F});
  MutableView source = source_tensor.view();
  MutableView destination(std::move(source));

  expect(destination.at(1) == 2.0F,
         "move-constructed view does not reference source tensor storage");
  expect_empty_view_sentinel(source,
                             "move construction left an invalid view sentinel");

  expect_throws<std::invalid_argument>(
      [&source] { static_cast<void>(source.at(0)); },
      "move-constructed-from view access did not throw std::invalid_argument",
      "move-constructed-from view access produced the wrong exception type");

  Tensor other_tensor = Tensor::from_data({1}, {3.0F});
  MutableView move_source = source_tensor.view();
  MutableView move_target = other_tensor.view();

  move_target = std::move(move_source);

  expect(move_target.at(0) == 1.0F,
         "move-assigned view does not reference source tensor storage");
  expect_empty_view_sentinel(move_source,
                             "move assignment left an invalid view sentinel");
}

void test_view_swap_exchanges_complete_states() {
  Tensor left_tensor = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor right_tensor = Tensor::from_data({1, 2}, {3.0F, 4.0F});
  MutableView left = left_tensor.view();
  MutableView right = right_tensor.view();

  using std::swap;
  swap(left, right);

  const Tensor::shape_type expected_left_shape{1, 2};
  const Tensor::shape_type expected_right_shape{2};

  expect(std::ranges::equal(left.shape(), expected_left_shape),
         "view swap did not transfer shape to the left view");
  expect(std::ranges::equal(right.shape(), expected_right_shape),
         "view swap did not transfer shape to the right view");

  left.at(0, 1) = 40.0F;
  right.at(1) = 20.0F;

  expect(right_tensor.at(0, 1) == 40.0F,
         "view swap did not transfer storage binding to the left view");
  expect(left_tensor.at(1) == 20.0F,
         "view swap did not transfer storage binding to the right view");
}

void test_view_metadata_is_independent_of_tensor_reshape() {
  Tensor tensor =
      Tensor::from_data({2, 3}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
  MutableView view = tensor.view();

  tensor.reshape({3, 2});

  const Tensor::shape_type expected_view_shape{2, 3};
  const Tensor::strides_type expected_view_strides{3, 1};
  const Tensor::shape_type expected_tensor_shape{3, 2};

  expect(std::ranges::equal(view.shape(), expected_view_shape),
         "tensor reshape changed existing view shape");
  expect(std::ranges::equal(view.strides(), expected_view_strides),
         "tensor reshape changed existing view strides");
  expect(std::ranges::equal(tensor.shape(), expected_tensor_shape),
         "tensor reshape produced an incorrect tensor shape");
  expect(view.at(1, 2) == 5.0F,
         "view access became incorrect after tensor reshape");

  view.at(0, 2) = 42.0F;

  expect(tensor.at(1, 0) == 42.0F,
         "view stopped aliasing storage after tensor reshape");
}

void test_scalar_view() {
  Tensor tensor = Tensor::scalar(5.0F);
  MutableView view = tensor.view();

  expect(view.rank() == 0, "scalar view rank must be zero");
  expect(view.numel() == 1, "scalar view numel must be one");
  expect(view.shape().empty(), "scalar view shape must be empty");
  expect(view.strides().empty(), "scalar view strides must be empty");
  expect(view.is_contiguous(), "scalar view must be contiguous");
  expect(view[] == 5.0F, "scalar view unchecked access is incorrect");

  view.at() = -5.0F;

  expect(tensor.at() == -5.0F, "scalar view write did not change the tensor");
}

void test_zero_extent_view() {
  Tensor tensor({2, 0, 4});
  MutableView view = tensor.view();

  const Tensor::shape_type expected_shape{2, 0, 4};
  const Tensor::strides_type expected_strides{0, 4, 1};

  expect(view.rank() == 3, "zero-extent view rank is incorrect");
  expect(view.numel() == 0, "zero-extent view numel must be zero");
  expect(std::ranges::equal(view.shape(), expected_shape),
         "zero-extent view shape is incorrect");
  expect(std::ranges::equal(view.strides(), expected_strides),
         "zero-extent view strides are incorrect");
  expect(view.is_contiguous(), "whole zero-extent view must be contiguous");

  expect_throws<std::out_of_range>(
      [&view] { static_cast<void>(view.at(0, 0, 0)); },
      "zero-extent view access did not throw std::out_of_range",
      "zero-extent view access produced the wrong exception type");
}

void test_checked_view_access_rejects_invalid_indices() {
  Tensor tensor({2, 3});
  MutableView view = tensor.view();

  expect_throws<std::invalid_argument>(
      [&view] { static_cast<void>(view.at(0)); },
      "view rank mismatch did not throw std::invalid_argument",
      "view rank mismatch produced the wrong exception type");

  expect_throws<std::out_of_range>(
      [&view] { static_cast<void>(view.at(-1, 0)); },
      "negative view index did not throw std::out_of_range",
      "negative view index produced the wrong exception type");

  expect_throws<std::out_of_range>(
      [&view] { static_cast<void>(view.at(2, 0)); },
      "view index equal to extent did not throw std::out_of_range",
      "view index equal to extent produced the wrong exception type");
}

void test_moved_from_tensor_rejects_view_creation() {
  Tensor source = Tensor::from_data({2}, {1.0F, 2.0F});
  Tensor destination(std::move(source));

  expect_throws<std::invalid_argument>(
      [&source] { static_cast<void>(source.view()); },
      "moved-from tensor view creation did not throw std::invalid_argument",
      "moved-from tensor view creation produced the wrong exception type");

  ConstView destination_view = std::as_const(destination).view();

  expect(destination_view.at(1) == 2.0F,
         "destination view is incorrect after tensor move construction");
}

}  // namespace

int main() {
  try {
    test_whole_view_metadata_and_row_major_access();
    test_mutable_view_aliases_tensor_storage();
    test_temporary_view_accesses_live_tensor_storage();
    test_const_view_reads_shared_storage();
    test_view_copy_and_const_conversion_share_storage();
    test_view_copy_assignment_rebinds_storage();
    test_view_move_semantics();
    test_view_swap_exchanges_complete_states();
    test_view_metadata_is_independent_of_tensor_reshape();
    test_scalar_view();
    test_zero_extent_view();
    test_checked_view_access_rejects_invalid_indices();
    test_moved_from_tensor_rejects_view_creation();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor view tests passed\n";
  return EXIT_SUCCESS;
}
