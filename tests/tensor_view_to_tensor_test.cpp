#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "nn/tensor.hpp"

namespace {

using Tensor = nn::Tensor<float>;
using MutableView = nn::TensorView<float>;
using ConstView = nn::TensorView<const float>;

struct CopyConstructibleOnly {
  explicit CopyConstructibleOnly(int initial_value) : value(initial_value) {}

  CopyConstructibleOnly(const CopyConstructibleOnly&) = default;
  CopyConstructibleOnly(CopyConstructibleOnly&&) noexcept = default;
  CopyConstructibleOnly& operator=(const CopyConstructibleOnly&) = delete;
  CopyConstructibleOnly& operator=(CopyConstructibleOnly&&) = delete;

  int value;
};

struct MoveOnly {
  MoveOnly() = default;
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly(MoveOnly&&) noexcept = default;
  MoveOnly& operator=(const MoveOnly&) = delete;
  MoveOnly& operator=(MoveOnly&&) noexcept = default;
};

struct ThrowOnCopy {
  explicit ThrowOnCopy(int initial_value) : value(initial_value) {}

  ThrowOnCopy(const ThrowOnCopy& other) : value(other.value) {
    if (copy_failure_enabled_) {
      if (successful_copies_remaining_ == 0) {
        throw std::runtime_error("copy failure");
      }

      --successful_copies_remaining_;
    }
  }

  ThrowOnCopy(ThrowOnCopy&&) noexcept = default;
  ThrowOnCopy& operator=(const ThrowOnCopy&) = delete;
  ThrowOnCopy& operator=(ThrowOnCopy&&) = delete;

  static void fail_after_successful_copies(std::size_t count) noexcept {
    successful_copies_remaining_ = count;
    copy_failure_enabled_ = true;
  }

  static void disable_copy_failure() noexcept { copy_failure_enabled_ = false; }

  int value;

 private:
  inline static std::size_t successful_copies_remaining_{0};
  inline static bool copy_failure_enabled_{false};
};

template <typename ViewType>
concept CanMaterialize = requires { std::declval<ViewType>().to_tensor(); };

static_assert(CanMaterialize<MutableView&>);
static_assert(CanMaterialize<const MutableView&>);
static_assert(CanMaterialize<MutableView&&>);
static_assert(CanMaterialize<const MutableView&&>);
static_assert(CanMaterialize<ConstView&>);
static_assert(CanMaterialize<const ConstView&>);
static_assert(CanMaterialize<ConstView&&>);
static_assert(CanMaterialize<const ConstView&&>);

static_assert(
    std::same_as<decltype(std::declval<const MutableView&>().to_tensor()),
                 Tensor>);
static_assert(
    std::same_as<decltype(std::declval<const ConstView&>().to_tensor()),
                 Tensor>);
static_assert(!noexcept(std::declval<const MutableView&>().to_tensor()));
static_assert(!noexcept(std::declval<const ConstView&>().to_tensor()));
static_assert(!std::convertible_to<MutableView, Tensor>);
static_assert(!std::convertible_to<ConstView, Tensor>);

static_assert(
    CanMaterialize<nn::TensorView<CopyConstructibleOnly>&>);
static_assert(!CanMaterialize<nn::TensorView<MoveOnly>&>);

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

void test_whole_view_materializes_independent_tensor() {
  Tensor source =
      Tensor::from_data({2, 3}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
  MutableView view = source.view();

  Tensor result = view.to_tensor();

  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_elements{0.0F, 1.0F, 2.0F,
                                                3.0F, 4.0F, 5.0F};

  expect(std::ranges::equal(result.shape(), expected_shape),
         "whole-view materialization changed shape");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "whole-view materialization produced incorrect strides");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "whole-view materialization changed element order");
  expect(result.elements().data() != source.elements().data(),
         "materialized tensor shares source storage");

  result.at(0, 1) = 20.0F;
  source.at(1, 2) = 50.0F;

  expect(source.at(0, 1) == 1.0F,
         "materialized tensor mutation changed its source");
  expect(result.at(1, 2) == 5.0F,
         "source mutation changed materialized tensor");
}

void test_contiguous_subview_uses_its_origin() {
  Tensor source = Tensor::from_data(
      {4, 3}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
               6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F});
  MutableView view = source.slice(0, 1, 3);

  Tensor result = view.to_tensor();

  const Tensor::shape_type expected_shape{2, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_elements{3.0F, 4.0F, 5.0F,
                                                6.0F, 7.0F, 8.0F};

  expect(view.is_contiguous(), "source subview must be contiguous");
  expect(std::ranges::equal(result.shape(), expected_shape),
         "contiguous subview materialization changed shape");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "contiguous subview materialization produced incorrect strides");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "contiguous subview materialization ignored its origin");
}

void test_non_contiguous_view_materializes_in_logical_row_major_order() {
  Tensor source = Tensor::from_data(
      {3, 5}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F,
               8.0F, 9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F});
  MutableView view = source.slice(1, 0, 5, 2);

  Tensor result = view.to_tensor();

  const Tensor::shape_type expected_shape{3, 3};
  const Tensor::strides_type expected_strides{3, 1};
  const Tensor::storage_type expected_elements{0.0F, 2.0F, 4.0F,
                                                5.0F, 7.0F, 9.0F,
                                                10.0F, 12.0F, 14.0F};

  expect(!view.is_contiguous(), "source view must be non-contiguous");
  expect(std::ranges::equal(result.shape(), expected_shape),
         "non-contiguous view materialization changed shape");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "non-contiguous view was not materialized contiguously");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "non-contiguous view materialization used physical instead of "
         "logical order");
}

void test_nested_strided_view_materializes_composed_layout() {
  Tensor source = Tensor::from_data(
      {4, 5}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
               7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F, 13.0F,
               14.0F, 15.0F, 16.0F, 17.0F, 18.0F, 19.0F});
  MutableView view =
      source.slice(0, 1, 4, 2).slice(1, 1, 5, 2);

  Tensor result = view.to_tensor();

  const Tensor::shape_type expected_shape{2, 2};
  const Tensor::strides_type expected_strides{2, 1};
  const Tensor::storage_type expected_elements{6.0F, 8.0F, 16.0F, 18.0F};

  expect(std::ranges::equal(result.shape(), expected_shape),
         "nested-view materialization changed shape");
  expect(std::ranges::equal(result.strides(), expected_strides),
         "nested-view materialization produced incorrect strides");
  expect(std::ranges::equal(result.elements(), expected_elements),
         "nested-view materialization did not compose origins and strides");
}

void test_const_and_temporary_views_materialize_mutable_tensors() {
  Tensor source =
      Tensor::from_data({2, 3}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
  const Tensor& const_source = source;
  ConstView const_view = const_source.slice(1, 1, 3);

  Tensor const_result = const_view.to_tensor();
  Tensor temporary_result = source.slice(0, 1, 2).to_tensor();

  const Tensor::storage_type expected_const_elements{1.0F, 2.0F, 4.0F, 5.0F};
  const Tensor::storage_type expected_temporary_elements{3.0F, 4.0F, 5.0F};

  expect(std::ranges::equal(const_result.elements(), expected_const_elements),
         "const view materialization produced incorrect elements");
  expect(std::ranges::equal(temporary_result.elements(),
                            expected_temporary_elements),
         "temporary view materialization produced incorrect elements");

  const_result.at(0, 0) = 10.0F;

  expect(source.at(0, 1) == 1.0F,
         "tensor materialized from const view shares source storage");
}

void test_scalar_and_empty_views_materialize_canonical_tensors() {
  Tensor scalar_source = Tensor::scalar(7.0F);
  Tensor scalar_result = scalar_source.view().to_tensor();

  expect(scalar_result.rank() == 0,
         "scalar-view materialization changed rank");
  expect(scalar_result.numel() == 1,
         "scalar-view materialization changed numel");
  expect(scalar_result.shape().empty(),
         "scalar-view materialization produced a nonempty shape");
  expect(scalar_result.strides().empty(),
         "scalar-view materialization produced nonempty strides");
  expect(scalar_result.at() == 7.0F,
         "scalar-view materialization changed its value");

  Tensor empty_source({2, 3});
  Tensor empty_result = empty_source.slice(1, 2, 2, 3).to_tensor();

  const Tensor::shape_type expected_empty_shape{2, 0};
  const Tensor::strides_type expected_empty_strides{0, 1};

  expect(std::ranges::equal(empty_result.shape(), expected_empty_shape),
         "empty-view materialization changed shape");
  expect(std::ranges::equal(empty_result.strides(), expected_empty_strides),
         "empty-view materialization produced noncanonical strides");
  expect(empty_result.numel() == 0,
         "empty-view materialization produced elements");
}

void test_copy_constructible_non_assignable_elements_are_supported() {
  using ElementTensor = nn::Tensor<CopyConstructibleOnly>;

  ElementTensor::storage_type source_elements;
  source_elements.emplace_back(1);
  source_elements.emplace_back(2);
  source_elements.emplace_back(3);
  source_elements.emplace_back(4);
  ElementTensor source =
      ElementTensor::from_data({2, 2}, std::move(source_elements));

  ElementTensor result = source.slice(1, 0, 2).to_tensor();

  expect(result.at(0, 0).value == 1 && result.at(0, 1).value == 2 &&
             result.at(1, 0).value == 3 && result.at(1, 1).value == 4,
         "materialization requires element assignment or changed values");
}

void test_copy_failure_leaves_source_view_unchanged() {
  using ElementTensor = nn::Tensor<ThrowOnCopy>;
  using ElementView = nn::TensorView<ThrowOnCopy>;

  ElementTensor::storage_type source_elements;
  source_elements.emplace_back(1);
  source_elements.emplace_back(2);
  source_elements.emplace_back(3);
  ElementTensor source =
      ElementTensor::from_data({3}, std::move(source_elements));
  ElementView view = source.view();

  ThrowOnCopy::fail_after_successful_copies(1);

  expect_throws<std::runtime_error>(
      [&view] { static_cast<void>(view.to_tensor()); },
      "view materialization did not propagate element copy failure",
      "view materialization copy failure produced the wrong exception type");

  ThrowOnCopy::disable_copy_failure();

  expect(view.rank() == 1 && view.numel() == 3,
         "copy failure changed source-view metadata");
  expect(view.at(0).value == 1 && view.at(1).value == 2 &&
             view.at(2).value == 3,
         "copy failure changed source-view elements");
}

void test_moved_from_view_rejects_materialization() {
  Tensor source = Tensor::from_data({2}, {1.0F, 2.0F});
  MutableView moved_from = source.view();
  MutableView destination(std::move(moved_from));

  expect_throws<std::invalid_argument>(
      [&moved_from] { static_cast<void>(moved_from.to_tensor()); },
      "moved-from view materialization did not throw std::invalid_argument",
      "moved-from view materialization produced the wrong exception type");

  expect(destination.at(1) == 2.0F,
         "failed moved-from materialization changed destination view");
}

}  // namespace

int main() {
  try {
    test_whole_view_materializes_independent_tensor();
    test_contiguous_subview_uses_its_origin();
    test_non_contiguous_view_materializes_in_logical_row_major_order();
    test_nested_strided_view_materializes_composed_layout();
    test_const_and_temporary_views_materialize_mutable_tensors();
    test_scalar_and_empty_views_materialize_canonical_tensors();
    test_copy_constructible_non_assignable_elements_are_supported();
    test_copy_failure_leaves_source_view_unchanged();
    test_moved_from_view_rejects_materialization();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tensor view materialization tests passed\n";
  return EXIT_SUCCESS;
}
