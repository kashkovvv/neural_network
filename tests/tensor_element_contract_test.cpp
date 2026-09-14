#include <concepts>
#include <cstdlib>
#include <iostream>

#include "nn/tensor.hpp"

namespace {

class AbstractElement {
 public:
  virtual ~AbstractElement() = default;
  virtual void operation() = 0;
};

class IncompleteElement;

class NonDestructibleElement {
 public:
  ~NonDestructibleElement() = delete;
};

class ThrowingDestructorElement {
 public:
  ~ThrowingDestructorElement() noexcept(false) {}
};

class RegularElement {};

enum class EnumElement { value };

using FunctionElement = void();

template <typename Element>
concept CanFormTensor = requires { typename nn::Tensor<Element>; };

template <typename Element>
concept CanFormTensorView = requires { typename nn::TensorView<Element>; };

static_assert(CanFormTensor<float>);
static_assert(CanFormTensor<int>);
static_assert(CanFormTensor<RegularElement>);
static_assert(CanFormTensor<EnumElement>);
static_assert(CanFormTensor<int*>);
static_assert(CanFormTensor<const int*>);
static_assert(CanFormTensor<IncompleteElement*>);

static_assert(!CanFormTensor<const float>);
static_assert(!CanFormTensor<volatile float>);
static_assert(!CanFormTensor<const volatile float>);
static_assert(!CanFormTensor<float&>);
static_assert(!CanFormTensor<float&&>);
static_assert(!CanFormTensor<void>);
static_assert(!CanFormTensor<FunctionElement>);
static_assert(!CanFormTensor<float[2]>);
static_assert(!CanFormTensor<AbstractElement>);
static_assert(!CanFormTensor<IncompleteElement>);
static_assert(!CanFormTensor<NonDestructibleElement>);
static_assert(!CanFormTensor<ThrowingDestructorElement>);
static_assert(!CanFormTensor<bool>);
static_assert(!CanFormTensor<int* const>);

static_assert(CanFormTensorView<float>);
static_assert(CanFormTensorView<const float>);
static_assert(CanFormTensorView<RegularElement>);
static_assert(CanFormTensorView<const RegularElement>);
static_assert(CanFormTensorView<const int*>);
static_assert(CanFormTensorView<int* const>);
static_assert(CanFormTensorView<IncompleteElement*>);

static_assert(!CanFormTensorView<volatile float>);
static_assert(!CanFormTensorView<const volatile float>);
static_assert(!CanFormTensorView<float&>);
static_assert(!CanFormTensorView<float&&>);
static_assert(!CanFormTensorView<void>);
static_assert(!CanFormTensorView<FunctionElement>);
static_assert(!CanFormTensorView<float[2]>);
static_assert(!CanFormTensorView<AbstractElement>);
static_assert(!CanFormTensorView<IncompleteElement>);
static_assert(!CanFormTensorView<NonDestructibleElement>);
static_assert(!CanFormTensorView<ThrowingDestructorElement>);
static_assert(!CanFormTensorView<bool>);
static_assert(!CanFormTensorView<const bool>);

static_assert(std::same_as<typename nn::Tensor<float>::value_type, float>);
static_assert(
    std::same_as<typename nn::TensorView<float>::element_type, float>);
static_assert(std::same_as<typename nn::TensorView<const float>::element_type,
                           const float>);
static_assert(std::same_as<typename nn::TensorView<float>::value_type, float>);
static_assert(
    std::same_as<typename nn::TensorView<const float>::value_type, float>);

}  // namespace

int main() {
  std::cout << "All tensor element contract tests passed\n";
  return EXIT_SUCCESS;
}
