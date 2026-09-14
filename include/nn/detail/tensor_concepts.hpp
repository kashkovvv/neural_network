#pragma once

#include <concepts>
#include <type_traits>

namespace nn::detail {

template <typename Element>
concept tensor_element =
    std::is_object_v<Element> && (!std::is_array_v<Element>) &&
    std::same_as<Element, std::remove_cv_t<Element>> &&
    (!std::same_as<Element, bool>) && requires { sizeof(Element); } &&
    (!std::is_abstract_v<Element>) && std::destructible<Element>;

template <typename Element>
concept tensor_view_element = tensor_element<std::remove_const_t<Element>>;

}  // namespace nn::detail
