#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

class Vector {
 public:
  using value_type = float;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = std::vector<value_type>::iterator;
  using const_iterator = std::vector<value_type>::const_iterator;

  Vector() = default;
  Vector(size_type);
  Vector(std::initializer_list<value_type>);

  size_type size() const noexcept;
  bool empty() const noexcept;

  reference operator[](size_type) noexcept;
  const_reference operator[](size_type) const noexcept;

  reference at(size_type);
  const_reference at(size_type) const;

  pointer data() noexcept;
  const_pointer data() const noexcept;

  iterator begin() noexcept;
  const_iterator begin() const noexcept;

  iterator end() noexcept;
  const_iterator end() const noexcept;

  const_iterator cbegin() const noexcept;
  const_iterator cend() const noexcept;

 private:
  std::vector<value_type> data_;
};