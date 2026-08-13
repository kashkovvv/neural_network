#include "math/vector.hpp"

Vector::Vector(size_type size) : data_(size) {}
Vector::Vector(std::initializer_list<value_type> values) : data_(values) {}

Vector::size_type Vector::size() const noexcept { return data_.size(); }
bool Vector::empty() const noexcept { return data_.empty(); }

Vector::reference Vector::operator[](size_type index) noexcept {
  return data_[index];
}
Vector::const_reference Vector::operator[](size_type index) const noexcept {
  return data_[index];
}

Vector::reference Vector::at(size_type index) { return data_.at(index); }
Vector::const_reference Vector::at(size_type index) const {
  return data_.at(index);
}

Vector::pointer Vector::data() noexcept { return data_.data(); }
Vector::const_pointer Vector::data() const noexcept { return data_.data(); }

Vector::iterator Vector::begin() noexcept { return data_.begin(); }
Vector::const_iterator Vector::begin() const noexcept { return data_.begin(); }

Vector::iterator Vector::end() noexcept { return data_.end(); }
Vector::const_iterator Vector::end() const noexcept { return data_.end(); }

Vector::const_iterator Vector::cbegin() const noexcept {
  return data_.cbegin();
}
Vector::const_iterator Vector::cend() const noexcept { return data_.cend(); }
