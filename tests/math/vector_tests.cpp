#include <cassert>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "math/vector.hpp"

void test_default_constructor() {
  Vector v;

  assert(v.size() == 0);
  assert(v.empty());
}

void test_size_constructor() {
  Vector empty(0);

  assert(empty.size() == 0);
  assert(empty.empty());

  const std::size_t size = 5;
  Vector non_empty(size);

  assert(non_empty.size() == size);
  assert(!non_empty.empty());

  for (std::size_t i = 0; i < size; ++i) {
    assert(non_empty[i] == 0.0f);
  }
}

void test_initializer_list() {
  Vector v{1.0f, 2.0f, 3.0f};

  assert(v.size() == 3);
  assert(!v.empty());

  assert(v[0] == 1.0f);
  assert(v[1] == 2.0f);
  assert(v[2] == 3.0f);
}

void test_size_and_empty() {
  Vector empty;
  Vector non_empty{1.0f};

  assert(empty.size() == 0);
  assert(empty.empty());

  assert(non_empty.size() == 1);
  assert(!non_empty.empty());
}

void test_operator_brackets() {
  Vector v{1.0f, 2.0f, 3.0f};

  assert(v[0] == 1.0f);
  assert(v[1] == 2.0f);
  assert(v[2] == 3.0f);

  v[1] = 42.0f;

  assert(v[1] == 42.0f);
}

void test_at() {
  Vector v{1.0f, 2.0f, 3.0f};

  assert(v.at(0) == 1.0f);
  assert(v.at(1) == 2.0f);
  assert(v.at(2) == 3.0f);

  try {
    v.at(3);
    assert(false);
  } catch (const std::out_of_range&) {
  }

  // The exception must not change the vector.
  assert(v.size() == 3);
  assert(v[0] == 1.0f);
  assert(v[1] == 2.0f);
  assert(v[2] == 3.0f);

  v.at(1) = 42.0f;

  assert(v.at(1) == 42.0f);
}

void test_data() {
  Vector v{1.0f, 2.0f, 3.0f};

  assert(v.data() == &v[0]);
  assert(v.data() + 1 == &v[1]);
  assert(v.data() + 2 == &v[2]);

  v.data()[1] = 42.0f;

  assert(v[1] == 42.0f);
}

void test_iterators() {
  Vector v{1.0f, 2.0f, 3.0f};

  std::size_t index = 0;

  for (auto it = v.begin(); it != v.end(); ++it) {
    assert(*it == v[index]);
    ++index;
  }

  assert(index == v.size());

  for (auto it = v.begin(); it != v.end(); ++it) {
    *it *= 2.0f;
  }

  assert(v[0] == 2.0f);
  assert(v[1] == 4.0f);
  assert(v[2] == 6.0f);

  static_assert(std::is_same_v<decltype(std::declval<Vector&>().begin()),
                               Vector::iterator>);

  static_assert(std::is_same_v<decltype(std::declval<const Vector&>().begin()),
                               Vector::const_iterator>);

  static_assert(std::is_same_v<decltype(std::declval<Vector&>().end()),
                               Vector::iterator>);

  static_assert(std::is_same_v<decltype(std::declval<const Vector&>().end()),
                               Vector::const_iterator>);
}

void test_const_iterators() {
  const Vector v{1.0f, 2.0f, 3.0f};

  std::size_t index = 0;

  for (auto it = v.begin(); it != v.end(); ++it) {
    assert(*it == v[index]);
    ++index;
  }

  assert(index == v.size());

  index = 0;

  for (auto it = v.cbegin(); it != v.cend(); ++it) {
    assert(*it == v[index]);
    ++index;
  }

  assert(index == v.size());
}

void test_range_for() {
  Vector v{1.0f, 2.0f, 3.0f};

  float sum = 0.0f;

  for (float value : v) {
    sum += value;
  }

  assert(sum == 6.0f);
}

void test_const_correctness() {
  const Vector v{1.0f, 2.0f, 3.0f};

  assert(v[0] == 1.0f);
  assert(v.at(1) == 2.0f);
  assert(v.size() == 3);

  static_assert(std::is_same_v<decltype(std::declval<Vector&>()[0]), float&>);

  static_assert(
      std::is_same_v<decltype(std::declval<const Vector&>()[0]), const float&>);

  static_assert(
      std::is_same_v<decltype(std::declval<Vector&>().at(0)), float&>);

  static_assert(std::is_same_v<decltype(std::declval<const Vector&>().at(0)),
                               const float&>);

  static_assert(
      std::is_same_v<decltype(std::declval<Vector&>().data()), float*>);

  static_assert(std::is_same_v<decltype(std::declval<const Vector&>().data()),
                               const float*>);
}

void test_copy() {
  Vector a{1.0f, 2.0f, 3.0f};

  // Self-assignment.
  a = a;

  assert(a[0] == 1.0f);
  assert(a[1] == 2.0f);
  assert(a[2] == 3.0f);

  // Copy construction.
  Vector b = a;

  assert(b.size() == 3);
  assert(b[0] == 1.0f);
  assert(b[1] == 2.0f);
  assert(b[2] == 3.0f);

  b[1] = 42.0f;

  assert(a[1] == 2.0f);
  assert(b[1] == 42.0f);

  // Copy assignment.
  Vector c{10.0f, 20.0f, 30.0f, 40.0f};

  c = a;

  assert(c.size() == 3);
  assert(c[0] == 1.0f);
  assert(c[1] == 2.0f);
  assert(c[2] == 3.0f);

  c[1] = 42.0f;

  assert(a[1] == 2.0f);
  assert(c[1] == 42.0f);
}

void test_move() {
  // Move construction.
  Vector source_1{1.0f, 2.0f, 3.0f};

  Vector destination_1 = std::move(source_1);

  assert(destination_1.size() == 3);
  assert(destination_1[0] == 1.0f);
  assert(destination_1[1] == 2.0f);
  assert(destination_1[2] == 3.0f);

  // Move assignment.
  Vector source_2{1.0f, 2.0f, 3.0f};
  Vector destination_2{10.0f};

  destination_2 = std::move(source_2);

  assert(destination_2.size() == 3);
  assert(destination_2[0] == 1.0f);
  assert(destination_2[1] == 2.0f);
  assert(destination_2[2] == 3.0f);
}

void test_noexcept() {
  static_assert(std::is_nothrow_default_constructible_v<Vector>);

  static_assert(std::is_nothrow_move_constructible_v<Vector>);

  static_assert(std::is_nothrow_move_assignable_v<Vector>);

  static_assert(noexcept(std::declval<const Vector&>().size()));

  static_assert(noexcept(std::declval<const Vector&>().empty()));

  static_assert(noexcept(std::declval<Vector&>().operator[](0)));

  static_assert(noexcept(std::declval<const Vector&>().operator[](0)));

  static_assert(!noexcept(std::declval<Vector&>().at(0)));

  static_assert(!noexcept(std::declval<const Vector&>().at(0)));

  static_assert(noexcept(std::declval<Vector&>().data()));

  static_assert(noexcept(std::declval<const Vector&>().data()));

  static_assert(noexcept(std::declval<Vector&>().begin()));

  static_assert(noexcept(std::declval<const Vector&>().begin()));

  static_assert(noexcept(std::declval<Vector&>().end()));

  static_assert(noexcept(std::declval<const Vector&>().end()));

  static_assert(noexcept(std::declval<const Vector&>().cbegin()));

  static_assert(noexcept(std::declval<const Vector&>().cend()));
}

int main() {
  test_default_constructor();
  test_size_constructor();
  test_initializer_list();
  test_size_and_empty();
  test_operator_brackets();
  test_at();
  test_data();
  test_iterators();
  test_const_iterators();
  test_range_for();
  test_const_correctness();
  test_copy();
  test_move();
  test_noexcept();
}