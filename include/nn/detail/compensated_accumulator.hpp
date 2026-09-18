#pragma once

#include <cmath>
#include <concepts>

namespace nn::detail {

template <std::floating_point T>
class CompensatedAccumulator {
 public:
  void add(T value) {
    const T next_sum = sum_ + value;

    if (!std::isfinite(sum_) || !std::isfinite(value) ||
        !std::isfinite(next_sum)) {
      sum_ = next_sum;
      correction_ = T{};

      return;
    }

    if (std::abs(sum_) >= std::abs(value)) {
      correction_ += (sum_ - next_sum) + value;
    } else {
      correction_ += (value - next_sum) + sum_;
    }

    sum_ = next_sum;
  }

  [[nodiscard]] T result() const { return sum_ + correction_; }

 private:
  T sum_{};
  T correction_{};
};

}  // namespace nn::detail
