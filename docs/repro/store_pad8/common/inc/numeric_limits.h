#pragma once

#include <cstdint>

namespace fa {
namespace limits {

// Use compiler builtin so device compiles can stay on C++17 (avoids
// clang-15 + libstdc++-12 breakage under -std=c++20 on some hosts).

inline constexpr float kFloatInf = __builtin_bit_cast(float, 0x7F800000u);
inline constexpr float kFloatNInf = __builtin_bit_cast(float, 0xFF800000u);
inline constexpr float kFloatNaN = __builtin_bit_cast(float, 0x7FC00000u);

} // namespace limits
} // namespace fa
