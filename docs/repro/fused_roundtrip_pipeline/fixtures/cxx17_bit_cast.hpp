#pragma once

// CANN's low-level headers use this C++20 utility; the compiler provides the
// builtin even when the device source is compiled in C++17 mode.
#if defined(__cplusplus) && (__cplusplus < 202002L)
#if defined(__has_builtin) && __has_builtin(__builtin_bit_cast)
namespace std {
template <class To, class From>
constexpr To bit_cast(const From &from) noexcept { return __builtin_bit_cast(To, from); }
}
#endif
#endif
