# Issue summary

See [README.md](README.md) for the complete reproducer and requested PTOAS work. The verified standalone `8192 x 2048` BF16/E4M3, group-32, no-rounded-scale case measures `22.743 us` for fused CCE and `34.796 us` for composed VMI by device-only FFTS (`CCE/VMI = 0.6536`).
