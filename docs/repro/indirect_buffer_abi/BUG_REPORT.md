# Issue summary

See [README.md](README.md) for the complete reproducer and requested PTOAS work. The full recurrence is numerically checked before timing with a BF16-scale `0.25` absolute tolerance. Verified device-only FFTS is `1457.808 us` for direct CCE and `4592.626 us` for stacked VMI (`CCE/VMI = 0.3174`).
