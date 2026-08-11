# Performance issue

The standalone device-1 harness measures 32.543 us for the direct CCE kernel
and 17557.001 us for the equivalent tiled VMI work extent. The CCE result is
both host-golden checked and independently demonstrates that this algorithmic
pattern is fast on A5. See `README.md` for the complete compile, ctypes launch,
correctness, and timing command.
