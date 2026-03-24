# SubView Split Board Validation (A3)

This case validates PTO subview split correctness on board.

- Input: one `16x16` f32 parent tile (`src.bin`)
- Kernel: subview into four `8x8` tiles at offsets `(0,0)`, `(0,8)`, `(8,0)`, `(8,8)`
- Outputs: `out0.bin`, `out1.bin`, `out2.bin`, `out3.bin`
- Golden: produced by `golden.py` via NumPy slicing of the input

Pass condition: all 4 outputs exactly match golden (`np.allclose` with `1e-6`).
