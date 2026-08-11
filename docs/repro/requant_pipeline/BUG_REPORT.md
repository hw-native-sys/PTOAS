# Performance issue

Stock A5 VMI lowering for dequantize/reduce/requantize compiles, but measured
ratios are 0.33–0.58 against direct CCE. Expected parity is at least 0.98. See
`README.md` for reproduction and acceptance criteria.
