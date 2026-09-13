# Kernel compose leftover — closed 0913

Compose calls `cast_back` then bf16 per_token
(`_compose_per_token_rescale`). Row-major fused/compose is now bitwise
vs ASC on `quant_missing_impl_0913`.

Not a PTOAS hole. `recorded.log` here is the old SF off-by-one.
