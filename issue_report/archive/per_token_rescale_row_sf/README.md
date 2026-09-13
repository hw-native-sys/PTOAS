# Kernel compose leftover (was old A)

Compose calls `cast_back` then bf16 per_token
(`_compose_per_token_rescale`). Recorded: fused path rejects; compose
SF off-by-one vs ASC fused.

Not a proven PTOAS hole and not closed by PRs 89–91. Keep the log.
Do not reopen as live **A**.
