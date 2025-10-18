# Security & Cost Guardrails (Public Repo Version)

- **Secrets**: Use IAM roles or environment variables; never commit credentials.
- **PII/Audio**: Treat uploads as sensitive. Consider S3 bucket with lifecycle + KMS.
- **Costs**: Start with small instance (e.g., `ml.m5.xlarge`) and enable auto-scaling.
- **Logging**: Send only non‑sensitive metadata to logs; keep transcripts out of logs.
- **Quotas**: Watch concurrency and payload size. Prefer async for long audio.
