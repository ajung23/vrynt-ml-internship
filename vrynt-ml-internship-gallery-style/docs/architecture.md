# Architecture & Notes

```
[Streamlit UI] --(audio bytes)--> [SageMaker Endpoint: Wav2Vec2]
      ^                                      |
      |--------------(transcript JSON)-------|
```

**Key decisions**
- **Wav2Vec2 (HF) on SageMaker**: fast path to production‑grade STT without training from scratch.
- **Serverless-ish ergonomics**: one‑file deploy, baked‑in auto scaling, CloudWatch logs.
- **Client simplicity**: Streamlit + `boto3` keeps the footprint tiny and demo‑friendly.

**Alternatives considered**
- DeepSpeech / Silero / RNNT variants depending on latency/accuracy tradeoffs.
- TorchServe vs. SM inference handler—kept handler skeleton to ease future custom models.

See also the high‑level runbook in `docs/SECURITY_AND_COSTS.md` for guardrails.
