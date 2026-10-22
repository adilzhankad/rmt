# Superseded runs

Complete, scored runs that were replaced by a later run of the same model and benchmark, kept
for the record. Their rows stay in `logs/run_log.csv`; `experiment/collect.py` uses the later row.

- `main/Qwen2.5-VL-7B-Instruct/OCRBench_v2`: run on transformers (2026-10-21, English Overall
  0.445). Qwen2.5-VL was moved to VLMEvalKit's vLLM backend to fit the GPU budget, and OCRBench v2
  was run again on vLLM so that all four benchmarks of the model use the same engine.
