# Legacy experiment artifacts

The files in this directory are preserved for provenance, but they are **not**
the verified result of the current pipeline.

Audit findings:

- `evaluation_metrics.json` and `model_results.json` contain non-standard
  `NaN` values for R²;
- the prediction CSV files contain only one or two rows and do not establish
  the stated full-test metrics;
- `tft_final_model.pth` and `tft_optimized_model.pth` are raw `state_dict`
  files without a complete, versioned preprocessing/model configuration;
- the old notebook fitted imputations/scaling before the chronological split
  and treated observed sensor measurements as future-known inputs.

For current outputs, run the documented pipeline and use `artifacts/current/`.
Legacy files are intentionally retained rather than silently deleted.
