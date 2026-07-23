# Derived results

This directory contains non-identifying derived outputs used in the manuscript.

- `primary_external/`: frozen PTT-PPG primary validation and SQI sensitivity analysis.
- `secondary_external/`: Vollmer resting replication and window-selection audit.
- `out_of_domain/`: UTSA office stress test; rows from model-development conditions are retained for auditability and are not classified as external validation.
- `posthoc/`: corrected NN-index sensitivity analysis performed after freezing.

The primary decision threshold is SQI >= 0.65. SQI >= 0.75 is a sensitivity analysis, not a replacement decision rule.

`primary_external/ptt_bootstrap_ci.csv` contains the participant-level confidence intervals based on 10,000 bootstrap resamples and reported in Table 3 of the manuscript. The corresponding record-level data are provided in the same directory.
