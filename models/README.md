# Frozen model

`frozen_candidate_v1_models.joblib.xz` is a losslessly XZ-compressed joblib artifact containing classifier/regressor pairs for 30 and 60 Hz. Compression keeps the file below GitHub's per-file limit without changing model parameters or predictions.

Model deserialization is restricted to the verified artifact distributed in this repository. As with all Python pickle/joblib objects, provenance verification is required before loading.

The companion JSON contains the frozen-candidate identifier and development fingerprint. Cryptographic file hashes are recorded in the root `MANIFEST.sha256`.
