# Frozen model

`frozen_candidate_v1_models.joblib.xz` is an XZ-compressed joblib artifact containing classifier and regressor pairs for 30 and 60 Hz. The lossless compression keeps the file below GitHub's per-file limit without changing model parameters or predictions.

Because joblib uses Python's pickle protocol, the model file should be loaded only after its provenance and checksum have been verified.

The companion JSON contains the frozen-candidate identifier and development fingerprint. Cryptographic file hashes are recorded in the root `MANIFEST.sha256`.
