# Frozen algorithm and evaluation protocol

## Intended use and limits

The candidate was designed for a 60-second contact finger-PPG segment recorded while the participant is seated, awake, and still. The algorithm operates at 30 and 60 Hz and requires agreement between both resolutions. Processing instrumental PPG at these rates addresses temporal sampling compatibility only. It does not constitute validation of smartphone-camera PPG or of RGB acquisition, illumination, compression, exposure control, rolling shutter, or device-dependent camera processing.

The UTSA office condition included computer or telephone use, conversation, drinking water, and walking. It is therefore treated as an out-of-domain stress test involving dynamic office activity, not as an isolated hand-motion experiment.

## Frozen processing specification

1. Analysis of the same 60-second segment at 30 and 60 Hz.
2. Third-order, zero-phase 0.55–3.50 Hz Butterworth band-pass filtering.
3. Signal-polarity selection based on agreement between spectral and peak-derived rates.
4. Three-point parabolic interpolation of detected peaks.
5. Calculation of 14 local interval and amplitude features.
6. Interval classification with the frozen Extra Trees classifier and a probability threshold of 0.20.
7. Pulse-interval correction with the frozen Extra Trees regressor.
8. Minimum accepted-interval fraction of 0.70 and at least 30 consecutive accepted pairs at each resolution.
9. SQI threshold of 0.65 at both resolutions; SQI 0.75 was evaluated as a sensitivity analysis.
10. Agreement between 30 and 60 Hz estimates, defined as a heart-rate difference ≤2 beats/min and an RMSSD difference ≤5 ms or ≤10% of the paired mean.

The ECG NN reference rule accepts 0.30–2.00 second intervals whose deviation from the local median is no greater than the larger of 20% of that median or 35 ms. Original indices are preserved so that RMSSD is calculated only from truly consecutive accepted intervals.

The fourth SQI component measures robust consistency of detected pulse amplitudes. The historical output column is named `sqi_prominence` for compatibility with the frozen result files.

## Model provenance

The two frozen models were trained on the designated development data from Li, BIDMC, and stable UTSA conditions. Welltory and WF-PPG contributed to exploratory development but were not external-validation cohorts. The distributed model has separate classifiers and regressors for 30 and 60 Hz, and its metadata fingerprint is stored beside the model. The evaluation scripts do not use PTT-PPG, Vollmer, or UTSA office records for refitting or threshold selection.

The recorded freeze date forms part of the development audit trail and does not constitute a prospective registration. The model checksum verifies artifact integrity only.

## Evaluation roles

- **PTT-PPG seated records:** primary independent external validation, 22 participants and three distal PPG channels.
- **Vollmer standing-rest phase:** secondary independent replication, fixed central 60-second window, 13 records assessed.
- **UTSA office:** out-of-domain stress test, five fixed-position windows per recording and channel.
- **Corrected NN-index audit:** explicitly post hoc sensitivity analysis conducted after freezing.

## Reproducibility

The repository provides the frozen model, the derived outputs used in the reported analyses, and deterministic evaluation code. Raw physiological datasets are not redistributed. Reproduction therefore also depends on the availability and original structure of the cited third-party datasets and on the specified Python package versions.
