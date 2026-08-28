# Historical development evidence

This file preserves earlier results without making version history part of the
final interpretation card. It is a record of model development, not an additional
test set or a claim of independent replication.

## Adaptive-noise negative result

The interpretation card at frozen commit
`1f1899f80c65ad2523f153a8eacfd51eeccc449d`, section 6.2, recorded:

| Earlier comparison | Public mean NLL |
|---|---:|
| StoVol-Lite with heuristic adaptive noise | 0.641966 |
| Same specification, `noise_adaptation_rate=0` | 0.638966 |
| Disabled minus enabled | -0.003000 |

The tested rule tried to reallocate reward error between process and observation
noise. Disabling it slightly improved the reported local public-data score.
This motivated using offline-selected, fixed noise variances.

**Provenance limit:** these numbers were transcribed from the committed historical
card; this final-development pass did not rerun that experiment. No standalone
raw log or exact historical fit manifest for this comparison was located in the
current checkout. Do not label it newly reproduced or a confirmatory ablation.
It rejects neither human volatility learning nor all possible adaptive-noise
models. It also does not demonstrate separate identification of process and
observation noise.

## Earlier policy change

The original single-softmax specification combined Kalman value, uncertainty,
choice traces and reward-stay bias. The frozen submission uses a stay-gate
factorization with signed/squared surprise, log run length, log trial count and
conditional target choice. This changes the available dynamic predictors.
As established by Q6, factorization itself is not a distinct psychological
architecture: a suitably parameterized single softmax is exactly equivalent.

The old card reported NLL 0.641966 / accuracy 79.0912% for the earlier model and
0.569410 / 79.7015% for the frozen submission. Only the latter was independently
rerun in this pass; see `artifacts/baseline_public_evaluation.json`. A lower NLL
alone is not proof of improved calibration, because discrimination and
probability sharpness also affect log loss.
