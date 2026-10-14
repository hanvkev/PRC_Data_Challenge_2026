# PRC Data Challenge 2026 — taxi-out time

This repository contains the solution of team **talented-jelly** for the [PRC Data Challenge
2026](https://prc-data-challenge-2026.netlify.app/). The team consists of one person; "we" in the documentation is
simply the usual writing style.

The task is to predict the taxi-out time of departures at ten European airports (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD,
LFPG, LIRF, LSZH, LTFM). The models learn from 2025, and the final ranking covers January, February, June and July 2026.

**Result.** Evaluated on months that the models did not see during training (out of fold), the final pipeline reaches an
RMSE of 236.5 s over the twelve months of 2025, and 186.8 s on labels up to one hour. On the daily leaderboard (January
and July 2026) the same models score 268.7.

## Approach

The label is take-off minus off-block, `MVT − BLOCK`, which for most flights is the taxi time. For about 0.3 % of the
departures, however, the label equals the delay `MVT − SCHED` and can amount to many hours, usually because the
off-block was recorded at the scheduled time although the aircraft left later. These labels, which we call artefacts,
are few, but under squared error they weigh heavily. The pipeline therefore separates two questions:

- **How likely is it that the label equals the delay?** A classifier answers this with the probability `p_artifact`.
- **What is the label if it does not equal the delay?** Base A, trained on all flights, provides a starting point, and
  base B, trained only on normal flights, a more precise estimate for normal traffic. A corrector receives both
  predictions together with `p_artifact`, learns only from labels that do not equal the delay, and turns them into the
  value `corrected`.

The final prediction weights the two cases, `final = corrected + p_artifact · (delay − corrected)`. Two safety nets
follow: a rule for one extreme situation at LIRF, and the LOBT window, which keeps every prediction within 3606 s of
`MVT − LOBT` because every 2025 label of a flight with an LOBT lay within that window. All four models are CatBoost
models trained on all twelve months of 2025. The development started on the leaderboard months January and July plus
December only; in the final phase it moved to all twelve months, and the choices that matter most were checked again on
all of them. A few earlier choices still rest on January and July
([docs/method.md](docs/method.md#how-choices-were-made)).

## Documentation

| Document | Content |
|---|---|
| [docs/method.md](docs/method.md) | how the pipeline works and why each part is there |
| [docs/results.md](docs/results.md) | all numbers: results per month, contribution of each step, comparisons, leaderboard history, data gaps in 2026, ideas without gain |
| [docs/features.md](docs/features.md) | the features and which model uses them |
| [docs/data_sources.md](docs/data_sources.md) | the external data and their licences |
| `analysis/` | notebooks on the questions about the data that shaped the design, each ending with what followed for the models, and the error analysis of the final pipeline |

## Repository

```
data/                 01 download → 02 clean → 03 ADS-B fetch → 04 ADS-B movement → 05 features
  adsb_*.py           helper modules for 03, 04 and 05
  tests/              unit tests of the ADS-B processing
model/                06 base A → 07 base B → 08 classifier → 09 corrector → 10 / 11 predict
  runner.py           shared code: training per month, refits, post-processing, prediction
  features/           the one feature that is fitted on the training months (median taxi time per stand)
  final/              the trained models of the final submission (written by 09, read by 10 and 11)
analysis/             the questions about the data behind the design, and the error analysis
docs/                 method, features, results, data sources
```

| Notebook | What it does |
|---|---|
| `data/01_download` | downloads the challenge files and merges the training months |
| `data/02_clean` | fills gaps in operator, aircraft type, wake category, … from the flight number's history |
| `data/03_adsb_fetch` | downloads ADS-B positions from ADSB.lol near the airports, from 6 h before to 30 min after each take-off |
| `data/04_adsb_movement` | finds each departure's trajectory and its first ground movement |
| `data/05_features` | builds all model features for training, ranking and final ranking |
| `model/06_base_a`, `07_base_b` | the two base models: each month is predicted by a model trained on the other eleven, then one final model is trained on all of 2025 |
| `model/08_classifier` | the probability of an artefact, `p_artifact` |
| `model/09_corrector` | corrector, mixture, rule and LOBT window; evaluation per month; saves all final models |
| `model/10_predict_ranking` | prediction for the daily leaderboard (`ranking.parquet`), with optional upload and score |
| `model/11_predict_final` | prediction for the final ranking (`final_ranking.parquet`) |
| `analysis/*` | label types, LOBT window, flights without NM data, ADS-B coverage, ADS-B as a lower bound, error analysis |

## Running it

The environment needs Python 3.11 and [uv](https://docs.astral.sh/uv/): `uv sync` installs everything from
`pyproject.toml` and `uv.lock`. The notebooks run in their numbered order, and each one reads what the step before
wrote. The challenge data need the organisers' access keys (`pyopensky` configuration). ADS-B, weather, OpenStreetMap
and X-Plane data come from public sources and are downloaded by the notebooks
([docs/data_sources.md](docs/data_sources.md)).

The data are not part of the repository. The ADS-B steps need about 50 GB of disk space, and `03`, `04`, `06` and `07`
run for several hours; we ran them as batch jobs with `jupyter nbconvert --to notebook --execute`. Training runs are
logged in a local MLflow store (`model/mlruns`, not versioned).

The final models are part of the repository (`model/final`, about 155 MB). Once `01`–`05` have prepared the data, `10`
and `11` therefore work on a fresh clone without any training.

## Licence

GNU GPL v3 (see [LICENSE](LICENSE)). External data keep their own licences, see
[docs/data_sources.md](docs/data_sources.md).