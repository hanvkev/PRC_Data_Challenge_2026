# Results

This document collects the numbers behind [method.md](method.md). All results are out of fold over the twelve months of
2025 unless stated otherwise; "≤ 1 h" means labels up to one hour.

## Final pipeline

The final models are runs 291 (base A), 292 (base B), 294 (classifier) and 295 (corrector), listed in
`model/final/manifest.json`. RMSE per month in seconds:

| Month | base A | base B | final | final, label ≤ 1 h |
|---|---|---|---|---|
| 1 | 340.3 | 458.9 | 328.0 | 186.3 |
| 2 | 283.8 | 459.2 | 218.9 | 180.1 |
| 3 | 197.6 | 315.1 | 181.5 | 167.3 |
| 4 | 201.3 | 294.1 | 182.5 | 172.9 |
| 5 | 284.8 | 324.2 | 275.0 | 170.6 |
| 6 | 303.2 | 431.5 | 294.3 | 191.1 |
| 7 | 351.7 | 635.7 | 271.0 | 227.7 |
| 8 | 269.4 | 407.8 | 223.7 | 205.7 |
| 9 | 238.2 | 439.0 | 229.8 | 193.8 |
| 10 | 209.2 | 258.2 | 200.2 | 182.7 |
| 11 | 193.0 | 385.3 | 183.0 | 169.1 |
| 12 | 198.8 | 361.9 | 188.5 | 176.0 |
| **all** | 262.2 | 409.8 | **236.5** | **186.8** |

Base B looks poor overall because it never learned the extreme labels; on labels up to one hour it is the most precise
single model (179.5 s, against 196.3 s for base A).

The months differ mainly through a few very large labels. For flights with an NM off-block time the final RMSE lies
between 171 s and 225 s in every month, for flights without one between 645 s and 2780 s. In January, May and June the
20 largest misses carry more than half of the squared error (`analysis/error_analysis.ipynb`).

On the daily leaderboard (January and July 2026) the same final models scored 268.68 (file v55).

## What each step contributes

The final out-of-fold predictions recomputed without some of the steps; nothing is retrained. The first rows leave out
one step each, the last rows show the way from base A alone to the final pipeline.

| Variant | Left out | RMSE over the year | RMSE ≤ 1 h | Months worse than the final pipeline |
|---|---|---|---|---|
| final pipeline | – | **236.5** | 186.8 | – |
| without the LOBT window | LOBT window | 237.0 | 187.0 | 12 of 12 |
| without the rule | rule | 238.7 | 186.8 | 6 of 12 (1 better) |
| without the mixture | mixture | 268.7 | 184.9 | 12 of 12 |
| without the corrector | corrector, and with it base B | 244.7 | 200.8 | 12 of 12 |
| base A with classifier and mixture only | corrector, rule, LOBT window | 247.3 | 201.0 | 12 of 12 |
| base A alone | corrector, classifier, mixture, rule, LOBT window | 262.2 | 196.3 | 12 of 12 |

## Comparisons of the final phase

Unless a table says otherwise, the variants use the same base A, base B and classifier.

### Base B: LightGBM or CatBoost

Both with a tree limit that no month model reached.

| Base B | base B, ≤ 1 h | final | final, ≤ 1 h |
|---|---|---|---|
| LightGBM, no depth limit, 255 leaves | 180.4 | 236.8 | 187.1 |
| CatBoost, depth 8 | **179.5** (10 of 12 months better) | **236.5** | **186.8** (9 of 12 months better) |

### Corrector with and without base B as an input

The corrector was trained a second time without the prediction of base B among its inputs, with all other settings
unchanged. In both cases each month is predicted by a corrector that did not learn from it.

| Month | with base B (final) | without base B | Difference |
|---|---|---|---|
| 1 | 328.0 | 333.6 | +5.6 |
| 2 | 218.9 | 224.7 | +5.8 |
| 3 | 181.5 | 187.6 | +6.1 |
| 4 | 182.5 | 188.0 | +5.5 |
| 5 | 275.0 | 278.5 | +3.5 |
| 6 | 294.3 | 299.6 | +5.3 |
| 7 | 271.0 | 275.6 | +4.6 |
| 8 | 223.7 | 230.6 | +7.0 |
| 9 | 229.8 | 236.9 | +7.1 |
| 10 | 200.2 | 205.5 | +5.2 |
| 11 | 183.0 | 188.4 | +5.4 |
| 12 | 188.5 | 194.1 | +5.6 |
| **all** | **236.5** | 241.9 | +5.4 |
| **≤ 1 h** | **186.8** | 193.5 | +6.7 |

### Classifier with and without the stand features

The classifier was trained a second time without its seven stand and inbound features, otherwise unchanged. The RMSE
columns use the final corrector and replace only `p_artifact` in the mixture.

| Month | AUC with | AUC without | RMSE with (final) | RMSE without | Difference |
|---|---|---|---|---|---|
| 1 | 0.954 | 0.951 | 328.0 | 328.4 | +0.5 |
| 2 | 0.939 | 0.938 | 218.9 | 218.1 | −0.8 |
| 3 | 0.959 | 0.959 | 181.5 | 182.0 | +0.5 |
| 4 | 0.957 | 0.955 | 182.5 | 183.1 | +0.6 |
| 5 | 0.968 | 0.966 | 275.0 | 275.7 | +0.8 |
| 6 | 0.961 | 0.958 | 294.3 | 294.5 | +0.2 |
| 7 | 0.961 | 0.957 | 271.0 | 273.4 | +2.5 |
| 8 | 0.965 | 0.964 | 223.7 | 226.2 | +2.5 |
| 9 | 0.970 | 0.969 | 229.8 | 231.4 | +1.6 |
| 10 | 0.963 | 0.961 | 200.2 | 199.1 | −1.1 |
| 11 | 0.955 | 0.953 | 183.0 | 182.9 | −0.1 |
| 12 | 0.966 | 0.964 | 188.5 | 188.5 | +0.0 |
| **all** | | | **236.5** | 237.2 | +0.7 |
| **≤ 1 h** | | | **186.8** | 187.3 | +0.5 |

### Tree limits

| Model | Limit | Month models at the limit | Largest month model | Trees of the final model |
|---|---|---|---|---|
| base A | 10 000 | 3 (Feb, Jun, Nov) | 10 000 | 9 200 |
| base A | 20 000 | 0 | 14 200 | 9 200 |
| base B (CatBoost) | 10 000 | many | 10 000 | 9 938 |
| base B (CatBoost) | 20 000 | 0 | 12 261 | 10 078 |

For base A the higher limit changed only February, June and November, by 0.2 to 0.6 s.

### Corrector trained on all months or on the leaderboard months

Both correctors use the final base models and classifier and no month feature, which cannot be evaluated out of fold.
The narrower one learns only from January, July and December, the months the corrector of v54 used; each month is
predicted by a corrector that did not learn from it.

| Month | all twelve months (final) | January, July, December only | Difference |
|---|---|---|---|
| 1 | 328.0 | 327.1 | −0.9 |
| 2 | 218.9 | 219.8 | +1.0 |
| 3 | 181.5 | 179.8 | −1.8 |
| 4 | 182.5 | 181.8 | −0.7 |
| 5 | 275.0 | 278.2 | +3.3 |
| 6 | 294.3 | 300.7 | +6.3 |
| 7 | 271.0 | 270.3 | −0.6 |
| 8 | 223.7 | 227.8 | +4.1 |
| 9 | 229.8 | 232.4 | +2.5 |
| 10 | 200.2 | 199.3 | −1.0 |
| 11 | 183.0 | 181.5 | −1.5 |
| 12 | 188.5 | 192.4 | +3.9 |
| **all** | **236.5** | 237.9 | +1.5 |
| **≤ 1 h** | **186.8** | 188.6 | +1.8 |

### Blend with base B

An earlier version of the pipeline pulled the corrected value part of the way towards base B,
`corrected + b · (base B − corrected)`, with `b` chosen from 0 to 0.3 together with α and W. With the earlier base
models `b = 0.2` gained 0.5 s over the year; with the final base models the choice was `b = 0` in every run, with both
base B variants and both tree limits, so the step was removed.

### Rules

Each rule replaces the prediction by the expected value over the label types seen in 2025 for the same situation (see
[method.md](method.md#rule-for-an-extreme-case)). The final phase started with three rules, which were switched off in
turn on the out-of-fold predictions of an earlier run of that phase; the rule estimates for each month came from the
other eleven months.

| Rule | Situation | RMSE over the year without it | Months without it |
|---|---|---|---|
| long NM wait | NM off-block more than 10 000 s before the take-off and a delay above 10 000 s | +0.01 | 1 better, 1 worse |
| LIRF, delay 50 000–60 000 s | LIRF, no NM off-block, delay of 50 000–60 000 s | +0.06 | 3 better, 3 worse |
| LIRF, delay 60 000–95 000 s | LIRF, no NM off-block, delay of 60 000–95 000 s | +1.87 | 1 better, 6 worse |

Only the last rule is kept.

## Earlier development steps

Before the final phase, variants were compared on January and July 2025 only, with a corrector of three seeds and
without month feature.

| Step | Jan | Jul | Jan ≤ 1 h | Jul ≤ 1 h |
|---|---|---|---|---|
| base A CatBoost depth 6 (basis of v50) | 331.3 | 278.9 | 188.0 | 238.4 |
| base A depth 8 | 330.1 | 273.7 | 186.1 | 233.7 |
| + CatBoost classifier with stand features | 328.7 | 272.3 | 186.3 | 233.3 |
| + ADS-B for flights without NM flight data | 328.1 | 268.4 | 185.7 | 230.0 |
| + deeper base B (v54) | 327.4 | 267.8 | 184.3 | 228.9 |

The step "ADS-B for flights without NM flight data" fixed an error: about 1 % of the departures have no `FLIGHT_ID_mvt`
because they could not be matched to an NM flight, and the ADS-B processing used to require that ID, so it silently
dropped these flights although their positions had been downloaded. Once a missing ID was allowed, 14 187 training
departures and 3 522 departures of the ranking file gained an ADS-B movement, of a quality comparable to the other
flights.

## Daily leaderboard (January and July 2026)

32 submissions were scored in total. The numbering has gaps (v6, v12, v15, v18–v35, v38, v42, v53) because the version
counter also advanced when a submission file was generated by mistake; these files were never submitted. The table lists
the main steps of the method; intermediate versions are left out.

| Version | RMSE | Change |
|---|---|---|
| v0–v8 | 426.1 → 347.9 | first single-model baselines and hyperparameter tuning; first ADS-B feature in v8 |
| v9 | 323.8 | CatBoost base model and further ADS-B features |
| v11 | 315.8 | classifier and mixture |
| v13 | 295.7 | corrector on out-of-fold base predictions, trained without artefacts |
| v16 | 291.1 | rules for extreme cases |
| v17 | 290.3 | missing visibility filled with the 2025 airport median |
| v37 | 289.4 | month feature in the corrector |
| v39 | 283.4 | ranking predictions from the same base model whose out-of-fold predictions the corrector learned from (before: an older base model) |
| v41 | 281.1 | NM timing features in the base model |
| v43 | 274.6 | base model refitted on all data instead of averaging the month models |
| v47 | 271.1 | second base model B, trained on normal flights only; LOBT window (−2.4 of the step) |
| v48 | 270.7 | traffic-flow and hour features in base B, three corrector seeds |
| v49 | 270.5 | queue and geometry features in base B |
| v50 | 265.8 | base A with CatBoost instead of LightGBM |
| **v54** | **259.5** | CatBoost classifier with stand features, base A depth 8, ADS-B for flights without NM flight data, deeper base B |
| v55 | 268.7 | final pipeline: corrector on all twelve months without month feature, CatBoost base B, no blend |

Up to v54 the corrector was trained and evaluated on January, July and December 2025 only, and it had a month feature,
which suits the two leaderboard months. v55 is the pipeline of this repository, built for all four months of the final
ranking.

## Data gaps in 2026

These gaps are in the source data. The pipeline handles them, but it cannot recover the missing information.

| Gap | Where | Handling |
|---|---|---|
| no visibility at NOAA in June and July 2026 | EDDM, LSZH (Dübendorf station), LTFM 0 %; LIRF 10 %; EDDF, EHAM, LEBL, LEMD, LFPG 32–51 %; EGLL 74–82 % | filled with the 2025 median of the airport ([method.md](method.md#prediction-for-new-data)) |
| hardly any ADS-B on the ground at LIRF in June and July 2026 | movement before take-off seen for 1 % and 4 % of the departures (2025: 71 % and 88 %); EDDF in February 2026: 34 % | the flight counts as not observed on the ground; 2025 has gaps of the same kind (LIRF January to April, LEMD in winter and summer), so the models learned with them |
| no current GHCNh station at Zurich airport | all of 2025 and 2026 | Dübendorf airfield, 10.5 km away, stands in ([data_sources.md](data_sources.md)) |

Temperature is complete in all four final months. The ADS-B coverage on the ground changes from month to month in both
years (`analysis/adsb_coverage.ipynb`); at EGLL it is much better in summer 2026 than in 2025.

## Tested without gain

| Idea | Result |
|---|---|
| Month as a feature in the base model or the classifier | cannot be evaluated out of fold, because the predicted month is always an unseen value; worse on the leaderboard when tried |
| Target encodings (artefact rate per stand or airline, long-taxi rate per stand) | worse in every combination |
| Arrival delay wave in the base model | worse |
| Traffic-flow, queue and geometry features in base A | no gain (they help base B) |
| Base A with depth 10 | January better, July worse |
| Separate base model for LIRF only | clearly worse than the global models at LIRF |
| Corrector trained including on-time pushbacks | worse |
| Corrector on the residual of base B instead of base A | worse |
| A second base A (LightGBM) as corrector input, corrector ensembles, deeper corrector | no gain |
| More corrector seeds (10 instead of 3) | no measurable gain |
| Mixture parameters per airport | no gain when chosen on one month and checked on the other |
| α and W fitted per calendar month | no gain when chosen on one half of each month and checked on the other |
| `p_artifact` recalibrated per airport | already calibrated, no gain |
| `p_artifact` below a threshold set to 0 | no gain |
| Artefact history of the same flight number, artefacts among the arrivals at the same time | no information beyond the classifier |
| Weather beyond temperature and visibility (wind, humidity, pressure, present weather) | no gain |
| Per-airport linear rule `y = a + b · delay` for long delays without NM data | worse at every threshold |
| Further hard label bounds (relative to SCHED, AOBT, EOBT, IOBT) | none holds like the LOBT window |
| Upper bound `prediction ≤ delay + x` for flights without NM data | worse in every month |
| ADS-B time `MVT − movement start` as a lower bound | 8 % of the labels lie below it; worse in every month, also when limited to short movements that start at the stand (`analysis/adsb_lower_bound.ipynb`) |
| Damping or amplifying the correction, `A + s · (C − A)` | slightly better overall, worse on normal flights |
| Learned linear combination of corrector, base A and base B | worse (+0.7) |
| Blend of the corrected value with base B | weight 0 chosen with the final base models |
