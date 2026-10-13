# Method

This document explains how the pipeline works and why it is built this way. It follows the pipeline from the data to the
prediction; for every part it first describes what the part does and then which results led to it. All numbers are
out-of-fold results over the twelve months of 2025 unless stated otherwise; the complete tables are in
[results.md](results.md).

## The problem

The task is to predict the taxi-out time of every departure, that is the time between off-block and take-off,
`y = MVT − BLOCK`. Submissions are scored by their RMSE.

The 2025 training data contain this label for every departure. In the ranking files the label and the off-block time of
the departures are hidden, while all other columns remain available, including the NM flight times and the arrivals with
their in-block times ([column description](https://prc-data-challenge-2026.netlify.app/data.html)).

Two terms appear throughout this document. The *delay* is `MVT − SCHED`, the time from the scheduled off-block to the
take-off. It equals the taxi time only when the aircraft leaves the stand exactly on schedule: a late off-block makes
the delay longer than the taxi time, an early one makes it shorter, and in 2025 about one departure in five left the
stand more than a minute early. A prediction is called *out of fold* when it comes from a model that did not see the
flight's month during training.

## The labels

Comparing the label with the delay divides the 2025 departures into four groups (`analysis/label_regimes.ipynb`):

| Kind | Definition | Share | Median label |
|---|---|---|---|
| normal | none of the three groups below | 90.4 % | 910 s |
| on-time pushback | label within 60 s of the delay, delay up to 2000 s: the recorded off-block equals the scheduled time, and the label is usually a genuine taxi time | 9.2 % | 927 s |
| artefact | label within 60 s of the delay, delay above 2000 s: usually an off-block recorded at the scheduled time although the aircraft left later | 0.34 % | 2577 s, up to about a day |
| long taxi | label above two hours that is not an artefact, including labels shifted by about one day | 0.01 % | 8330 s, up to about a day |

The limit of 2000 s between on-time pushbacks and artefacts comes from the data. A label that equals the delay is a
genuine taxi time only if the aircraft really left at its scheduled time, and in that case the NM data agree:
`MVT − AOBT`, measured from the actual off-block time reported to NM, then matches the label as well. For delays up to
about 1500 s most of these labels pass this check, whereas from 2000 s onwards about four out of five fail it, so above
that limit a label equal to the delay is usually an artefact. A higher limit such as 3000 s would treat the flights
between 2000 s and 3000 s, most of which are artefacts, as genuine taxi times.

Artefacts and long taxis are rare, but their labels are so large that they weigh heavily under squared error: a single
miss of three hours counts as much as about 3000 misses of 200 s, which is a typical error for a normal flight.

The artefacts are concentrated at one airport. Of all artefacts, 62 % occur at LIRF, where 2.7 % of the departures are
artefacts, compared with at most 0.4 % at any other airport. Among flights without NM data and with a delay above 2000 s
the contrast is even sharper: 49 % of them are artefacts at LIRF, at most 2 % elsewhere.

## The approach

A single regression model trained on all labels has to cover both a genuine taxi time and a label equal to the delay
with one number, and under squared error it is pulled towards the large labels at the expense of the normal flights. The
pipeline therefore separates two questions.

- **How likely is it that the label equals the delay?** The classifier answers this with the probability `p_artifact`.
- **What is the label if it does not equal the delay?** Base A, trained on all labels, provides a starting point, and
  base B, trained only on normal flights, a precise estimate for normal traffic. The corrector receives both predictions
  together with `p_artifact`, learns only from labels that do not equal the delay, and turns them into the value
  `corrected`.

The mixture then uses `p_artifact` a second time, as the weight between the two cases,
`(1 − p_artifact) · corrected + p_artifact · delay`, so each flight's own delay enters the prediction exactly. Two
safety nets follow at the end: a rule for one extreme situation at LIRF and the LOBT window.

```
         2025 departures with features from data/05_features
                              │
     ┌────────────────────────┼─────────────────────────┐
  base A (CatBoost)      base B (CatBoost)        classifier (CatBoost)
  all flights            normal flights           p_artifact
     └──────── out-of-fold predictions for every month ──┘
                              │
                 corrector (CatBoost, 3 seeds)
                 learns y − base A, without labels = delay
                              │
          mixture: corrected + W · p_artifact^α · (delay − corrected)
                              │
              rule for one extreme case → LOBT window → ≥ 0
```

Separating the two questions is worth it. Base A on its own, a single model trained on all labels, reaches an RMSE of
262.2 s over the year. Combining it with the classifier through the mixture lowers this to 247.3 s and improves 9 of the
12 months. The corrector, which also brings in base B, together with the rule and the LOBT window, then brings the RMSE
down to 236.5 s ([results.md](results.md#what-each-step-contributes)). Without base B as an input, the corrector would
only reach 241.9 s.

Long taxis do not equal the delay, so they belong to the second question and are predicted by base A and the corrector.
The models predict them poorly, and they remain the largest source of error: in the final out-of-fold predictions they
carry 64 % of the squared error in January 2025, 58 % in May and 49 % in June, while the artefacts cause less than 1 %
in every month (`analysis/error_analysis.ipynb`).

## Training month by month

The corrector learns from the predictions of the two base models and the classifier. If these predictions came from
models that had already seen the same flights, their errors would be unrealistically small, and the corrector would
learn the wrong corrections. Each month of 2025 is therefore predicted by models trained on the other eleven months.

Whole months are held out rather than randomly chosen flights, because flights on the same day share their weather,
runway configuration and disruptions; with a random split, a model would already have seen the same situation during
training. Early stopping uses the month before the held-out one (for January the month after it), so the held-out month
plays no part in any choice. Since every month of 2025 has out-of-fold predictions, the corrector learns from all
seasons, and the choices of the final phase, such as α, W, the rules and the model variants, are made on all twelve
months together.

For the prediction of new data every model is trained once more on all of 2025: the base models with the median number
of trees of their twelve month models (`runner.train_blocks`, `runner.refit`), the classifier on all departures with a
delay above 2000 s and the corrector on all out-of-fold rows. For the base models this step cannot be evaluated out of
fold, because the final model has seen every month. It was chosen because a single model learns from all twelve months
instead of eleven and avoids averaging twelve models that stopped at different numbers of trees.

## How choices were made

The development went through two phases. At the start, models were developed only on January and July 2025, the months
that the daily leaderboard covers in 2026, plus December: out-of-fold predictions were made only for these months, the
corrector learned only from them, and variants were compared on January and July. This made each experiment fast, but it
ties the choices to the leaderboard months. In the final phase the whole pipeline was moved to all twelve months, and
from then on every choice was made on all twelve months. The reduced comparisons were only used where a choice from the
first phase was kept without being checked again; this concerns the depth of base A. The other choices that matter for
the final ranking were checked again on all twelve months: base B and its role as an input of the corrector, the tree
limits, the stand features of the classifier, the months the corrector learns from, the blend and the rules.

- Every comparison was made out of fold, month by month, both overall and on labels up to one hour, and separately for
  flights with and without an NM off-block time.
- A small difference was accepted only if it pointed in the same direction in most months.
- The daily leaderboard (January and July 2026) served to confirm candidates, not to tune parameters. Most ideas were
  rejected out of fold before any submission; they are listed in [results.md](results.md#tested-without-gain).
- Running notebooks 06 to 11 again reproduces the out-of-fold figures, the models and the submission files exactly.

## The models and steps

### Base A

**What it does.** Base A is a CatBoost model with depth 8 and a learning rate of 0.05, trained with early stopping and
at most 20 000 trees on all 2025 departures. It uses 24 features together with the median taxi time per airport, stand
and runway, which is computed from the training months only ([features.md](features.md)). Because base A also learns the
extreme labels, it serves as the starting point for the corrector.

**Why this setup.** Base A was a LightGBM model in early development; CatBoost performed better and replaced it. The two
libraries were not compared again for base A over all twelve months. A depth of 8 instead of 6 lowered the out-of-fold
RMSE by 1.2 s in January and by 5.2 s in July 2025. The tree limit was raised until no month model reached it any more;
the largest now needs 14 200 trees, and compared with the earlier limit of 10 000 only the three months that had hit
that limit changed, by at most 0.6 s.

### Base B

**What it does.** Base B is a CatBoost model with the same settings, but it is trained only on normal flights, that is
on labels of up to one hour that are not artefacts. It nevertheless predicts every flight and answers the question of
how long the taxi would take if the flight were normal. Besides all features of base A it uses features that describe
the current traffic and the airport layout, namely the departure flow, the queue in front of the flight, the runway
configuration and the distances between stand and runway, and a few others such as the wake category and the local hour
([features.md](features.md)).

**Why.** Freed from the extreme labels, base B is clearly more precise on normal traffic: on labels up to one hour its
RMSE is 179.5 s, compared with 196.3 s for base A. This precision makes it a useful second input for the corrector: a
corrector trained without base B reaches 241.9 s instead of 236.5 s over the year and 193.5 s instead of 186.8 s on
labels up to one hour, and it is worse in every month. The traffic features improve base B but did not improve base A,
which is why the two models use different feature sets. For most of the development base B was a LightGBM model. Once
the tree limit was high enough for both libraries, the CatBoost version of base B turned out slightly better than the
LightGBM version: 179.5 s instead of 180.4 s on labels up to one hour, better in 10 of 12 months, and after the
corrector 236.5 instead of 236.8 over the year. The difference is small, but every comparison pointed in the same
direction.

### Classifier

**What it does.** The classifier is a CatBoost model (1500 iterations, depth 6, learning rate 0.05, log loss) that
estimates the probability `p_artifact` that the label equals the delay to within 60 s. It only considers departures with
a delay above 2000 s and sets `p_artifact = 0` for all others. Its strongest signals are the differences between the
schedule and the NM times and the airport. It also uses the stand: when the probable inbound aircraft reached the stand
relative to the scheduled off-block, and how many aircraft arrived there between the scheduled off-block and the
take-off.

**Why.** Below 2000 s a label equal to the delay is usually a genuine taxi time, and since the delay is then itself a
plausible taxi time, knowing that the two coincide would change the prediction very little. Above the limit such a label
is mostly an artefact, but not always; for the mixture this does not matter, because it only needs to know whether the
label equals the delay, not why. Out of fold the classifier reaches an AUC between 0.94 and 0.97 in every month, and its
probabilities are well calibrated: in each range of `p_artifact` the share of flights whose label equals the delay
matches the mean predicted probability. Recalibrating them per airport brought no gain, so they enter the mixture
unchanged. Without the stand features the AUC is lower in 11 of the 12 months, and the RMSE of the pipeline rises from
236.5 to 237.2 over the year and is worse in 9 of the 12 months, with the corrector left unchanged.

### Corrector

**What it does.** While the base models judge each flight on its own, the corrector learns where base A is
systematically wrong, given what base B and the classifier say about the same flight. It is a CatBoost model with 1500
iterations, depth 6 and an L2 regularisation of 10, averaged over three random seeds, and its target is the difference
`y − base A`. Its inputs are the predictions of both base models and `p_artifact`, together with airport, stand, runway,
airline, aircraft type, delay, the NM time differences, the UTC hour, the ADS-B movement time and the stand
re-occupation gap. It learns from all twelve months of 2025, but not from flights whose label equals the delay, and it
is not told which month a flight belongs to.

**Why.** The corrector is one of the largest steps: without it, that is with base A passed directly to the mixture, the
RMSE rises from 236.5 to 244.7 over the year and from 186.8 to 200.8 on labels up to one hour, and every month gets
worse. Labels equal to the delay are left out because the mixture adds that case afterwards, weighted with `p_artifact`;
`corrected` therefore has to estimate the label for the other case, otherwise the delay would be counted twice.
Correcting base B instead of base A gave worse results, probably because base B has never seen the extreme labels and
therefore makes very large errors on them.

**Why all months and no month feature.** A month feature cannot be evaluated out of fold, and for a month that was not
part of the corrector's training it carries no information. The alternative of a corrector that learns only from
January, July and December, as the corrector of our best daily-leaderboard file did, was compared out of fold with the
same base models and without month feature. It is slightly better in January and July 2025 (by 0.9 s and 0.6 s), but
clearly worse in other months, most of all in June (by 6.3 s), and worse over the year (237.9 instead of 236.5) and on
labels up to one hour (188.6 instead of 186.8). Since the final ranking also covers February and June, the corrector
learns from every month. That leaderboard file (v54, 259.55) also had a month feature and older base models; the final
models score 268.68 on the January/July leaderboard.

### Mixture

**What it does.** The mixture combines `corrected` and the delay into the prediction
`corrected + W · p_artifact^α · (delay − corrected)`, with α and W chosen on the mean RMSE over the twelve months. The
result, α = 1 and W = 1, is the plain expected value `(1 − p_artifact) · corrected + p_artifact · delay`.

**Why.** If `p_artifact` is the probability that the label equals the delay and `corrected` is the right prediction for
any other label, this expected value minimises the squared error. Within the final pipeline the mixture is the largest
single step: without it the RMSE over the year would be 268.7 instead of 236.5, and every month is better with it.
Without the mixture the pipeline is even worse than base A alone (262.2), because the corrector never learned labels
equal to the delay and so treats every artefact like a normal flight. On labels up to one hour the mixture costs a
little (186.8 instead of 184.9), because normal flights with a large delay are pulled slightly towards their delay.

### Rule for an extreme case

**What it does.** At LIRF a handful of flights without an NM off-block time have a delay of 60 000 to 95 000 s, roughly
17 to 26 hours. For these flights the prediction is replaced by the expected value over the two label types seen in
2025, weighted by how often each occurred; in the out-of-fold evaluation these frequencies are estimated without the
month being predicted.

**Why.** There were only ten such flights in 2025: for two of them the label was the delay, for the other eight it was
about one day plus a normal taxi time, as if the off-block had been recorded a day too early. The models predict
something in between, which is wrong in either case. Although the rule rests on very few flights, the two label types
are clearly distinct, and the rule lowers the RMSE over the year from 238.7 to 236.5 and improves 6 of the 7 months in
which it applies. Two similar rules, one for flights whose NM off-block lies more than 10 000 s before the take-off with
a delay above 10 000 s, and one for LIRF flights without NM off-block and a delay of 50 000 to 60 000 s, made no
measurable difference over the year and were removed.

### LOBT window

**What it does.** Every prediction is moved into the window `[max(MVT − LOBT − 3606, 0), MVT − LOBT + 3606]`, where LOBT
is the last known off-block time in the NM flight data. Predictions below zero are set to zero.

**Why.** In 2025, every one of the 2.06 million departures with an LOBT left the stand within 3606 s of it, at every
airport and in every month (`analysis/lobt_window.ipynb`). The window lowers the RMSE from 237.0 to 236.5 and improves
every month. Comparable limits relative to SCHED, AOBT, EOBT or IOBT do not hold in the data.

## Prediction for new data

`09_corrector` stores the four final models in `model/final`. The notebooks `10_predict_ranking` and `11_predict_final`
only load them and apply the steps above through `runner.predict`, so the daily-leaderboard file and the final
submission are produced by exactly the same models.

Some weather stations stop reporting visibility in 2026: for June and July, NOAA has no visibility at all for EDDM, LSZH
and LTFM and only partial values for the other airports ([results.md](results.md#data-gaps-in-2026)). At prediction time
such gaps are filled with the airport's median visibility in 2025, which is stored with the base models; this is done
only for airports where 2025 itself has visibility for at least half of the flights. The filling cannot be evaluated out
of fold, because 2025 contains no gaps of this kind. It keeps the input within the values the models learned for the
airport, whereas a station that stops reporting entirely is a situation they never saw.