# Features

All features are built in `data/05_features.ipynb` and only use what is also known for the ranking files: movement
and schedule times of all flights, NM flight times, arrivals with their block times, weather, ADS-B and airport
geometry. A departure's own label and block time are blanked when the data are loaded. Counts and queues of a file are
built from the movements of that file, with the 2025 movements as history before its first day.

The one fold-dependent feature, `median_taxi_out` (median taxi time per airport, stand and runway), is fitted inside
each training fold (`model/features/median_taxi_out.py`, leave-one-out on the training rows).

`05_features` writes only features that one of the models uses. Ideas that were tried and dropped are listed in
[results.md](results.md#tested-without-gain).

## Who uses what

A = base A, B = base B, C = classifier, K = corrector.

| Group | Features | A | B | C | K |
|---|---|---|---|---|---|
| challenge data | `ADEP_mvt`, `RUNWAY_mvt`, `AIRCRAFT_TYPE_mvt`, `AIRCRAFT_OPERATOR_flt` | ✓ | ✓ | ✓ | airport, runway, type |
| | `STAND_mvt`, `ADES_mvt`, `MARKET_SEGMENT_flt` | | | ✓ | stand |
| | `WK_TBL_CAT_flt` (wake category) | | ✓ | | |
| schedule and NM times | `movement_minus_scheduled_offblock_sec` = `delay_sec` (MVT − SCHED), `mvt_minus_aobt_sec` | ✓ | ✓ | ✓ | ✓ |
| | `mvt_minus_lobt_sec`, `aobt_minus_eobt_sec`, `lobt_minus_eobt_sec` | ✓ | ✓ | AOBT − EOBT | |
| | MVT − EOBT / IOBT / ARVT1 / ARVT3, SCHED − EOBT / IOBT, AOBT − SCHED | | | ✓ | MVT − EOBT, SCHED − EOBT, AOBT − SCHED |
| | `aobt_missing`, `nm_missing`, `carrier` (airline prefix of the flight number), `mvt_hour` (UTC) | | | ✓ | ✓ (not `nm_missing`) |
| | `aobt_has_seconds` (AOBT not on a full minute) | | ✓ | | |
| local time | `movement_day_of_week_sin/cos`, `local_hour` | weekday | ✓ | | |
| traffic | `deps_before_30min`, `deps_before_30min_same_runway`, `deps_after_30min_same_runway`, `seconds_since_last_movement` | ✓ | ✓ | | |
| departure flow | `dep_median_delay_30min_sec` (delay wave of the other departures) | ✓ | ✓ | | |
| | `dep_rate_60min_ratio`, `dep_runway_share_30min`, `dep_n_runways_60min`, `deps_same_runway_10min`, `dep_max_gap_2h_sec` | | ✓ | | |
| queue in front | `nbr_mvt_minus_aobt_30min_sec`, `…_60min_sec`, `…_rwy_60min_sec` (how long the other departures needed from off-block to take-off) | | ✓ | | |
| | `deps_between_offblock_and_mvt`, `deps_same_runway_between_offblock_and_mvt`, `taxi_queue_at_offblock`, `gate_retention_at_offblock`, `arr_taxiing_at_offblock` | | ✓ | | |
| | `dep_runway_config_60min`, `arr_runway_config_60min` (runways used in the last hour) | | ✓ | | |
| stand and inbound | `stand_reocc_gap_sec`, `stand_reocc_other_type`, `stand_prev_arr_gap_sec` | ✓ | ✓ | ✓ | gap |
| | `stand_inbound_minus_sched_sec`, `stand_first_arr_after_sched_gap_sec`, `stand_n_arr_sched_to_mvt`, `arr_taxi_in_last_hour_sec` | | | ✓ | |
| weather (NOAA GHCNh) | `weather_temperature_c`, `weather_visibility_km` (latest report at most 2 h before take-off; Zurich uses the Dübendorf station) | ✓ | ✓ | | |
| geometry (OSM, X-Plane) | `stand_to_runway_distance_m`, `stand_runway_parallel_m`, `stand_runway_perpendicular_m`, `taxi_route_distance_m`, `runway_length_m` | | ✓ | | |
| ADS-B | `adsb_movement_to_mvt_sec` (take-off minus first sustained ground movement) | ✓ | ✓ | ✓ | ✓ |
| | `adsb_first_observation_to_mvt_sec`, `adsb_movement_to_runway_direct_m` | ✓ | ✓ | | |
| | `adsb_movement_to_gate_direct_m` | | | ✓ | |
| fold-dependent | `median_taxi_out` | ✓ | ✓ | | |
| first stage | base A and B predictions, `p_artifact` | | | | ✓ |

## Notes on a few features

* **Stand and inbound.** Arrival block times are public. An aircraft that parked at the stand shortly before the
  take-off suggests that the departure had already left, which hints at a longer taxi time. The time at which the likely
  own inbound aircraft reached the stand, compared with the scheduled off-block, helps to tell genuine taxi times from
  labels that equal the delay, especially where NM data are missing. Both are only hints, not exact rules: when a flight
  is heavily delayed because its own aircraft has not arrived yet, for example, other aircraft may park at the stand and
  leave again before it departs, so an arrival at the stand does not prove that the departure had already left.
* **Queue in front.** The off-block reference is the own AOBT, or the EOBT when AOBT is missing. Only NM times of the
  other departures are used, never their labels.
* **ADS-B.** The first sustained movement (above 2 kn for at least 30 s) of the trajectory selected by
  `04_adsb_movement`. At LFPG and LTFM, and in some months at EGLL, LEMD and LIRF, aircraft are seen
  only after the take-off; the value is then negative and carries no taxi information (`analysis/adsb_coverage.ipynb`).
  Even before the take-off the movement is usually seen only away from the stand, so the value is a hint rather than
  the answer. For some flights the detected movement starts well before the recorded off-block, so it is not a
  safe lower bound either (`analysis/adsb_lower_bound.ipynb`); a stricter movement definition, such as a higher speed
  threshold, might help but was not tested. Departures without NM flight data are processed like all others.
* **Weather.** GHCNh has no current station at Zurich airport, so the reports of Dübendorf airfield (10.5 km away)
  stand in for it. Visibility that a station stops reporting, as several did in June and July 2026, is filled at
  prediction time with the 2025 median of the airport.
* **Geometry.** Stand positions come from OpenStreetMap, X-Plane serves as fallback; runways and the taxiway network
  come from X-Plane. The taxi route is the shortest path over the network from the stand to the threshold.
