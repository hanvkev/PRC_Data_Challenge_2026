"""Training helpers for the model notebooks: MLflow runs, month blocks, fold-dependent features and refits."""
import fcntl
import json
import os
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import joblib
import mlflow
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import mean_squared_error
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
TARGET = "TAXITIME_SEC_mvt"
TIME = "SCHED_TIME_UTC_mvt"
EARLY_STOPPING_ROUNDS = 200
METADATA = ["ADEP_mvt", "MVT_ID_mvt", "FLIGHT_ID_mvt", TIME]
TRACKING_DIR = Path(__file__).resolve().parent / "mlruns"
SEED = 42
EXPERIMENT_NAME = "PRC Data Challenge 2026 - Experiments"
FEATURE_DIR = Path(__file__).resolve().parent / "features"


def _feature_module(name, source):
    module = ModuleType(name)
    exec(compile(source, f"{name}.py", "exec"), module.__dict__)
    return module


def _feature_values(module, frame, state, training=False):
    values = module.transform(frame.copy(), state, training=training)
    if not isinstance(values, pd.Series) or not values.index.equals(frame.index):
        raise ValueError(f"{module.__name__}.transform must return a Series with the input index.")
    return values


def transform_features(data, features, states, sources):
    """Rebuild model inputs from saved per-fold state, without fitting anything."""
    frame = data.drop(columns=[TARGET], errors="ignore").copy()
    for name, source in sources.items():
        module = _feature_module(name, source)
        inputs = frame.loc[:, module.REQUIRED_COLUMNS]
        frame[name] = _feature_values(module, inputs, states[name])
    return frame.loc[:, features]


def prepare_model_inputs(model, data):
    """CatBoost inputs: categorical columns as strings with one explicit missing value, all others as float."""
    frame = data.copy()
    categorical = set(model.get_params(deep=False).get("cat_features") or [])
    missing = categorical - set(frame.columns)
    if missing:
        raise ValueError(f"categorical features not in the model inputs: {sorted(missing)}")
    for column in frame.columns:
        if column in categorical:
            frame[column] = frame[column].astype("string").fillna("__MISSING__").astype(object)
        else:
            try:
                frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"feature {column!r} is not numeric but not listed in cat_features") from exc
    return frame


def configure_tracking():
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(TRACKING_DIR.as_uri())
    return mlflow.get_tracking_uri()


def _start_numbered_run(experiment_id, model_name):
    # A shared file lock also serializes numbering across notebook kernels.
    counter_path = TRACKING_DIR / experiment_id / ".run_number"
    with counter_path.open("a+") as counter:
        fcntl.flock(counter, fcntl.LOCK_EX)
        counter.seek(0)
        previous = int(counter.read().strip() or "0")
        client = mlflow.tracking.MlflowClient()
        count, highest, token = 0, previous, None
        while True:
            runs = client.search_runs(
                [experiment_id], run_view_type=mlflow.entities.ViewType.ALL,
                max_results=1000, page_token=token)
            count += len(runs)
            for run in runs:
                value = run.data.tags.get("run_number", "")
                if value.isdigit():
                    highest = max(highest, int(value))
            token = runs.token
            if not token:
                break
        number = max(count, highest) + 1
        counter.seek(0)
        counter.truncate()
        counter.write(str(number))
        counter.flush()
        return mlflow.start_run(
            experiment_id=experiment_id, run_name=f"{number:03d}_{model_name}",
            tags={"run_number": str(number)})


def _available_cpus():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _prepare_model(model, cpus, **changes):
    """Fresh, unfitted copy of a CatBoost model with the thread count and a fixed seed."""
    params = deepcopy(model.get_params(deep=False))
    params.update(thread_count=cpus, allow_writing_files=False, **changes)
    if params.get("random_seed") is None and params.get("random_state") is None:
        params["random_seed"] = SEED
    return type(model)(**params)


def _rmse(target, prediction):
    return float(np.sqrt(mean_squared_error(target, prediction)))


def _fold_features(training, validation, sources):
    training, validation = training.copy(), validation.copy()
    states = {}
    for feature_name, source in sources.items():
        module = _feature_module(feature_name, source)
        inputs = training.loc[:, list(dict.fromkeys([*module.REQUIRED_COLUMNS, TARGET]))]
        state = module.fit(inputs.copy())
        training[feature_name] = _feature_values(module, inputs, state, training=True)
        # Validation target is never passed to feature code.
        validation[feature_name] = _feature_values(
            module, validation.loc[:, module.REQUIRED_COLUMNS], state)
        states[feature_name] = state
    return training, validation, states


def _select_iterations(model, training, evaluation, features, sources):
    """Number of trees by early stopping on an evaluation month inside the training data."""
    training, evaluation, _ = _fold_features(training, evaluation, sources)
    patience = model.get_params().get("early_stopping_rounds") or EARLY_STOPPING_ROUNDS
    model.fit(prepare_model_inputs(model, training[features]), training[TARGET],
              eval_set=[(prepare_model_inputs(model, evaluation[features]), evaluation[TARGET])],
              early_stopping_rounds=patience, use_best_model=True)
    return model.get_best_iteration() + 1


def _refit_model(model, iterations, cpus):
    params = deepcopy(model.get_params(deep=False))
    for key in ["iterations", "n_estimators", "num_boost_round", "num_trees",
                "early_stopping_rounds", "od_type", "od_wait", "od_pval", "best_model_min_trees"]:
        params.pop(key, None)
    return _prepare_model(type(model)(**params), cpus, iterations=iterations, use_best_model=False)


def _fit_fold(name, model, data, features, train, valid, cpus, directory, sources, inner_split):
    start = time.perf_counter()
    inner_train, inner_eval = inner_split
    with threadpool_limits(limits=cpus):
        selected = _select_iterations(_prepare_model(model, cpus), data.iloc[inner_train], data.iloc[inner_eval],
                                      features, sources)
    model = _refit_model(model, selected, cpus)
    # Recompute all target-dependent state from the complete outer training set.
    training, validation, states = _fold_features(data.iloc[train], data.iloc[valid], sources)
    with threadpool_limits(limits=cpus):
        train_inputs = prepare_model_inputs(model, training[features])
        model.fit(train_inputs, training[TARGET])
        train_prediction = model.predict(train_inputs, thread_count=cpus)
        prediction = np.asarray(model.predict(prepare_model_inputs(model, validation[features]),
                                              thread_count=cpus)).reshape(-1)
    metrics = {f"train_rmse_{name}": _rmse(data.iloc[train][TARGET], train_prediction),
               f"rmse_{name}": _rmse(data.iloc[valid][TARGET], prediction),
               f"training_seconds_{name}": time.perf_counter() - start,
               f"best_iteration_{name}": selected}
    output = Path(directory)
    model_dir = output / "models" / name
    model_dir.mkdir(parents=True)
    joblib.dump(model, model_dir / "model.joblib")
    (model_dir / "hyperparameters.json").write_text(
        json.dumps(model.get_params(deep=True), default=str, indent=2))
    if sources:
        joblib.dump(states, model_dir / "feature_state.joblib")
    oof = data.iloc[valid][[col for col in METADATA if col in data]].copy()
    oof.insert(0, "row_id", valid)
    oof["fold"] = name
    oof["target"] = data.iloc[valid][TARGET].to_numpy()
    oof["prediction"] = prediction
    oof["residual"] = prediction - oof["target"]
    for feature_name in sources:
        oof[feature_name] = validation[feature_name].to_numpy()
    pd.DataFrame({"feature": features, "importance": model.feature_importances_}).to_parquet(
        output / f"feature_importance_{name}.parquet", index=False)
    return name, metrics, oof


# ---------------------------------------------------------------------------------------------------------------------
# Base models on month blocks (used by 06_base_a and 07_base_b)
# ---------------------------------------------------------------------------------------------------------------------

def month_blocks(oof_months=None):
    """One block per month of 2025; `oof_months` keeps only some of them (None = all twelve)."""
    return {f"m{m:02d}": [m] for m in range(1, 13) if oof_months is None or m in oof_months}


def inner_eval_month(months):
    """Early stopping uses the month right before the block (for January the month after it)."""
    return months[0] - 1 if months[0] > 1 else months[-1] + 1


def resolve_run(ref):
    """MLflow run from a 32-character run ID or a run number such as "246"."""
    import re
    ref = str(ref).strip()
    if re.fullmatch(r"[0-9a-f]{32}", ref):
        return mlflow.get_run(ref)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME], output_format="list",
                              filter_string=f"tags.run_number = '{int(ref.split('_')[0])}'")
    if len(runs) != 1:
        raise ValueError(f"Run {ref!r} not found uniquely ({len(runs)} matches).")
    return runs[0]


def artifact_dir(run):
    return Path(run.info.artifact_uri.removeprefix("file://"))


def train_blocks(run_name, model, data, features, sources, fit_mask, blocks, max_parallel=6, params=None):
    """Out-of-fold predictions for every block: a model per block, trained on the other months of 2025.

    `fit_mask` selects the rows a model may learn from (base B: normal flights only); every row of a block is
    predicted. Returns the MLflow run, the out-of-fold frame and the metrics.
    """
    month = pd.to_datetime(data[TIME], utc=True).dt.month.to_numpy()
    splits, inner = {}, {}
    for name, months in blocks.items():
        in_block = np.isin(month, months)
        train_rows, valid_rows = np.flatnonzero(fit_mask & ~in_block), np.flatnonzero(in_block)
        splits[name] = (train_rows, valid_rows)
        es = month[train_rows] == inner_eval_month(months)
        inner[name] = (train_rows[~es], train_rows[es])
    available = _available_cpus()
    workers = max(1, min(max_parallel, len(blocks), available))
    cpus = max(1, available // workers)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    started = time.perf_counter()
    with _start_numbered_run(experiment.experiment_id, run_name) as run, tempfile.TemporaryDirectory() as directory:
        print(f"Run {run.info.run_name}: {len(blocks)} blocks, {workers} in parallel with {cpus} CPUs each", flush=True)
        mlflow.log_params({"model_type": type(model).__name__, "pipeline": "blocks", "block_months": 1,
                           "n_blocks": len(blocks), "feature_count": len(features), "fit_rows": int(fit_mask.sum()),
                           "early_stopping_rounds": model.get_params().get("early_stopping_rounds") or EARLY_STOPPING_ROUNDS,
                           **(params or {})})
        mlflow.log_dict(blocks, "block_months.json")
        mlflow.log_dict(list(features), "features.json")
        mlflow.log_dict(list(sources), "features_dynamic.json")
        for name, source in sources.items():
            mlflow.log_text(source, f"feature_code/{name}.py")
        mlflow.log_dict(json.loads(json.dumps(_prepare_model(model, cpus).get_params(deep=True), default=str)),
                        "hyperparameters.json")
        results = {}
        with parallel_config(backend="loky", inner_max_num_threads=cpus):
            with Parallel(n_jobs=workers, return_as="generator_unordered") as parallel:
                for name, metrics, frame in parallel(
                        delayed(_fit_fold)(name, _prepare_model(model, cpus), data, features, tr, va, cpus, directory,
                                           sources, inner[name])
                        for name, (tr, va) in splits.items()):
                    results[name] = (metrics, frame)
                    print(f"[{len(results)}/{len(blocks)}] {name}: RMSE {metrics[f'rmse_{name}']:.1f}, "
                          f"trees {metrics.get(f'best_iteration_{name}')}, {(time.perf_counter() - started) / 60:.0f} min",
                          flush=True)
        oof = pd.concat([results[n][1] for n in blocks], ignore_index=True)
        metrics = {k: v for n in blocks for k, v in results[n][0].items()}
        metrics["rmse_oof"] = _rmse(oof["target"], oof["prediction"])
        metrics["training_seconds"] = time.perf_counter() - started
        mlflow.log_metrics(metrics)
        oof.to_parquet(Path(directory) / "oof_predictions.parquet", index=False)
        mlflow.log_artifacts(directory)
    return run, oof, metrics


def refit(run, model, data, features, sources, fit_mask, metrics, blocks, extra=None):
    """Final model on all allowed 2025 rows; trees = median of the block early-stopping results. Saved under refit/."""
    cpus = _available_cpus()
    trees = int(np.median([metrics[f"best_iteration_{n}"] for n in blocks]))
    final = _refit_model(model, trees, cpus)
    rows = data.loc[fit_mask]
    training, _, state = _fold_features(rows, rows.iloc[:0], sources)
    started = time.perf_counter()
    final.fit(prepare_model_inputs(final, training[features]), training[TARGET])
    meta = {"features": list(features), "features_dynamic": list(sources), "trees": trees, "fit_rows": len(rows),
            "refit_seconds": time.perf_counter() - started, **(extra or {})}
    client = mlflow.tracking.MlflowClient()
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        joblib.dump(final, directory / "model.joblib")
        joblib.dump(state, directory / "feature_state.joblib")
        (directory / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        client.log_artifacts(run.info.run_id, str(directory), "refit")
    return final, meta


def refit_folder(run):
    return artifact_dir(run) / "refit"


def predict_refit(folder, frame):
    """Prediction of a saved base model (an MLflow refit/ folder or its copy in model/final) for any prepared dataset."""
    folder = Path(folder)
    model = joblib.load(folder / "model.joblib")
    state = joblib.load(folder / "feature_state.joblib")
    meta = json.loads((folder / "metadata.json").read_text())
    sources = {name: (folder / f"{name}.py").read_text() for name in meta["features_dynamic"]}
    frame = frame.copy()
    for column, medians in meta.get("fill_medians", {}).items():   # e.g. visibility when a station stops reporting it
        frame[column] = frame[column].fillna(frame["ADEP_mvt"].map(medians))
    inputs = transform_features(frame, meta["features"], state, sources)
    return np.clip(np.asarray(model.predict(prepare_model_inputs(model, inputs))).reshape(-1), 0, None)


def export_final(runs, corrector, folder):
    """Copy the final models into one versioned folder (compressed, so each file stays below GitHub's 100 MB)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for key, name in [("A", "base_a"), ("B", "base_b")]:
        target = folder / name
        target.mkdir(exist_ok=True)
        source = refit_folder(runs[key])
        joblib.dump(joblib.load(source / "model.joblib"), target / "model.joblib", compress=3)
        joblib.dump(joblib.load(source / "feature_state.joblib"), target / "feature_state.joblib", compress=3)
        meta = json.loads((source / "metadata.json").read_text())
        (target / "metadata.json").write_text(json.dumps(meta, indent=2))
        for dynamic in meta["features_dynamic"]:
            (target / f"{dynamic}.py").write_text((artifact_dir(runs[key]) / "feature_code" / f"{dynamic}.py").read_text())
    joblib.dump(joblib.load(artifact_dir(runs["C"]) / "models/final/classifier.joblib"), folder / "classifier.joblib", compress=3)
    joblib.dump(corrector, folder / "corrector.joblib", compress=3)
    manifest = {key: {"run_id": run.info.run_id, "run_name": run.info.run_name} for key, run in runs.items()}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {p.relative_to(folder).as_posix(): round(p.stat().st_size / 1e6, 1) for p in sorted(folder.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------------------------------------------------------------
# Post-processing shared by 09_corrector and the prediction notebooks
# ---------------------------------------------------------------------------------------------------------------------

def mix(corrected, p_artifact, delay, alpha, w):
    """Expected value over the two label regimes: a real taxi time or a label equal to the delay."""
    return corrected + w * np.power(p_artifact, alpha) * (delay - corrected)


def lobt_window(frame, values, window_sec=3606):
    """Every 2025 label with an LOBT lies within MVT - LOBT +/- window_sec; move predictions inside."""
    gap = (pd.to_datetime(frame["MVT_TIME_UTC_mvt"], utc=True)
           - pd.to_datetime(frame["LOBT_flt"], utc=True)).dt.total_seconds().to_numpy()
    values = np.asarray(values, dtype=float)
    return np.where(np.isnan(gap), values, np.clip(values, np.maximum(gap - window_sec, 0), gap + window_sec))


def _lirf_without_nm(frame):
    return frame["ADEP_mvt"].eq("LIRF") & frame["aobt_missing"].eq(1)


RULES = {   # a row is handled by the first rule that matches
    "LIRF without NM, delay 60k-95k s": lambda f: _lirf_without_nm(f) & f["delay_sec"].between(60000, 95000, inclusive="right"),
}
RULE_CLASS_TOL = 120


def rule_estimate(rows):
    """Shares of the label types in 2025 for one situation: label = delay, = MVT − AOBT, = about a day, other."""
    y, delay, aobt = rows[TARGET], rows["delay_sec"], rows["mvt_minus_aobt_sec"]
    is_delay = (y - delay).abs() <= RULE_CLASS_TOL
    is_aobt = ~is_delay & ((y - aobt).abs() <= RULE_CLASS_TOL)
    is_day = ~is_delay & ~is_aobt & (y - 86400).between(0, 3600)
    is_other = ~(is_delay | is_aobt | is_day)
    n = max(len(rows), 1)
    return {"n": len(rows), "q_delay": is_delay.sum() / n, "q_aobt": is_aobt.sum() / n, "q_day": is_day.sum() / n,
            "q_other": is_other.sum() / n, "day_value": float(y[is_day].median()) if is_day.any() else 87300.0,
            "other_value": float(y[is_other].mean()) if is_other.any() else float(y.median()) if len(y) else 900.0}


def rule_predict(frame, estimate):
    aobt = frame["mvt_minus_aobt_sec"].fillna(frame["delay_sec"])
    return (estimate["q_delay"] * frame["delay_sec"] + estimate["q_aobt"] * aobt
            + estimate["q_day"] * estimate["day_value"] + estimate["q_other"] * estimate["other_value"])


def apply_rules(frame, values, estimates):
    """Replace `values` for rows matching a rule; `estimates` maps rule name -> rule_estimate (rules not listed stay off)."""
    values, taken = np.asarray(values, dtype=float).copy(), np.zeros(len(frame), dtype=bool)
    for name, condition in RULES.items():
        rows = condition(frame).to_numpy() & ~taken
        if name in estimates and estimates[name]["n"] >= 3 and rows.any():
            values[rows] = rule_predict(frame.loc[rows], estimates[name]).to_numpy()
        taken |= rows
    return values


def predict(frame, folder):
    """Final prediction for prepared departures with the models in `folder` (model/final). Returns a copy of `frame` with
    both base predictions, p_artifact, the corrected value and the final `prediction`."""
    folder = Path(folder)
    corrector = joblib.load(folder / "corrector.joblib")
    classifier = joblib.load(folder / "classifier.joblib")
    frame = frame.copy()
    frame["base_prediction"] = predict_refit(folder / "base_a", frame)
    frame["base_normal_prediction"] = predict_refit(folder / "base_b", frame)

    X = frame[classifier["features"]].copy()
    X[classifier["categorical"]] = X[classifier["categorical"]].astype("string").fillna("__MISSING__").astype(object)
    candidate = (frame["delay_sec"] > classifier["candidate_min_delay_sec"]).to_numpy()
    frame["p_artifact"] = 0.0
    frame.loc[candidate, "p_artifact"] = classifier["model"].predict_proba(X.loc[candidate])[:, 1]

    X = frame[corrector["features"]].copy()
    X[corrector["categorical"]] = X[corrector["categorical"]].astype("string").fillna("__MISSING__").astype(object)
    X = X.astype({c: "float64" for c in corrector["features"] if c not in corrector["categorical"]})
    frame["corrected"] = frame["base_prediction"] + np.mean([m.predict(X) for m in corrector["models"]], axis=0)
    values = mix(frame["corrected"], frame["p_artifact"], frame["delay_sec"], corrector["alpha"], corrector["w"])
    values = apply_rules(frame, values, corrector["rules"])
    frame["prediction"] = np.clip(lobt_window(frame, values, corrector["lobt_window_sec"]), 0, None)
    return frame
