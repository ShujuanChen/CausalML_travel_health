from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torchtuples as tt
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH
from sklearn.model_selection import StratifiedKFold, train_test_split

from deepsurv.standalone.losses import WeightedCoxPHLoss
from deepsurv.standalone.model import DeepSurvNet


class WeightedCoxPH(CoxPH):
    def __init__(self, net, optimizer=None, device=None, loss=None):
        super().__init__(net=net, optimizer=optimizer, device=device)
        self.loss = WeightedCoxPHLoss() if loss is None else loss


@dataclass
class TrainConfig:
    input_dim: int
    hidden_dim: int = 64
    input_dropout: float = 0.3
    batch_size: int = 1024
    epochs: int = 100
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4
    lr_decay_factor: float = 0.1
    n_splits: int = 5
    random_state: int = 42
    use_weighted_loss: bool = False
    verbose: bool = True


def make_model(config):
    net = DeepSurvNet(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        input_dropout=config.input_dropout,
    )
    optimizer = tt.optim.Adam(config.learning_rate)

    if config.use_weighted_loss:
        return WeightedCoxPH(net, optimizer=optimizer)
    return CoxPH(net, optimizer=optimizer)


def _make_target(durations, events, weights=None):
    if weights is None:
        return (durations, events)
    return (durations, events, weights)


def _fit_one_stage(model, x_train, target_train, x_val, target_val, config):
    callbacks = [tt.callbacks.EarlyStopping()]
    return model.fit(
        x_train,
        target_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        callbacks=callbacks,
        val_data=(x_val, target_val),
        verbose=config.verbose,
    )


def _learning_rate_schedule(config):
    learning_rates = [config.learning_rate]
    current_lr = config.learning_rate

    while True:
        next_lr = current_lr * config.lr_decay_factor
        if next_lr < config.min_learning_rate:
            break
        learning_rates.append(next_lr)
        current_lr = next_lr

    if learning_rates[-1] != config.min_learning_rate:
        learning_rates.append(config.min_learning_rate)

    return list(dict.fromkeys(learning_rates))


def _epochs_completed(training_log):
    if training_log is None:
        return None

    if hasattr(training_log, "to_pandas"):
        try:
            return len(training_log.to_pandas())
        except Exception:
            pass

    for attr_name in ("epoch", "epochs"):
        if hasattr(training_log, attr_name):
            value = getattr(training_log, attr_name)
            if isinstance(value, int):
                return value
            try:
                return len(value)
            except TypeError:
                pass

    if hasattr(training_log, "monitors"):
        monitors = getattr(training_log, "monitors")
        if isinstance(monitors, dict):
            for monitor_value in monitors.values():
                if isinstance(monitor_value, dict):
                    for series in monitor_value.values():
                        try:
                            return len(series)
                        except TypeError:
                            continue
                else:
                    try:
                        return len(monitor_value)
                    except TypeError:
                        continue

    try:
        return len(training_log)
    except TypeError:
        return None


def fit_single_split(
    x_train,
    durations_train,
    events_train,
    x_val,
    durations_val,
    events_val,
    weights_train=None,
    weights_val=None,
    config=None,
):
    if config is None:
        config = TrainConfig(input_dim=x_train.shape[1])

    model = make_model(config)
    train_target = _make_target(durations_train, events_train, weights_train if config.use_weighted_loss else None)
    val_target = _make_target(durations_val, events_val, weights_val if config.use_weighted_loss else None)

    logs = []

    for stage_index, current_lr in enumerate(_learning_rate_schedule(config)):
        model.optimizer.set_lr(current_lr)
        stage_log = _fit_one_stage(
            model=model,
            x_train=x_train,
            target_train=train_target,
            x_val=x_val,
            target_val=val_target,
            config=config,
        )
        logs.append(stage_log)

        completed_epochs = _epochs_completed(stage_log)
        early_stopped = (
            completed_epochs is not None and completed_epochs < config.epochs
        )

        is_last_stage = stage_index == len(_learning_rate_schedule(config)) - 1
        if not early_stopped or is_last_stage:
            break

    return model, logs


def evaluate_model(model, x_test, durations_test, events_test):
    model.compute_baseline_hazards()
    surv = model.predict_surv_df(x_test)
    evaluator = EvalSurv(surv, durations_test, events_test, censor_surv="km")
    return {
        "concordance_td": float(evaluator.concordance_td()),
    }


def run_five_fold_cv(features, durations, events, weights=None, config=None):
    x = np.asarray(features, dtype=np.float32)
    t = np.asarray(durations, dtype=np.float32)
    e = np.asarray(events, dtype=np.float32)
    w = None if weights is None else np.asarray(weights, dtype=np.float32)

    if config is None:
        config = TrainConfig(
            input_dim=x.shape[1],
            use_weighted_loss=w is not None,
        )

    splitter = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    fold_rows = []
    fitted_models = []

    for fold_idx, (train_idx, holdout_idx) in enumerate(splitter.split(x, e), start=1):
        holdout_events = e[holdout_idx]
        split_result = train_test_split(
            holdout_idx,
            test_size=0.5,
            stratify=holdout_events,
            random_state=config.random_state + fold_idx,
        )
        val_idx, test_idx = split_result

        x_train, x_val, x_test = x[train_idx], x[val_idx], x[test_idx]
        t_train, t_val, t_test = t[train_idx], t[val_idx], t[test_idx]
        e_train, e_val, e_test = e[train_idx], e[val_idx], e[test_idx]

        w_train = None if w is None else w[train_idx]
        w_val = None if w is None else w[val_idx]

        model, logs = fit_single_split(
            x_train=x_train,
            durations_train=t_train,
            events_train=e_train,
            x_val=x_val,
            durations_val=t_val,
            events_val=e_val,
            weights_train=w_train,
            weights_val=w_val,
            config=config,
        )

        metrics = evaluate_model(
            model=model,
            x_test=x_test,
            durations_test=t_test,
            events_test=e_test,
        )

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
                "n_events_test": int(e_test.sum()),
                "concordance_td": metrics["concordance_td"],
            }
        )
        fitted_models.append(
            {
                "fold": fold_idx,
                "model": model,
                "logs": logs,
            }
        )

    results = pd.DataFrame(fold_rows)
    summary = {
        "config": asdict(config),
        "mean_concordance_td": float(results["concordance_td"].mean()),
        "std_concordance_td": float(results["concordance_td"].std(ddof=1)) if len(results) > 1 else 0.0,
    }
    return results, summary, fitted_models
