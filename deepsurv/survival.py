from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class BreslowEstimate:
    times: np.ndarray
    cumulative_hazard: np.ndarray
    survival: np.ndarray


def predict_log_risk(model, features, device=None):
    if not torch.is_tensor(features):
        features = torch.as_tensor(features, dtype=torch.float32)

    if device is None:
        device = next(model.parameters()).device

    return model(features.to(device)).detach().cpu().numpy().reshape(-1)


def breslow_baseline_survival(model, features, durations, events, device=None):

    log_risk = predict_log_risk(model, features, device=device)
    risk = np.exp(log_risk)

    durations = np.asarray(durations, dtype=np.float64)
    events = np.asarray(events).astype(bool)

    event_times = np.unique(durations[events])
    event_times.sort()

    cumulative_hazard = np.zeros_like(event_times, dtype=np.float64)
    running_total = 0.0

    for idx, event_time in enumerate(event_times):
        d_j = np.sum((durations == event_time) & events)
        denom = risk[durations >= event_time].sum()
        if denom > 0:
            running_total += d_j / denom
        cumulative_hazard[idx] = running_total

    survival = np.exp(-cumulative_hazard)
    return BreslowEstimate(
        times=event_times.astype(np.float32),
        cumulative_hazard=cumulative_hazard.astype(np.float32),
        survival=survival.astype(np.float32),
    )


def individual_survival_curve(log_risk, baseline):

    log_risk = np.asarray(log_risk, dtype=np.float64).reshape(-1, 1)
    cumulative_hazard = np.asarray(baseline.cumulative_hazard, dtype=np.float64).reshape(1, -1)
    survival = np.exp(-np.exp(log_risk) * cumulative_hazard)
    return survival.astype(np.float32)
