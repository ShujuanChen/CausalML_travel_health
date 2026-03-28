import torch
from torch import nn


def negative_cox_partial_log_likelihood(log_risk, durations, events, eps=1e-12):
  
    return weighted_negative_cox_partial_log_likelihood(
        log_risk=log_risk,
        durations=durations,
        events=events,
        weights=None,
        eps=eps,
    )


def weighted_negative_cox_partial_log_likelihood(
    log_risk,
    durations,
    events,
    weights=None,
    eps=1e-12,
):

    eta = log_risk.reshape(-1)
    time = durations.reshape(-1)
    event = events.reshape(-1).float()

    if weights is None:
        sample_weight = torch.ones_like(event, dtype=torch.float32, device=event.device)
    else:
        sample_weight = weights.reshape(-1).float().to(event.device)

    order = torch.argsort(time, descending=True)
    eta = eta[order]
    event = event[order]
    sample_weight = sample_weight[order]

    log_risk_set = torch.logcumsumexp(eta, dim=0)
    weighted_event_contrib = sample_weight * event * (eta - log_risk_set)

    normalizer = (sample_weight * event).sum()
    if normalizer <= eps:
        return torch.zeros((), dtype=eta.dtype, device=eta.device)

    return -weighted_event_contrib.sum() / normalizer


class WeightedCoxPHLoss(nn.Module):
    def forward(self, log_risk, durations, events, weights):
        return weighted_negative_cox_partial_log_likelihood(
            log_risk=log_risk,
            durations=durations,
            events=events,
            weights=weights,
        )
