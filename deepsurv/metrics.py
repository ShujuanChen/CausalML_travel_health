import torch


def concordance_index(log_risk, durations, events, eps=1e-12):
    risk = log_risk.reshape(-1)
    time = durations.reshape(-1)
    event = events.reshape(-1)

    event_rows = event == 1
    if event_rows.sum() == 0:
        return torch.tensor(0.0, device=risk.device)

    r_i = risk[event_rows].unsqueeze(1)
    t_i = time[event_rows].unsqueeze(1)
    r_j = risk.unsqueeze(0)
    t_j = time.unsqueeze(0)

    comparable = t_i < t_j
    if comparable.sum() == 0:
        return torch.tensor(0.0, device=risk.device)

    concordant = ((r_i > r_j) & comparable).sum(dtype=torch.float32)
    tied = ((r_i == r_j) & comparable).sum(dtype=torch.float32)
    permissible = comparable.sum(dtype=torch.float32)

    return (concordant + 0.5 * tied) / (permissible + eps)
