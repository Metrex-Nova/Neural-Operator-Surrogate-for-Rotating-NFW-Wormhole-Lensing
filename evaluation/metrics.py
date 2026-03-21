import numpy as np
import torch


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def mse(pred, target):
    p, t = _to_np(pred), _to_np(target)
    return float(np.mean((p - t) ** 2))


def mae(pred, target):
    p, t = _to_np(pred), _to_np(target)
    return float(np.mean(np.abs(p - t)))


def relative_l2(pred, target, eps=1e-8):
    p, t = _to_np(pred), _to_np(target)
    if p.ndim == 1:
        return float(np.linalg.norm(p - t) / (np.linalg.norm(t) + eps))
    return float(np.mean(np.linalg.norm(p - t, axis=-1) /
                         (np.linalg.norm(t, axis=-1) + eps)))


def peak_position_error(pred, target, b_grid):
    p, t, bg = _to_np(pred), _to_np(target), _to_np(b_grid)
    if p.ndim == 1:
        return float(abs(bg[np.argmax(p)] - bg[np.argmax(t)]))
    return float(np.mean([abs(bg[np.argmax(p[i])] - bg[np.argmax(t[i])])
                          for i in range(len(p))]))


def evaluate_model(model, loader, device, model_type="fno"):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for fno_input, params, profile in loader:
            profile = profile.to(device)
            pred    = model(fno_input.to(device)) if model_type == "fno" \
                      else model(params.to(device))
            all_pred.append(pred.cpu())
            all_true.append(profile.cpu())

    preds  = torch.cat(all_pred).numpy()
    truths = torch.cat(all_true).numpy()
    return {
        "mse":       mse(preds, truths),
        "mae":       mae(preds, truths),
        "rel_l2":    relative_l2(preds, truths),
        "n_samples": len(preds),
    }
