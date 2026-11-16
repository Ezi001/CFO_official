"""Error metrics used in training and evaluation."""

import numpy as np

# def masked_mse_loss(pred, target, ocean_mask):
#     """
#     pred:       (B, H, W, C)
#     target:     (B, H, W, C)
#     ocean_mask: (H, W), 1=ocean, 0=land
#     """

#     mask = ocean_mask.to(
#         device=pred.device,
#         dtype=pred.dtype
#     )

#     mask = mask[None, :, :, None]

#     squared_error = (pred - target) ** 2

#     loss = (squared_error * mask).sum() / (
#         mask.expand_as(pred).sum()
#     )

#     return loss


# def masked_relative_frobenius_error(
#     A: np.ndarray,
#     B: np.ndarray,
#     ocean_mask: np.ndarray
# ) -> float:
#     """
#     Masked relative Frobenius norm error.

#     A:          (batch, H, W, C) reference
#     B:          (batch, H, W, C) prediction
#     ocean_mask: (H, W), 1=ocean, 0=land
#     """
#     mask = ocean_mask[None, :, :, None]

#     err = np.linalg.norm((A - B) * mask)
#     ref = np.linalg.norm(A * mask)

#     if ref == 0:
#         return float("nan")

#     return float(err / ref)


def relative_frobenius_error(A: np.ndarray, B: np.ndarray) -> float:
    """Relative Frobenius norm error between A and B."""
    err = np.linalg.norm(A - B)
    ref = np.linalg.norm(A)
    return float(err / ref)

# def masked_relative_L2_error(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> np.ndarray:
#     """Mean and max relative RMSE over spatial dims.
#     gt:       (B, H, W, C)
#     pred:     (B, H, W, C)
#     ocean_mask: (H, W), 1=ocean, 0=land
    
#     """
#     axes = tuple(range(1, gt.ndim))
#     err = np.sqrt(np.mean((gt - pred) ** 2, axis=axes))
#     den = np.sqrt(np.mean(gt**2, axis=axes)) + eps
#     rel = err / den
#     return np.mean(rel, axis=0)

def relative_L2_error(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Mean and max relative RMSE over spatial dims."""
    axes = tuple(range(1, gt.ndim))
    err = np.sqrt(np.mean((gt - pred) ** 2, axis=axes))
    den = np.sqrt(np.mean(gt**2, axis=axes)) + eps
    rel = err / den
    return np.mean(rel, axis=0)

# def masked_relative_L2_error_all_steps(
#     gt: np.ndarray,
#     pred: np.ndarray,
#     eps: float = 1e-8,
#     keep_channel: bool = False,
# ) -> np.ndarray:
#     """Per-step mean relative RMSE (optionally per channel)."""
#     if keep_channel:
#         axes = tuple(range(2, gt.ndim - 1))
#     else:
#         axes = tuple(range(2, gt.ndim))
#     err = np.sqrt(np.mean((gt - pred) ** 2, axis=axes))
#     den = np.sqrt(np.mean(gt**2, axis=axes)) + eps
#     return np.mean(err / den, axis=0)

def relative_L2_error_all_steps(
    gt: np.ndarray,
    pred: np.ndarray,
    eps: float = 1e-8,
    keep_channel: bool = False,
) -> np.ndarray:
    """Per-step mean relative RMSE (optionally per channel)."""
    if keep_channel:
        axes = tuple(range(2, gt.ndim - 1))
    else:
        axes = tuple(range(2, gt.ndim))
    err = np.sqrt(np.mean((gt - pred) ** 2, axis=axes))
    den = np.sqrt(np.mean(gt**2, axis=axes)) + eps
    return np.mean(err / den, axis=0)

# def masked_rmse(gt: np.ndarray, pred: np.ndarray) -> float:
#     error_images = gt - pred
#     root_mean_square_error = np.sqrt(np.mean(np.square(error_images)))
#     return float(root_mean_square_error)


def rmse(gt: np.ndarray, pred: np.ndarray) -> float:
    error_images = gt - pred
    root_mean_square_error = np.sqrt(np.mean(np.square(error_images)))
    return float(root_mean_square_error)
