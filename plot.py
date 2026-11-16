import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from utils.data import select_data_split
from utils.readers import load_dataset_splits

dataset = "nordic"  # dr, swe, nordic, burgers, or lorenz
split = "test"

target = select_data_split(load_dataset_splits(dataset), split)
pred = np.load("pred.npy")
error = pred - target

# One RMSE value per time step, averaged over samples and all state dimensions.
axes = (0, *range(2, error.ndim))
rmse_time = np.sqrt(np.mean(error**2, axis=axes))
plt.figure()
plt.plot(np.arange(len(rmse_time)), rmse_time)
plt.xlabel("Time step")
plt.ylabel("RMSE")
plt.title(f"{dataset}: RMSE over time")
plt.tight_layout()
plt.savefig(f"{dataset}_rmse_over_time.png", dpi=200, bbox_inches="tight")
plt.close()

if dataset == "burgers":
    # RMSE at each spatial point, averaged over samples and time.
    rmse_space = np.sqrt(np.mean(error**2, axis=(0, 1)))
    plt.figure()
    plt.plot(rmse_space)
    plt.xlabel("Spatial grid point")
    plt.ylabel("RMSE")
    plt.title(f"{dataset}: RMSE over space")
    plt.tight_layout()
    plt.savefig(f"{dataset}_rmse_over_space.png", dpi=200, bbox_inches="tight")
    plt.close()

elif dataset in {"dr", "swe", "nordic"}:
    # Select channel 0, then compute RMSE at each grid cell.
    rmse_space = np.sqrt(np.mean(error[..., 0]**2, axis=(0, 1)))
    plt.figure()
    plt.imshow(rmse_space, origin="lower", aspect="auto")
    plt.colorbar(label="RMSE")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"{dataset}: spatial RMSE (channel 0)")
    plt.tight_layout()
    plt.savefig(f"{dataset}_rmse_spatial_map.png", dpi=200, bbox_inches="tight")
    plt.close()

elif dataset == "lorenz":
    # Lorenz has state variables, not a spatial grid: plot RMSE per variable.
    rmse_state = np.sqrt(np.mean(error**2, axis=0))  # (T, 3)
    plt.figure()
    for i in range(rmse_state.shape[-1]):
        plt.plot(rmse_state[:, i], label=f"state {i}")
    plt.xlabel("Time step")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Lorenz: RMSE by state variable")
    plt.tight_layout()
    plt.savefig(f"{dataset}_rmse_by_state.png", dpi=200, bbox_inches="tight")
    plt.close()