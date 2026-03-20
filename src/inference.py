"""
Full-field inference: tile a periodic 3D field with a trained UNet patch-by-patch
and average the overlapping predictions.

The striding and periodic-boundary treatment match gen_patches.extract_patches
exactly, so the model sees the same "view" of the data at inference as at training.
"""

import time
import numpy as np
import torch
from tqdm import tqdm


def apply_unet_to_field(field, model, patch_size, pad, overlap, device=None):
    """Apply a trained UNet to every patch of a periodic 3D field.

    Patches are extracted with periodic boundary wrapping; overlapping
    regions are averaged in the output.

    Parameters
    ----------
    field : np.ndarray, shape (N, N, N)
        Input field with periodic boundary conditions.
    model : torch.nn.Module
        Trained model (will be set to eval mode inside).
    patch_size : int
        Spatial extent of each prediction patch (excluding padding).
    pad : int
        Number of context cells added on each side (periodic wrap).
    overlap : float in [0, 1)
        Fraction of patch_size that consecutive patches overlap.
        Must match the value used in extract_patches at training time.
    device : torch.device or None
        Inference device; defaults to CUDA if available.

    Returns
    -------
    np.ndarray
        Reconstructed field, same shape and dtype as ``field``.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N    = field.shape[0]
    step = max(1, int(patch_size * (1 - overlap)))

    output = np.zeros_like(field, dtype=np.float64)
    weight = np.zeros_like(field, dtype=np.float64)

    model.eval()
    t0 = time.time()

    z_indices = range(0, N, step)
    with torch.no_grad():
        for z in tqdm(z_indices, desc='apply_unet (z-slices)'):
            for y in range(0, N, step):
                for x in range(0, N, step):
                    # Periodic-boundary patch extraction (same as extract_patches)
                    zs = np.arange(z - pad, z + patch_size + pad) % N
                    ys = np.arange(y - pad, y + patch_size + pad) % N
                    xs = np.arange(x - pad, x + patch_size + pad) % N
                    block = field[np.ix_(zs, ys, xs)]

                    inp  = torch.from_numpy(block).unsqueeze(0).unsqueeze(0).float().to(device)
                    pred = model(inp).cpu().numpy().squeeze()
                    crop = pred[pad:pad + patch_size,
                                pad:pad + patch_size,
                                pad:pad + patch_size]

                    # Write back (handle grid boundary gracefully)
                    z_end, y_end, x_end = min(z + patch_size, N), min(y + patch_size, N), min(x + patch_size, N)
                    cz, cy, cx = z_end - z, y_end - y, x_end - x
                    output[z:z_end, y:y_end, x:x_end] += crop[:cz, :cy, :cx]
                    weight[z:z_end, y:y_end, x:x_end] += 1

    output = np.where(weight > 0, output / weight, 0.0).astype(field.dtype)
    elapsed = time.time() - t0
    print(f'apply_unet done: step={step}, elapsed={elapsed:.1f}s')
    return output
