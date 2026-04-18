"""
Full-field inference for vector (multi-channel) displacement fields.

Extends inference.py to handle (N, N, N, C) vector fields, transposing
to/from PyTorch's (C, D, H, W) convention.
"""

import time
import numpy as np
import torch
from tqdm import tqdm


def apply_model_to_field(field, model, patch_size, pad, overlap, device=None):
    """Apply a trained model to every patch of a periodic 3D or 4D field.

    Parameters
    ----------
    field : np.ndarray
        Input field: (N, N, N) scalar or (N, N, N, C) vector.
    model : torch.nn.Module
    patch_size, pad : int
    overlap : float
    device : torch.device or None

    Returns
    -------
    np.ndarray – same shape and dtype as ``field``.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_vector = field.ndim == 4
    N = field.shape[0]
    step = max(1, int(patch_size * (1 - overlap)))

    if is_vector:
        C = field.shape[-1]
        output = np.zeros((N, N, N, C), dtype=np.float64)
        weight = np.zeros((N, N, N), dtype=np.float64)
    else:
        output = np.zeros((N, N, N), dtype=np.float64)
        weight = np.zeros((N, N, N), dtype=np.float64)

    model.eval()
    t0 = time.time()

    with torch.no_grad():
        for z in tqdm(range(0, N, step), desc='apply_model (z-slices)'):
            for y in range(0, N, step):
                for x in range(0, N, step):
                    zs = np.arange(z - pad, z + patch_size + pad) % N
                    ys = np.arange(y - pad, y + patch_size + pad) % N
                    xs = np.arange(x - pad, x + patch_size + pad) % N

                    block = field[np.ix_(zs, ys, xs)] if not is_vector else field[np.ix_(zs, ys, xs)]

                    if is_vector:
                        # (ps+2*pad, ps+2*pad, ps+2*pad, C) → (1, C, D, H, W)
                        inp = torch.from_numpy(block).permute(3, 0, 1, 2).unsqueeze(0).float().to(device)
                    else:
                        # (ps+2*pad,)^3 → (1, 1, D, H, W)
                        inp = torch.from_numpy(block).unsqueeze(0).unsqueeze(0).float().to(device)

                    pred = model(inp).cpu().numpy()

                    if is_vector:
                        # (1, C, D, H, W) → (D, H, W, C)
                        pred = pred[0].transpose(1, 2, 3, 0)
                    else:
                        pred = pred.squeeze()

                    crop = pred[pad:pad + patch_size,
                                pad:pad + patch_size,
                                pad:pad + patch_size]

                    z_end = min(z + patch_size, N)
                    y_end = min(y + patch_size, N)
                    x_end = min(x + patch_size, N)
                    cz, cy, cx = z_end - z, y_end - y, x_end - x

                    if is_vector:
                        output[z:z_end, y:y_end, x:x_end] += crop[:cz, :cy, :cx]
                    else:
                        output[z:z_end, y:y_end, x:x_end] += crop[:cz, :cy, :cx]
                    weight[z:z_end, y:y_end, x:x_end] += 1

    if is_vector:
        w = weight[..., np.newaxis]
        output = np.where(w > 0, output / w, 0.0).astype(field.dtype)
    else:
        output = np.where(weight > 0, output / weight, 0.0).astype(field.dtype)

    elapsed = time.time() - t0
    print(f'apply_model done: step={step}, elapsed={elapsed:.1f}s')
    return output
