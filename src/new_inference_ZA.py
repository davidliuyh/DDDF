"""
Full-field inference for vector (multi-channel) displacement fields.

Extends inference.py to handle (N, N, N, C) vector fields, transposing
to/from PyTorch's (C, D, H, W) convention.
"""

import time
import numpy as np
import torch
from tqdm import tqdm


def apply_model_to_field(field, model, patch_size, pad, overlap, device=None, batch_size=8):
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

    grid = range(0, N, step)
    total_patches = len(grid) ** 3

    def flush_batch(patch_tensors, patch_meta):
        if not patch_tensors:
            return

        inp = torch.stack(patch_tensors, dim=0).float().to(device, non_blocking=True)
        pred = model(inp).cpu().numpy()

        for i, (z, y, x, cz, cy, cx) in enumerate(patch_meta):
            if is_vector:
                # (B, C, D, H, W) -> (D, H, W, C)
                pred_i = pred[i].transpose(1, 2, 3, 0)
            else:
                # (B, 1, D, H, W) -> (D, H, W)
                pred_i = pred[i, 0]

            crop = pred_i[
                pad:pad + patch_size,
                pad:pad + patch_size,
                pad:pad + patch_size,
            ]

            output[z:z + cz, y:y + cy, x:x + cx] += crop[:cz, :cy, :cx]
            weight[z:z + cz, y:y + cy, x:x + cx] += 1

    with torch.no_grad():
        patch_tensors = []
        patch_meta = []

        with tqdm(total=total_patches, desc='apply_model (patch-batch)') as pbar:
            for z in grid:
                for y in grid:
                    for x in grid:
                        zs = np.arange(z - pad, z + patch_size + pad) % N
                        ys = np.arange(y - pad, y + patch_size + pad) % N
                        xs = np.arange(x - pad, x + patch_size + pad) % N

                        block = field[np.ix_(zs, ys, xs)]

                        if is_vector:
                            # (D, H, W, C) -> (C, D, H, W)
                            inp = torch.from_numpy(block).permute(3, 0, 1, 2)
                        else:
                            # (D, H, W) -> (1, D, H, W)
                            inp = torch.from_numpy(block).unsqueeze(0)

                        z_end = min(z + patch_size, N)
                        y_end = min(y + patch_size, N)
                        x_end = min(x + patch_size, N)
                        cz, cy, cx = z_end - z, y_end - y, x_end - x

                        patch_tensors.append(inp)
                        patch_meta.append((z, y, x, cz, cy, cx))

                        if len(patch_tensors) >= batch_size:
                            flush_batch(patch_tensors, patch_meta)
                            pbar.update(len(patch_tensors))
                            patch_tensors.clear()
                            patch_meta.clear()

            if patch_tensors:
                flush_batch(patch_tensors, patch_meta)
                pbar.update(len(patch_tensors))

    if is_vector:
        w = weight[..., np.newaxis]
        output = np.where(w > 0, output / w, 0.0).astype(field.dtype)
    else:
        output = np.where(weight > 0, output / weight, 0.0).astype(field.dtype)

    elapsed = time.time() - t0
    print(
        f'apply_model done: step={step}, overlap={overlap}, '
        f'batch_size={batch_size}, elapsed={elapsed:.1f}s'
    )
    return output
