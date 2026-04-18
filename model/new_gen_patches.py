"""
Patch extraction supporting both 3D scalar and 4D vector fields.

Extends gen_patches.py to handle (D, H, W, C) vector fields.
Rotation augmentation is disabled for 4D fields (vector component
transformation under rotation is non-trivial).
"""

import warnings
from tqdm import tqdm
import numpy as np


def extract_patches(field, patch_size=32, padding=0, overlap=0.5, rotate=True):
    """Return a numpy array containing all extracted patches.

    Supports 3D (D, H, W) scalar fields and 4D (D, H, W, C) vector fields.
    For 4D fields, rotation is automatically disabled.

    Parameters
    ----------
    field : np.ndarray
        Input field; 3D or 4D.
    patch_size : int
    padding : int
    overlap : float
    rotate : bool
    """
    assert 0 <= overlap < 1.0
    step = int(patch_size * (1 - overlap))
    if step < 1:
        step = 1

    is_vector = field.ndim == 4
    if is_vector and rotate:
        warnings.warn(
            'Rotation augmentation disabled for 4D vector fields '
            '(component transformation not implemented).',
            stacklevel=2,
        )
        rotate = False

    patches = []

    if field.ndim == 3:
        D, H, W = field.shape
    elif field.ndim == 4:
        D, H, W, C = field.shape
    else:
        raise ValueError(f'Field must be 3D or 4D, got {field.ndim}D.')

    nZ = (D + step - 1) // step
    nY = (H + step - 1) // step
    nX = (W + step - 1) // step
    total = nZ * nY * nX
    pbar = tqdm(total=total, desc='extract_patches', unit='block')

    for z in range(0, D, step):
        for y in range(0, H, step):
            for x in range(0, W, step):
                zs = np.arange(z - padding, z + patch_size + padding)
                ys = np.arange(y - padding, y + patch_size + padding)
                xs = np.arange(x - padding, x + patch_size + padding)

                if is_vector:
                    # (D, H, W, C) → periodic take on spatial axes
                    block = np.take(field, zs, axis=0, mode='wrap')
                    block = np.take(block, ys, axis=1, mode='wrap')
                    block = np.take(block, xs, axis=2, mode='wrap')
                else:
                    block = np.take(field, zs, axis=0, mode='wrap')
                    block = np.take(block, ys, axis=1, mode='wrap')
                    block = np.take(block, xs, axis=2, mode='wrap')

                if rotate and not is_vector:
                    patches.append(block)
                    for axes in ((1, 2), (0, 2), (0, 1)):
                        for k in (1, 2, 3):
                            patches.append(np.rot90(block, k=k, axes=axes))
                else:
                    patches.append(block)
                pbar.update(1)
    pbar.close()

    return np.stack(patches, axis=0)
