from tqdm import tqdm
import numpy as np

# Generate training patches.
# Extract fixed-size blocks from target_delta with 50% overlap and optional
# 90-degree rotation augmentation.
# No randomness is used; step size is computed by patch_size * (1 - overlap).

def extract_patches(field, patch_size=32, padding=0, overlap=0.5, rotate=True):
    """Return a numpy array containing all extracted patches.

    Edge patches are handled with periodic padding via
    `np.take(..., mode='wrap')`.

    Parameters
    ----------
    field : np.ndarray
        Input field; supports 2D or 3D.
    patch_size : int
        Side length of each patch (square/cubic assumption).
    padding : int
        Padding size used to add extra context at patch borders.
    overlap : float
        Overlap ratio, where 0 <= overlap < 1.0. 50% overlap is 0.5.
    rotate : bool
        Whether to apply 90-degree rotation augmentation per patch.
        If True, return the original patch plus 90/180/270-degree rotations
        around the x/y/z principal axes.

    Notes
    -----
    The generation process is deterministic: traversal order is fixed and
    no RNG is used.
    Periodic boundaries are handled by `np.take`, so out-of-range indices
    automatically wrap around.
    """
    assert 0 <= overlap < 1.0
    step = int(patch_size * (1 - overlap))
    if step < 1:
        step = 1

    patches = []

    if field.ndim == 3:
        D, H, W = field.shape
        # Precompute total blocks for progress bar display.
        nZ = (D + step - 1) // step
        nY = (H + step - 1) // step
        nX = (W + step - 1) // step
        total = nZ * nY * nX
        pbar = tqdm(total=total, desc='extract_patches', unit='block')

        for z in range(0, D, step):
            for y in range(0, H, step):
                for x in range(0, W, step):
                    # Use np.take to handle periodic boundaries.
                    zs = np.arange(z - padding, z + patch_size + padding)
                    ys = np.arange(y - padding, y + patch_size + padding)
                    xs = np.arange(x - padding, x + patch_size + padding)
                    block = np.take(field, zs, axis=0, mode='wrap')
                    block = np.take(block, ys, axis=1, mode='wrap')
                    block = np.take(block, xs, axis=2, mode='wrap')
                    if rotate:
                        # 3D axis order is (z, y, x).
                        # Keep original block and add 90/180/270-degree
                        # rotations around x/y/z principal axes.
                        patches.append(block)
                        for axes in ((1, 2), (0, 2), (0, 1)):
                            for k in (1, 2, 3):
                                patches.append(np.rot90(block, k=k, axes=axes))
                    else:
                        patches.append(block)
                    pbar.update(1)
        pbar.close()
    else:
        raise ValueError("Field must be 3D.")

    return np.stack(patches, axis=0)