# data/utils/voxels.py
import numpy as np

def events_to_voxel(ts, xs, ys, ps,
                    num_bins: int, H: int, W: int):
    """
    ts, xs, ys, ps can be Python lists or 1D numpy arrays.
    Returns a float32 voxel grid: shape (T, 2, H, W).
    """
    ts  = np.asarray(ts, dtype=np.int64)
    xs  = np.asarray(xs, dtype=np.int64)
    ys  = np.asarray(ys, dtype=np.int64)
    ps  = np.asarray(ps, dtype=np.int64)

    if ts.size == 0:
        # no events → return zeros
        return np.zeros((num_bins, 2, H, W), dtype=np.float32)

    t0, t1 = ts.min(), ts.max()
    bins    = np.linspace(t0, t1, num_bins + 1)
    voxel   = np.zeros((num_bins, 2, H, W), dtype=np.float32)

    # digitize: which temporal bin each ts belongs to
    bin_ids = np.digitize(ts, bins) - 1
    valid   = (0 <= bin_ids) & (bin_ids < num_bins)

    for b, x, y, p in zip(bin_ids[valid], xs[valid], ys[valid], ps[valid]):
        c = 0 if p == 1 else 1
        voxel[b, c, y, x] += 1.0
    return voxel
