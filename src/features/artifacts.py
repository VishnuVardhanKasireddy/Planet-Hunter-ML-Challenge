
import numpy as np
import pandas as pd


def extract_artifact_features(flux, batch=2000):
    """Extract robust sharp-outlier and jump features."""
    flux = np.asarray(flux, dtype=np.float32)
    n_stars = flux.shape[0]

    output = np.empty((n_stars, 3), dtype=np.float32)

    for start in range(0, n_stars, batch):
        end = min(start + batch, n_stars)
        values = flux[start:end]

        median = np.nanmedian(values, axis=1, keepdims=True)
        mad = np.nanmedian(
            np.abs(values - median), axis=1, keepdims=True
        )

        # Avoid division by zero for nearly constant curves.
        scale = np.maximum(1.4826 * mad, 1e-8)
        robust_z = np.abs((values - median) / scale)

        # Ignore missing samples in the outlier fraction.
        valid = np.isfinite(values)
        outlier = (robust_z > 8) & valid
        outlier_count = outlier.sum(axis=1)
        valid_count = valid.sum(axis=1)

        outlier_fraction = np.divide(
            outlier_count,
            valid_count,
            out=np.zeros(len(values), dtype=np.float32),
            where=valid_count > 0,
        )

        # Adjacent flux jumps, ignoring pairs with missing samples.
        differences = np.abs(np.diff(values, axis=1))
        valid_pairs = (
            np.isfinite(values[:, :-1])
            & np.isfinite(values[:, 1:])
        )
        differences[~valid_pairs] = np.nan

        max_jump = np.nanmax(differences, axis=1)
        max_jump = np.nan_to_num(
            max_jump, nan=0.0, posinf=0.0, neginf=0.0
        )

        diff_median = np.nanmedian(differences, axis=1)
        diff_mad = np.nanmedian(
            np.abs(differences - diff_median[:, None]),
            axis=1,
        )
        jump_threshold = (
            diff_median + 8 * 1.4826 * diff_mad
        )

        large_jump_fraction = np.nanmean(
            (differences > jump_threshold[:, None])
            & valid_pairs,
            axis=1,
        )
        large_jump_fraction = np.nan_to_num(
            large_jump_fraction, nan=0.0
        )

        output[start:end] = np.column_stack([
            outlier_fraction,
            max_jump,
            large_jump_fraction,
        ])

    return pd.DataFrame(
        output,
        columns=[
            "extreme_outlier_fraction",
            "max_consecutive_jump",
            "large_jump_fraction",
        ],
    )
