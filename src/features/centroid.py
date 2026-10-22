
import numpy as np
import pandas as pd


def extract_centroid_features(centroid, batch=2000):
    """
    Extract positional variability features from centroid data.

    Parameters
    ----------
    centroid : np.ndarray
        Shape: (n_stars, 2, n_cadences).
        Axis 1 contains the two centroid coordinates.
    batch : int
        Number of stars processed per batch.

    Returns
    -------
    pd.DataFrame
        Eight centroid features per star.
    """
    if centroid.ndim != 3 or centroid.shape[1] != 2:
        raise ValueError(
            "Expected centroid shape (n_stars, 2, n_cadences)"
        )

    feature_names = [
        "centroid_x_std",
        "centroid_x_mad",
        "centroid_x_diff_median",
        "centroid_x_dev_p95",
        "centroid_y_std",
        "centroid_y_mad",
        "centroid_y_diff_median",
        "centroid_y_dev_p95",
    ]

    output = np.empty(
        (centroid.shape[0], len(feature_names)),
        dtype=np.float32,
    )

    for start in range(0, centroid.shape[0], batch):
        end = min(start + batch, centroid.shape[0])

        # Shape: (batch_stars, 2, cadences)
        chunk = centroid[start:end].astype(np.float32)

        for axis, axis_name in enumerate(["x", "y"]):
            values = chunk[:, axis, :]

            with np.errstate(invalid="ignore"):
                median = np.nanmedian(
                    values, axis=1, keepdims=True
                )

                deviation = np.abs(values - median)

                std = np.nanstd(values, axis=1)
                mad = np.nanmedian(deviation, axis=1)
                dev_p95 = np.nanpercentile(
                    deviation, 95, axis=1
                )

                differences = np.abs(np.diff(values, axis=1))
                diff_median = np.nanmedian(
                    differences, axis=1
                )

            offset = axis * 4

            output[start:end, offset] = std
            output[start:end, offset + 1] = mad
            output[start:end, offset + 2] = diff_median
            output[start:end, offset + 3] = dev_p95

    return pd.DataFrame(output, columns=feature_names)
