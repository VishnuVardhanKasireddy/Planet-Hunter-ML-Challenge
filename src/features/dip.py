
import numpy as np
import pandas as pd


def dip_features(flux, batch=2000):
    """
    Extract four downward-dip features per light curve.

    Features:
      - dip_fraction: fraction of points substantially below baseline
      - deepest_dip: deepest downward deviation in robust MAD units
      - dip_asymmetry: downward vs upward excursion magnitude
      - dip_concentration: fraction of total downward excursion
        contributed by the strongest 10% of dip points
    """
    n_stars = flux.shape[0]
    results = []

    for start in range(0, n_stars, batch):
        block = np.asarray(
            flux[start:start + batch], dtype=np.float64
        )

        # Robustly normalize each star by its median flux.
        baseline = np.nanmedian(block, axis=1, keepdims=True)
        baseline[np.abs(baseline) < 1e-12] = np.nan
        normalized = block / baseline - 1.0

        center = np.nanmedian(normalized, axis=1, keepdims=True)
        deviation = normalized - center

        mad = np.nanmedian(np.abs(deviation), axis=1, keepdims=True)
        scale = 1.4826 * mad

        # Avoid division by zero for nearly constant curves.
        scale[~np.isfinite(scale) | (scale < 1e-8)] = np.nan

        z = deviation / scale

        dip_fraction = np.nanmean(z < -3.0, axis=1)
        deepest_dip = np.nanmax(-z, axis=1)

        down = np.clip(-z, 0, None)
        up = np.clip(z, 0, None)

        down_mean = np.nanmean(down, axis=1)
        up_mean = np.nanmean(up, axis=1)

        dip_asymmetry = down_mean / (up_mean + 1e-8)

        # How much of the downward excursion is concentrated
        # in the strongest 10% of valid cadence positions?
        concentration = np.full(len(block), np.nan)

        for i in range(len(block)):
            values = down[i]
            values = values[np.isfinite(values)]

            if len(values) and np.sum(values) > 0:
                k = max(1, int(np.ceil(0.10 * len(values))))
                largest = np.partition(values, len(values) - k)[-k:]
                concentration[i] = largest.sum() / values.sum()
            elif len(values):
                concentration[i] = 0.0

        features = np.column_stack([
            dip_fraction,
            deepest_dip,
            dip_asymmetry,
            concentration,
        ])

        results.append(features)

    values = np.vstack(results)

    # Keep the feature matrix finite for model training.
    values[~np.isfinite(values)] = 0.0

    return pd.DataFrame(
        values,
        columns=[
            "dip_fraction",
            "deepest_dip",
            "dip_asymmetry",
            "dip_concentration",
        ],
    )
