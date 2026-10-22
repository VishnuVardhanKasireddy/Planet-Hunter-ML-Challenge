
import numpy as np
import pandas as pd


def extract_periodic_features(flux, lags=(5, 10, 20, 50, 100)):
    """
    Compute autocorrelation at selected cadence lags
    for each normalized flux curve.
    """
    flux = np.asarray(flux, dtype=np.float32)
    median = np.nanmedian(flux, axis=1, keepdims=True)

    normalized = flux / np.where(
        np.abs(median) < 1e-8, np.nan, median
    )

    normalized = normalized - np.nanmedian(
        normalized, axis=1, keepdims=True
    )

    features = {}

    for lag in lags:
        left = normalized[:, :-lag]
        right = normalized[:, lag:]

        left_centered = left - np.nanmean(
            left, axis=1, keepdims=True
        )
        right_centered = right - np.nanmean(
            right, axis=1, keepdims=True
        )

        numerator = np.nansum(
            left_centered * right_centered, axis=1
        )
        denominator = np.sqrt(
            np.nansum(left_centered**2, axis=1)
            * np.nansum(right_centered**2, axis=1)
        )

        features[f"autocorr_lag_{lag}"] = np.divide(
            numerator,
            denominator,
            out=np.full(len(flux), np.nan),
            where=denominator > 1e-12,
        )

    return pd.DataFrame(features)
