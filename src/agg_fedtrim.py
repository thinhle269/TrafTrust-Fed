# -*- coding: utf-8 -*-
import numpy as np

def coord_trimmed_mean(stacked_params, trim=0.1):
    """Perform coordinate-wise trimmed mean aggregation."""
    out = []
    p = float(np.clip(trim, 0.0, 0.49))
    lo_q = p * 100
    hi_q = (1.0 - p) * 100
    for arr in stacked_params:
        lo = np.percentile(arr, lo_q, axis=0, keepdims=True)
        hi = np.percentile(arr, hi_q, axis=0, keepdims=True)
        mask = (arr >= lo) & (arr <= hi)
        denom = np.maximum(mask.sum(axis=0, keepdims=True), 1)
        s = (arr * mask).sum(axis=0, keepdims=True) / denom
        out.append(np.squeeze(s, axis=0))
    return out

def aggregate_fedtrim(local_weights, sample_counts, trim=0.1):
    """Aggregate client models using coordinate-wise trimmed mean."""
    new_weights = []
    for i in range(len(local_weights[0])):
        stacked = np.stack([lw[i] for lw in local_weights], axis=0)
        arrs = [stacked]  # for coord_trimmed_mean
        trimmed = coord_trimmed_mean(arrs, trim=trim)
        new_weights.append(trimmed[0])
    return new_weights
