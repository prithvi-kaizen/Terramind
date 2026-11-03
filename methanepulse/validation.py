"""Strict mask regression with a bounded cross-CPU decision-boundary allowance."""
import numpy as np


def check_reference_mask(actual, reference, probability=None, threshold=0.5,
                         max_pixels=8, probability_tolerance=1e-4):
    if actual.shape != reference.shape:
        raise AssertionError('Reference mask shape differs')
    different=actual != reference
    count=int(different.sum())
    if count==0:
        return {'exact':True,'different_pixels':0}
    if probability is None or probability.shape!=actual.shape or not np.isfinite(probability).all():
        raise AssertionError(f'{count} mask differences require probability diagnostics')
    distances=np.abs(probability[different]-threshold)
    diagnostic={'exact':False,'different_pixels':count,
                'max_threshold_distance':float(distances.max()),
                'max_allowed_pixels':max_pixels,'probability_tolerance':probability_tolerance}
    if count>max_pixels or np.any(distances>probability_tolerance):
        raise AssertionError(f'Mask regression exceeds numerical allowance: {diagnostic}')
    return diagnostic
