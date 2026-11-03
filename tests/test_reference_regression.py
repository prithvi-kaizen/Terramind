import numpy as np
import pytest
from methanepulse.validation import check_reference_mask


def test_reference_matches_exactly_without_probability():
    mask=np.zeros((4,4),dtype='uint8')
    assert check_reference_mask(mask,mask)['exact']


def test_small_near_threshold_difference_is_reported():
    actual=np.zeros((4,4),dtype='uint8');reference=actual.copy();reference[0,0]=1
    probability=np.full((4,4),.1);probability[0,0]=.49999
    result=check_reference_mask(actual,reference,probability)
    assert not result['exact'] and result['different_pixels']==1


def test_few_confident_differences_fail():
    actual=np.zeros((4,4),dtype='uint8');reference=actual.copy();reference[0,0]=1
    with pytest.raises(AssertionError,match='exceeds'):
        check_reference_mask(actual,reference,np.full((4,4),.1))


def test_many_boundary_differences_fail():
    with pytest.raises(AssertionError,match='exceeds'):
        check_reference_mask(np.zeros((4,4)),np.ones((4,4)),np.full((4,4),.49999))
