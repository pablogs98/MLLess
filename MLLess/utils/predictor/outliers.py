# Authors: Marc Sanchez Artigas <marc.sanchez@urv.cat>
# License: BSD 3 clause
import numpy as np
from scipy.stats import iqr as IQR
from collections.abc import Iterable
from pandas import Series

'''
IQR based method
This method will test if the value is less than q1 - 1.5 * iqr or
greater than q3 + 1.5 * iqr.
'''
def Q1(x, axis = None):
    return np.percentile(x, 25, axis = axis)

def Q3(x, axis = None):
    return np.percentile(x, 75, axis = axis)

def IQR_outlier(x, axis = 0, bar = 1.5, side = 'both'):
    assert side in ['gt', 'lt', 'both'], 'Side should be `gt`, `lt` or `both`.'

    d_IQR = IQR(x, axis = axis)
    d_Q1 = Q1(x, axis = axis)
    d_Q3 = Q3(x, axis = axis)
    IQR_distance = np.multiply(d_IQR, bar)

    stat_shape = list(x.shape)

    if isinstance(axis, Iterable):
        for single_axis in axis:
            stat_shape[single_axis] = 1
    else:
        stat_shape[axis] = 1

    if side in ['gt', 'both']:
        upper_range = d_Q3 + IQR_distance
        upper_outlier = np.greater(x - upper_range.reshape(stat_shape), 0)
    if side in ['lt', 'both']:
        lower_range = d_Q1 - IQR_distance
        lower_outlier = np.less(x - lower_range.reshape(stat_shape), 0)

    if side == 'gt':
        return upper_outlier
    if side == 'lt':
        return lower_outlier
    if side == 'both':
        return np.logical_or(upper_outlier, lower_outlier)

'''
 Rolling median based method
 This method will test if the relative error of the rolling median is
 greater than a threshold.
 '''
def rm_outlier(y, threshold=0.1, window=10):
     y_ = Series(y).rolling(window).median().fillna(method='bfill').fillna(method='ffill')
     difference = np.abs(y - y_)
     outlier_idx = difference/y > threshold
     return outlier_idx
