import pandas as pd
import numpy as np
import json
import os
import numba
from pycox.evaluation import EvalSurv
from torch import Tensor

def idx_at_times(index_surv, times, steps='pre', assert_sorted=True):
    """Gives index of `index_surv` corresponding to `time`, i.e. 
    `index_surv[idx_at_times(index_surv, times)]` give the values of `index_surv`
    closest to `times`.
    """
    if assert_sorted:
        assert pd.Series(index_surv).is_monotonic_increasing, "Need 'index_surv' to be monotonic increasing"
    if steps == 'pre':
        idx = np.searchsorted(index_surv, times)
    elif steps == 'post':
        idx = np.searchsorted(index_surv, times, side='right') - 1
    return idx.clip(0, len(index_surv)-1)

def _is_comparable(t_i, t_j, d_i, d_j):
    return ((t_i < t_j) & d_i) | ((t_i == t_j) & (d_i | d_j))

def _is_comparable_antolini(t_i, t_j, d_i, d_j):
    return ((t_i < t_j) & d_i) | ((t_i == t_j) & d_i & (d_j == 0))

def _is_concordant(s_i, s_j, t_i, t_j, d_i, d_j):
    conc = 0.
    if t_i < t_j:
        conc = (s_i < s_j) + (s_i == s_j) * 0.5
    elif t_i == t_j: 
        if d_i & d_j:
            conc = 1. - (s_i != s_j) * 0.5
        elif d_i:
            conc = (s_i < s_j) + (s_i == s_j) * 0.5  # different from RSF paper.
        elif d_j:
            conc = (s_i > s_j) + (s_i == s_j) * 0.5  # different from RSF paper.
    return conc * _is_comparable(t_i, t_j, d_i, d_j)

def _is_concordant_antolini(s_i, s_j, t_i, t_j, d_i, d_j):
    return (s_i < s_j) & _is_comparable_antolini(t_i, t_j, d_i, d_j)

def _sum_comparable(t, d, is_comparable_func):
    n = t.shape[0]
    count = 0.
    for i in numba.prange(n):
        for j in range(n):
            if j != i:
                count += is_comparable_func(t[i], t[j], d[i], d[j])
    return count

def _sum_concordant(s, t, d):
    n = len(t)
    count = 0.
    for i in numba.prange(n):
        for j in range(n):
            if j != i:
                count += _is_concordant(s[i, i], s[i, j], t[i], t[j], d[i], d[j])
    return count

def _sum_concordant_disc(s, t, d, s_idx, is_concordant_func):
    n = len(t)
    count = 0
    for i in numba.prange(n):
        idx = s_idx[i]
        for j in range(n):
            if j != i:
                count += is_concordant_func(s[idx, i], s[idx, j], t[i], t[j], d[i], d[j])
    return count

def concordance_td(durations, events, surv, surv_idx, method='adj_antolini'):
    """Time dependent concorance index from
    Antolini, L.; Boracchi, P.; and Biganzoli, E. 2005. A timedependent discrimination
    index for survival data. Statistics in Medicine 24:3927–3944.
    """
    if np.isfortran(surv):
        surv = np.array(surv, order='C')
    assert durations.shape[0] == surv.shape[1] == surv_idx.shape[0] == events.shape[0]
    assert type(durations) is type(events) is type(surv) is type(surv_idx) is np.ndarray
    if events.dtype in ('float', 'float32'):
        events = events.astype('int32')
    if method == 'adj_antolini':
        is_concordant = _is_concordant
        is_comparable = _is_comparable

        a1 = _sum_concordant_disc(surv, durations, events, surv_idx, is_concordant)
        a2 = _sum_comparable(durations, events, is_comparable)
        print(f"Sum of concordant pairs {a1}")
        print(f"Sum of comparable pairs {a2}")
        return (((a1)/(a2))*(100))
    elif method == 'antolini':
        is_concordant = _is_concordant_antolini
        is_comparable = _is_comparable_antolini
        return (((_sum_concordant_disc(surv, durations, events, surv_idx, is_concordant)) /
                _sum_comparable(durations, events, is_comparable))*(100))
    return ValueError(f"Need 'method' to be e.g. 'antolini', got '{method}'.")