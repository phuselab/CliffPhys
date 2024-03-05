#================================================================================================================================
#==            'CliffPhys: Camera-based Respiratory Measurement using Clifford Neural Networks' (Paper ID #11393)              ==
#================================================================================================================================

"""
Code containing error metrics functions.

ERRORS:
    This script provides the following functions:
    - bpm_diff: Computes the difference between RPM estimates and RPM gt.
    - RMSEerror: Computes the Root Mean Square Error (RMSE).
    - MAEerror: Computes the Mean Absolute Error (MAE).
    - MAPEerror: Computes the Mean Absolute Percentage Error (MAPE).
    - PearsonCorr: Computes the Pearson Correlation Coefficient (PCC).
    - LinCorr: Computes the Lin's Concordance Correlation Coefficient (CCC).
    - concordance_correlation_coefficient: Utility function to compute the Concordance Correlation Coefficient (CCC).
"""

import numpy as np

def bpm_diff(bpmES, bpmGT, timesES=None, timesGT=None, normalize=False):
    """ Computes the difference between RPM estimates and RPM gt"""
    n, m = bpmES.shape  # n = num channels, m = bpm length

    if (timesES is None) or (timesGT is None):
        timesES = np.arange(m)
        timesGT = timesES

    diff = np.zeros((n, m))
    for j in range(m):
        t = timesES[j]
        i = np.argmin(np.abs(t-timesGT))
        for c in range(n):
            if not normalize:
                diff[c, j] = bpmGT[i]-bpmES[c, j]
            else:
                diff[c, j] = (bpmGT[i]-bpmES[c, j]) / bpmGT[i]
    return diff

def RMSEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes RMSE """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.zeros(n)
    for j in range(m):
        for c in range(n):
            df[c] += np.power(diff[c, j], 2)

    # -- final RMSE
    RMSE = round(float(np.sqrt(df/m)),2)
    return RMSE

def MAEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes MAE """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.sum(np.abs(diff), axis=1)

    # -- final MAE
    MAE = round(float(df/m),2)
    return MAE

def MAPEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes MAPE """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT, normalize=True)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.sum(np.abs(diff), axis=1)

    # -- final MAE
    MAPE = round(float((df/m) * 100),2)
    return MAPE

def PearsonCorr(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes PCC """
    from scipy import stats

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length

    if m < 2:
        print('> Warning: Correlation cannot be calculated for signals with len < 2. Returning NaN')
        return np.nan

    CC = np.zeros(n)
    for c in range(n):
        # -- corr
        r, p = stats.pearsonr(diff[c, :]+bpmES[c, :], bpmES[c, :])
        CC[c] = r
    return round(float(CC),2)

def LinCorr(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes CCC """
    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length

    if m < 2:
        print('> Warning: Correlation cannot be calculated for signals with len < 2. Returning NaN')
        return np.nan

    CCC = np.zeros(n)
    for c in range(n):
        # -- Lin's Concordance Correlation Coefficient
        ccc = concordance_correlation_coefficient(bpmES[c, :], diff[c, :]+bpmES[c, :])
        CCC[c] = ccc
    return round(float(CCC),2)

def concordance_correlation_coefficient(bpm_true, bpm_pred):
    """ Utility to compute the CCC """
    cor=np.corrcoef(bpm_true, bpm_pred)[0][1]
    mean_true = np.mean(bpm_true)
    mean_pred = np.mean(bpm_pred)   
    var_true = np.var(bpm_true)
    var_pred = np.var(bpm_pred)  
    sd_true = np.std(bpm_true)
    sd_pred = np.std(bpm_pred)   
    numerator = 2*cor*sd_true*sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator/denominator
