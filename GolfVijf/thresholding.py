import os
import sys
import warnings
import numpy as np
import pandas as pd
import xarray as xr

def digitize(combined_frame : pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """
    Supply a frame with timeseries that need to be classified/digitized
    returns frame with same shape and integers for classes
    shape (n_samples, n_variables), 
    Also supply a frame with thresholds
    such as the output of "combined_frame.quantile([0.33,0.66])"
    shape (nthresholds, n variables)
    """
    which_category = combined_frame.copy()
    for var in combined_frame.columns:
        which_category[var] = np.digitize(combined_frame[var], thresholds[var])
    return which_category

def _quantile_per_var(df : pd.DataFrame, quantiles : pd.DataFrame):
    """
    contrary to pd.DataFrame.quantile this function can handle 
    unique quantiles specified per variable
    """
    thresholds = quantiles.copy()
    for var in thresholds.columns:
        thresholds[var] = df[var].quantile(quantiles[var]).values
    return thresholds

def digitize_trended(combined_frame : pd.DataFrame, quantiles : pd.DataFrame, nyearslice: int = 21):
    """
    Supply a frame with timeseries that need to be classified/digitized
    returns frame with same shape and integers for classes
    A frame of quantiles can be supplied. shape (n_samples, n_variables), 
    Then these are estimated repeatedly for each rolling window.
    nyear length can be chosen
    """
    assert (nyearslice % 2) == 1,'only odd numbers are allowed for nyearslice'
    if nyearslice == 1:
        centeryears = combined_frame.index.year.unique().sort_values()
    else:
        centeryears = combined_frame.index.year.unique().sort_values()[(nyearslice//2):-(nyearslice//2)]
    which_category = combined_frame.copy().astype(int)

    for i, centeryear in enumerate(centeryears):
        roll_sl = slice(pd.Timestamp(f'{centeryear - nyearslice//2}-01-01'),pd.Timestamp(f'{centeryear + nyearslice//2}-12-31'))
        roll_comb = combined_frame.loc[roll_sl,:] 
        thresholds = _quantile_per_var(df = roll_comb, quantiles = quantiles)
        roll_categories = digitize(roll_comb, thresholds=thresholds)
        if i == 0: # Filling from zeroth enry onwards
            which_category.loc[slice(None,roll_categories.index.max()),:] = roll_categories
        else: # Otherwise from central year until the end
            which_category.loc[slice(f'{centeryear}-01-01',roll_categories.index.max()),:] = roll_categories.loc[slice(f'{centeryear}-01-01',None),:] # replacing centeryear and beyond

    return which_category

def digitize_separately(combined_frame: pd.DataFrame, per: str, quantiles : pd.DataFrame, return_thresholds : bool = True):
    """
    Supply a frame with timeseries that need to be classified/digitized
    returns frame with same shape and integers for classes
    A frame of quantiles can be supplied. shape (n_samples, n_variables), 
    Then these are estimated per background unit (removing e.g. seasonality if per week)
    TODO: Could be expanded to multiindex, year.month
    """
    assert per in ['year','month','week'], f'{per} not available'
    
    grouper = getattr(combined_frame.index, per)
    thresholds = combined_frame.groupby(grouper).apply(_quantile_per_var, quantiles = quantiles)

    which_category = combined_frame.copy()
    
    for group in grouper.unique():
        which_category.loc[grouper.get_loc(group),:] = digitize(combined_frame.loc[grouper.get_loc(group),:], thresholds=thresholds.loc[group,:])

    if return_thresholds:
        return which_category, thresholds
    else:
        return which_category
