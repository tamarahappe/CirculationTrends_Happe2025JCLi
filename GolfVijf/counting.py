import os
import sys
import warnings
import numpy as np
import pandas as pd
import xarray as xr

"""
Counting is for operations on binarized/multiclass data that can result from thresholding (integers for classes)
"""

def count_combinations(digitized : pd.DataFrame, normalize: bool = False, classes: dict = None) -> xr.DataArray:
    """
    Digitized frame has integers for classes. 
    returns count matrix. If normalized then frequencies (count/total)
    Classes can be presupplied (only neccesary if potentially not all are
    """
    assert not np.isnan(digitized.values).any(), 'no NaN allowed'
    varnames = digitized.columns
    if classes is None:
        classes = {v:np.unique(digitized[v]) for v in varnames}  # Already sorted by np.unique
    counts = xr.DataArray(np.zeros(tuple(len(c) for c in classes.values())),
                            dims = tuple(varnames),
                            coords = classes)
    for i in range(len(digitized)):
        indices = tuple(digitized.iloc[i])
        counts[indices] += 1
    if normalize:
        counts = counts/counts.sum()
    return counts

def count_combinations_rolling(digitized : pd.DataFrame, nyearslice: int = 21, normalize: bool = False) -> xr.DataArray:
    """
    Digitized frame has integers for classes. 
    returns count matrix. If normalized then frequencies (count/total)
    """
    assert (nyearslice % 2) == 1,'only odd numbers are allowed for nyearslice'
    if nyearslice == 1:
        centeryears = digitized.index.year.unique().sort_values()
    else:
        centeryears = digitized.index.year.unique().sort_values()[(nyearslice//2):-(nyearslice//2)]
    print(centeryears)
    # Pre-allocation
    overall_classes = {v:np.unique(digitized[v]) for v in digitized}
    overall_counts = count_combinations(digitized=digitized)
    rolling_counts = xr.DataArray(np.zeros(overall_counts.shape + (len(centeryears),)), 
                            dims = overall_counts.dims + ('year',),
                            coords = overall_counts.coords)
    rolling_counts.coords.update({'year':centeryears.values})

    for i, centeryear in enumerate(centeryears):
        roll_sl = slice(pd.Timestamp(f'{centeryear - nyearslice//2}-01-01'),pd.Timestamp(f'{centeryear + nyearslice//2}-12-31'))
        roll_digitized = digitized.loc[roll_sl,:] 
        #print(i, centeryear, roll_digitized)
        count = count_combinations(digitized=roll_digitized,normalize=normalize, classes = overall_classes)
        rolling_counts.loc[...,centeryear] = count
    return rolling_counts

def sum_counts(counts : xr.DataArray, varnames : list) -> pd.DataFrame:
    """
    making use of the count combinations function
    aggregating the multivariate counting matrix into a 2D dataframe
    varnames are needed to not overwrite another dimension
    """
    sums = {}
    for var in varnames:
        othervars = varnames.copy() 
        othervars.remove(var)
        sums[var] = counts.sum(othervars).to_pandas()
    return pd.concat(sums, axis = 0, names = ['varname','class'])

def _intermediate_func(digitized : pd.DataFrame, normalize: bool = False, classes: dict = None) -> pd.DataFrame:
    """
    Not counting combinations but instead counting individual occurrences of each c 
    """
    combs = count_combinations(digitized, normalize = normalize, classes = classes)

    sums = sum_counts(combs, varnames = list(combs.dims))
    return sums

def count_imprecise_per(digitized: pd.DataFrame, per: str = 'year', normalize: bool = True, return_background : bool = False) -> pd.DataFrame:
    """
    Digitized frame has integers for classes. 
    returns sums. per is the lowest resolution at which it is returned
    axis is left-stamped.
    """
    assert per in ['year','month'], f'{per} not available'
    
    if per == 'year':
        monmin = digitized.index.month.min()
        grouper = pd.DatetimeIndex([pd.Timestamp(year = i.year, month = monmin, day = 1) for i in digitized.index])
    else:
        grouper = pd.DatetimeIndex([pd.Timestamp(year = i.year, month = i.month, day = 1) for i in digitized.index])

    #pd.MultiIndex.from_arrays([digitized.index.year,getattr(digitized.index, per)], names = ['year',per])
    overall_classes = {v:np.unique(digitized[v]) for v in digitized}

    reduced = digitized.groupby(grouper).apply(lambda x: _intermediate_func(x, normalize = True, classes = overall_classes))

    return reduced

if __name__ == '__main__':
    """
    Testing purposes
    """
    #digitized = pd.read_csv(os.path.expanduser('~/Documents/Timeseries_DDDays110_Stream1p5STD_Stream1p5STDNA.csv'),index_col = 0).set_index('dates')
    digitized = pd.read_csv('/scistor/ivm/the410/WAVE5/Timeseries_DDDays110_Stream1p5STD_Stream1p5STDNA.csv',index_col = 0).set_index('dates')
    digitized.index = pd.DatetimeIndex(digitized.index).floor('d')
    reduced = imprecise_count_per(digitized, per = 'year')

    combs = count_combinations(digitized, normalize = True)
    
    sums = sum_counts(combs, varnames = list(combs.dims))

    roll = count_combinations_rolling(digitized, normalize = True)

    test = sum_counts(roll, varnames = list(combs.dims))
