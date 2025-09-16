import warnings
import numpy as np 
import pandas as pd 
import xarray as xr
import matplotlib.pyplot as plt
from netCDF4 import Dataset,date2num, num2date
from scipy.stats import linregress
import os
import netCDF4

try:
    import cartopy.crs as ccrs
except ImportError:
    pass

from typing import Union

def xr_linregress(y, x, min_count = 30):
    """
    Fit between two 1D arrays: y = a + b*x
    Returns 1D array [a, b]
    Removal of NaNs because scipy linregress cannot handle them
    Also setting a minimum amount of samples
    """
    isnan = np.isnan(y)
    y = y[~isnan]
    x = x[~isnan]
    if len(y) >= min_count:
        reg_result = linregress(x = x, y = y)
        result = [reg_result[1], reg_result[0], reg_result[3]] # intercept, slope unit y / unit x, p_value (two-sided) 
    else:
        warnings.warn(f'Too little samples found. Minimum: {min_count}')
        result = [np.nan, np.nan, np.nan]
    return np.array(result, dtype = np.float32) 

def trendfit_robust(da : xr.DataArray, standardize: bool = False):
    """
    Using ufunc on array of which one dimension should be 'time' or 'year',
    returns same array but with this dimension replaced by 
    dimension of length three: [intercept, slope, p-value] slope is per year
    """
    # Setting up an x variable.
    if not ('year' in da.dims):
        x_year = da.time.dt.year # retains the dimension name 'time'
        do_not_broadcast = 'time'
    else:
        x_year = da.year # Dimension name itself is also 'year'
        do_not_broadcast = 'year'
    coefs = xr.apply_ufunc(xr_linregress, da, x_year, exclude_dims = set((do_not_broadcast,)),
            input_core_dims=[[do_not_broadcast],[do_not_broadcast]],
            output_core_dims=[['what']], 
            vectorize = True, dask = "parallelized",
            output_dtypes=[np.float32],
            dask_gufunc_kwargs = dict(output_sizes={'what':3}))
    coefs.coords.update({'what':['intercept','slope','pvalue']})
    if standardize:
        coefs = coefs / da.std(do_not_broadcast) # Cannot do skipna = False because then many slices would fail.
        coefs.attrs.update({'units':'std/yr'})
    else:
        coefs.attrs.update({'units':f'{da.attrs["units"]}/yr'})
    coefs.attrs.update({'test':'two-sided'})
    return coefs

def fit_trends_per_gridpoint(var_array, start_year:int, n_years:int, summerdays: int=122):

    #caculate yearly values
    var_midlats_year = np.zeros((n_years, var_array.shape[1], var_array.shape[2]))
    x = np.arange(start_year, start_year+n_years, 1)
    for i, year in enumerate(x):
        #print(year, i)
        var_midlats_year[i] = np.nanmean(var_array[i*summerdays:(i+1)*summerdays], axis=(0))

    return var_midlats_year
        
    #now calculate the trend per gridpoint
    midlats_yearly_trend_per_gridpoint = np.zeros((var_array.shape[1], var_array.shape[2])) #out array, empty
    x = np.arange(0, n_years, 1)
    for i in range(var_array.shape[1]):
        for j in range(var_array.shape[2]):
            #iterate over each gridpoint, calculate yearly trend, and save this in array
            #print(x.shape, stream_midlats_year.shape)
            midlats_yearly_trend_per_gridpoint[i,j] = np.polyfit(x,var_midlats_year[:,i,j],1)[0] 

    return midlats_yearly_trend_per_gridpoint

def plot_trends_per_gridpoint(var_array, var_lons, var_lats, var_name:str, start_year:int, summerdays = 122,
                              mask=False, mask_array=None, return_array=False, cumulative_trend=False):
    n_years = int(var_array.shape[0]/summerdays)

    midlats_yearly_trend_per_gridpoint = fit_trends_per_gridpoint(var_array = var_array, start_year = start_year, n_years = n_years, summerdays = summerdays)
            
    #now plot
    cmap = plt.get_cmap('RdBu_r')
    #cmap.set_bad(color = 'white')
    trend_ = "linear yearly trend"


    fig, ax = plt.subplots(1,1, figsize=(50,30))
    ax1 = plt.subplot(111, projection=ccrs.PlateCarree())
    
    if mask:
        var = midlats_yearly_trend_per_gridpoint * mask_array

    if not mask:
        var = midlats_yearly_trend_per_gridpoint
        
    if cumulative_trend:
        print("Cumulative trend is calculated by trend per year * n_years")
        var = var * n_years
        trend_ = f"linear trend over {n_years} years"
        
    mx = int(max(np.max(var), abs(np.min(var))))
    cs = ax1.pcolormesh(var_lons, var_lats, var,
                 transform=ccrs.PlateCarree(), cmap=cmap, vmin=-mx, vmax=mx, edgecolors = 'None')

    ax1.coastlines()
    ax1.set_title(f"ERA5 {start_year}-{start_year+n_years-1} {var_name} - {trend_}", fontsize=35)
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False

    fig.colorbar(cs, ax=ax1, fraction=0.005, pad=0.02, label="trend per year")

    plt.show()
    
    if return_array:
        print(f"returning array for {var_name}, mask is {mask} (if mask is True the sea points are set to 0)")
        return var


def onelag_cov(onepatterndiff : np.ndarray, precursordiff : np.ndarray):
    """
    covariance timeseries for one lag, unbiased estimate by n - 1   
    input: onepatterndiff 1D (nobs,), precursordiff 2D (ntime,nobs)
    output: 1D (ntime,)
    """
    return(np.nansum(precursordiff * onepatterndiff, axis = -1)/(len(onepatterndiff) - np.isnan(onepatterndiff).sum()))

def covariance_timeseries(var_pattern, var_array):
    """
    calculates covariance timeseries using onelag_cov, for a given pattern.
    IN: var_pattern: array of (lat,lon) - pattern you want to check covariance with
    var_array: array of (time, lat, lon) - for which you want to get a timeseries with the covariance
    OUT: out_series: array of (time,) - with covariance for each timestep
    """
    var_time_mean = np.nanmean(var_array, axis=(1,2))
    var_diff = np.empty_like(var_array)
    for i in range(var_diff.shape[0]):
        var_diff[i] = var_array[i] - var_time_mean[i]
    out_series = onelag_cov(var_pattern.flatten(), var_diff.reshape((var_diff.shape[0], (var_diff.shape[1] * var_diff.shape[2]))))
    return out_series

def covariance_timeseries2(pattern: Union[xr.DataArray, np.ndarray], array: Union[xr.DataArray, np.ndarray], latitude_weighting: bool = True, normalize: bool = False) -> Union[xr.DataArray, np.ndarray]:
    """
    calculates covariance timeseries using onelag_cov, for a given pattern.
    IN: var_pattern: array of (lat,lon) - pattern you want to check covariance with
    var_array: array of (time, lat, lon) - for which you want to get a timeseries with the covariance
    OUT: out_series: array of (time,) - with covariance for each timestep
    weighting as in: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient
    normalization means that you get a correlation value
    """
    # Flattening, and numpy from here on, reconstruction later
    if isinstance(pattern, xr.DataArray): # Stacked will become the last fimension
        pattern = pattern.stack({'latlon':['latitude','longitude']}) 
        array = array.stack({'latlon':['latitude','longitude']}) 
        original_coords = array.coords 
        pattern = pattern.values
        array = array.values
        reconstruct = True
    else:
        pattern = pattern.flatten()
        array = array.reshape((array.shape[0], (array.shape[1] * array.shape[2])))
        reconstruct = False
    if latitude_weighting: # array should be an xr.DataArray
        assert reconstruct, 'lat weighting only possible with access to coordinates, supply xr.DataArray'
        weights = np.cos(np.deg2rad(original_coords['latlon'].latitude.values))
    else:
        weights = np.ones_like(pattern) # Equal weight
    weights = weights/weights.sum() # Normalization, such that summing to one
    spatial_array_mean = np.nansum(weights[np.newaxis,:] * array, axis = -1) # shape (time,)
    spatial_pattern_mean = np.nansum(weights*pattern) # just a number
    array_diff = array - spatial_array_mean[:,np.newaxis] # Broadcasting to all gridcells
    pattern_diff = pattern - spatial_pattern_mean
    covariance = np.nansum(weights[np.newaxis,:] * (array_diff * pattern_diff[np.newaxis,:]), axis = -1)
    if normalize:
        variance_array = np.nansum(weights[np.newaxis,:] * (array_diff * array_diff), axis = -1)
        variance_pattern = np.nansum(weights * (pattern_diff * pattern_diff))
        covariance = covariance / np.sqrt(variance_array*variance_pattern)

    if reconstruct:
        reconstruct_dims = list(original_coords.dims)
        reconstruct_dims.remove('latlon')
        covariance = xr.DataArray(covariance, dims = reconstruct_dims, coords = {d:original_coords[d] for d in reconstruct_dims})
        
    return covariance

def data_for_pcolormesh(array, shading:str):
    """Xarray array to usuable things"""
    lats = array.latitude.values # Interpreted as northwest corners (90 is in there)
    lons = array.longitude.values # Interpreted as northwest corners (-180 is in there, 180 not)
    if shading == 'flat':
        lats = np.concatenate([lats[[0]] - np.diff(lats)[0], lats], axis = 0) # Adding the sourthern edge 
        lons = np.concatenate([lons, lons[[-1]] + np.diff(lons)[0]], axis = 0)# Adding the eastern edge (only for flat shating)
    return lons, lats, array.values.squeeze()
