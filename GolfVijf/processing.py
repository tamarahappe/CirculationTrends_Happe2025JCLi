import os
import sys
import numpy as np 
import pandas as pd 
import xarray as xr
import warnings
import netCDF4

from netCDF4 import Dataset,date2num, num2date
from pathlib import Path
from datetime import date
from collections import OrderedDict
from typing import Union
from scipy.stats import rankdata

from .utils import trendfit_robust

try:
    sys.path.append('/scistor/ivm/jsn295/Documents/telegates/')
    from telegates.processing import makeindex2
except ImportError:
    pass

datadir = Path('/scistor/ivm/the410/WAVE5/')

# Domains defined by (latmin, latmax, lonmin, lonmax)
subdomains = {'midlat':(35,70,None,None),
        'noratl':(35,70,-60,60),
        'noramer':(35,70,-170,-60),
        'europe':(35,70,-15,43),
        'ned':(50,54,-2,3),
        'circumglobal': (35,70,-180,180),
        'Both': (35,70,-100,110),
        'US_Atl':(35,70,-100,0),
        'EURASIA': (35,70,15,110),
        'tropics':(-20,30,-180,180),
        'nhext':(-20,90,-180,180)
             }

bookkeeping_names = ['variable','domain','months','ndayavg']

def spatial_subset(ds: Union[xr.Dataset, xr.DataArray], subdomain: str = 'midlat'):
    """
    Spatially subsetting an xarray object, based on latitude and longitude
    Checking if latitude is sored decreasing.
    """
    latmin, latmax, lonmin, lonmax = subdomains[subdomain]
    if np.diff(ds.latitude)[0] < 0:
        # Spatial cropping, latitude is stored decreasing, longitude increasing
        subset = ds.sel(latitude = slice(latmax, latmin), longitude = slice(lonmin, lonmax))
    else: # latitude is increasing
        subset = ds.sel(latitude = slice(latmin, latmax), longitude = slice(lonmin, lonmax))
    return subset

def load_and_subset_data(variable: str = 'STREAM500', ndays: int = 1, subdomain: str = 'midlat', months: list[int] = [6,7,8]) -> xr.DataArray:
    """
    Loading of data
    ndays = timescale of runmean averages by cdo (ignored if not loading streamfunction)
    subdomain can be chosen
    also a subset of months, e.g. [7,8] for July, August
    """
    assert variable in ['STREAM250', 'STREAM500', 'MSLP','T2M', "OLR"]
    assert subdomain in subdomains.keys(), f'choose one of {subdomains.keys()}'
    assert all(m in [6,7,8] for m in months), 'only JJA are available.' 
    
    var_dir = {'STREAM250':"stream", 'STREAM500':"stream", 'MSLP':"msl",'T2M':"t2m", "OLR":"ttr"}
    
    if subdomain=='midlat':
        subd_str = 'midlats'
    else:
        subd_str = subdomain
        
    if variable == "STREAM250": #IF var is stream250, runmean are available 
        assert ndays in [1,5,14], 'only daily, 5day, and 14day- means are currently possible'
        if ndays == 1:
            ds = xr.open_dataset(datadir / f'1940-2023_{variable}_{subd_str}_JJA.nc')
        else:
            print("WARNING: only 1950-2022 available")
            ds = xr.open_dataset(datadir / f'1950_2022_stream_250hPa_lonlat_{subd_str}_JJA_{ndays}drunmean.nc')     

    elif variable == "OLR":
        ds = xr.open_dataset(datadir / f'1940-2023_{variable}_{subd_str}_JJA.nc')
        
    else:
        ds = xr.open_dataset(datadir / f'1940-2023_{variable}_{subd_str}_JJA.nc')
    
    da = ds[var_dir[variable]]

    # Spatial cropping
    subset = spatial_subset(ds = da, subdomain = subdomain)

    # Temporal subsetting
    is_a_desired_month = xr.concat([da.time.dt.month == m for m in months], dim = 'month').any('month')
    subset = subset.sel(time = is_a_desired_month)

    # flooring the time coordinates, for easier matching to dddays
    subset.coords['time'] = subset.coords['time'].dt.floor('d') 
    return subset 

def remove_spatmean_variability(da: xr.DataArray, how: str = 'all', return_removed_signal: bool = False) -> xr.DataArray:
    """
    Removes spatial mean (e.g. NH midlatitude) variability from every gridpoint
    such that joint variability is removed (e.g. seasonal warming).
    da: daily input data, such as out of load_and_subset_data
    how: choice for how much to remove: 'all', 'interannual_only', 'seasonality_only'
    """
    assert how in ['all', 'interannual_only','seasonality_only']
    midlatmean = da.mean(['latitude','longitude'])
    if how == 'all':
        da = da - midlatmean # Automatic alignment of axes
        removed = midlatmean
    elif how == 'interannual_only':
        annual_midlatmean = midlatmean.groupby(midlatmean.time.dt.year).mean()
        da = da.groupby(da.time.dt.year) - annual_midlatmean
        removed = annual_midlatmean
    else:  # seasonality only by construction
        week_midlatmean = midlatmean.groupby(midlatmean.time.dt.weekofyear).mean()
        da = da.groupby(da.time.dt.weekofyear) - week_midlatmean
        removed = week_midlatmean
    if return_removed_signal:
        return da, removed
    else:
        return da

def compute_trend_pattern(da: xr.DataArray, aggregate_first: bool = True, aggregate_how: str = 'mean', startyear: int = None, endyear: int = None, standardize: bool = False, sign: bool=False) -> xr.DataArray:
    """
    aggregate_first = True, First seasonal aggregation, then fitting trend
    aggregation method is mean by default but can be min or max
    startyear (optional) inclusive
    endyear (optional) inclusive
    returns a DataArray with the slope (if sign=True, also returns pvals)
    """
    da = da.sel(time = slice(startyear if (startyear is None) else f'{startyear}-01-01',
            endyear if (endyear is None) else f'{endyear}-12-31'))
    if aggregate_first:
        assert aggregate_how in ['min','mean','max','quantile']
        grouped = da.groupby(da.time.dt.year)
        method = getattr(grouped, aggregate_how)
        data = method()
    else:
        data = da
    if not sign:
        intercept_slope = trendfit_robust(da = data, standardize = standardize)
        pattern = intercept_slope.sel(what = 'slope')
        pattern.attrs.update({'aggregate_first':aggregate_first,'aggregate_how':aggregate_how})
        return pattern
    elif sign:
        intercept_slope = trendfit_robust(da = data, standardize = standardize)
        pattern = intercept_slope.sel(what = 'slope')
        pattern.attrs.update({'aggregate_first':aggregate_first,'aggregate_how':aggregate_how})
        pvals = intercept_slope.sel(what = 'pvalue')
        pvals.attrs.update({'aggregate_first':aggregate_first,'aggregate_how':aggregate_how})
        return pattern, pvals

def make_wpd_sst_index(nday_average = 5) -> pd.DataFrame:
    """
    Using Chiem's code for West Pacific Dipole index (telegates repository)
    based on SST anomalies (deseasonalized with 5, 6, or 7 degree polynomials)
    """
    assert (nday_average % 2) == 1, 'Because of center time stamping only odd amounts of days are allowed'
    wpd = makeindex2(deseason = True, remove_interannual = False, timeagg = nday_average, degree = 7)
    # In the original repository it is left-stamped so correct to center stamped.
    wpd.coords['time'] = wpd.coords['time'] + pd.Timedelta(nday_average//2, unit = 'd')
    df = wpd.to_dataframe()
    df.columns = pd.MultiIndex.from_tuples([(wpd.name,'westpacific','all',nday_average)], names = bookkeeping_names)
    return df

def mean_per_rank(values: np.ndarray, ranks: np.ndarray, timecoords: np.ndarray):
    """
    One dimensional application
    values and ranks need to have the same shape
    returns rankmeans (len(ranks),) and a version for values with the rankmean subtracted (len(values),)
    """
    values = xr.DataArray(values, dims = ('time',), coords = {'time':timecoords})
    ranks = xr.DataArray(ranks, dims = ('time',), coords = {'time':timecoords}, name = 'rank')
    local_rank_day_mean = values.groupby(ranks).mean() # Creates a new dimension, namely 'rank'
    local_rank_day_anomalies = values - local_rank_day_mean.sel(rank = ranks).drop('rank')
    return local_rank_day_mean, local_rank_day_anomalies


def calculate_rank_anoms(variable: str = ['MSLP','T2M','STREAM500','STREAM250'], subdomain: str = 'midlat', remove_variability: str = None, startyear: int = 1940, use_dask: bool = False, njobs = 10):
    """
    Create rank anomalies arrays based on the method of Rothlisberger:
    Röthlisberger, M., Sprenger, M., Flaounas, E., Beyerle, U., & Wernli, H. (2020).
    The substructure of extremely hot summers in the Northern Hemisphere. 
    Weather and Climate Dynamics, 1(1), 45-62.
    possible to do some preprocessing by removing spatial mean variability (see remove_spatmean_variability for valid arguments)
    and also starting only at a certain year
    """
    da = load_and_subset_data(variable = variable, ndays = 1, subdomain = subdomain, months = [6,7,8])
    da = da.sel(time = slice(f'{startyear}-01-01',None))
    if not (remove_variability is None):
        da = remove_spatmean_variability(da = da, how = remove_variability) 
    # We can groupby year, because data is JJA only and tied to one year.
    rank_within_season = da.groupby(da.time.dt.year).apply(lambda a: rankdata(a, axis = da.dims.index('time'), method = 'ordinal')) # time, latitude, longitude
    rank_within_season.name = 'rank'
    if use_dask:
        from dask.distributed import Client
        import dask.array as daskarray
        # https://github.com/pydata/xarray/issues/6803
        client = Client(n_workers = njobs)
        vd = daskarray.from_array(da, chunks = da.shape[:1] + (5,5), name = da.name)
        rd = daskarray.from_array(rank_within_season, chunks = rank_within_season.shape[:1] + (5,5), name = rank_within_season.name)
        future_vd = client.scatter(dict(vd.dask))
        future_rd = client.scatter(dict(rd.dask))
        vd = daskarray.Array(future_vd, name = vd.name, chunks = vd.chunks, dtype = vd.dtype, meta = vd._meta, shape = vd.shape)
        rd = daskarray.Array(future_rd, name = rd.name, chunks = rd.chunks, dtype = rd.dtype, meta = rd._meta, shape = rd.shape)
        values = xr.DataArray(vd, name = vd.name, dims = da.dims, coords = da.coords)
        ranks = xr.DataArray(rd, name = rd.name, dims = rank_within_season.dims, coords = rank_within_season.coords)
        print(values)
        print(ranks)
    else:
        values = da
        ranks = rank_within_season
    # Cannot do vectorized (ortoganal) indexing, based on the ranks, e.g. da[rank_within_season == 71]
    # so now we basically need one operation per lat,lon, wih two outputs
    #local_rank_day_mean, local_rank_day_anomalies = mean_per_rank(da.values[:,0,0], rank_within_season.values[:,0,0], da.time)
    rank_day_mean, rank_day_anomalies = xr.apply_ufunc(mean_per_rank, values, ranks, 
            #exclude_dims = set(('time',)), # Would be dropped from output (rank day anomalies) as well
            input_core_dims=[['time'],['time']],
            output_core_dims=[['rank'],['time']], 
            kwargs={'timecoords':da.time},
            vectorize = True, dask = "parallelized",
            output_dtypes=[np.float32,np.float32],
            dask_gufunc_kwargs = dict(output_sizes={'rank':92})) # rank, latitude, longitude

    if use_dask:
        rank_day_mean = rank_day_mean.compute()
        rank_day_anomalies = rank_day_anomalies.compute()
        client.shutdown()
    rank_day_mean.coords['rank'] = np.arange(1,93)
    return da, rank_within_season, rank_day_mean, rank_day_anomalies.transpose(*da.dims)
    
path_data='/scistor/ivm/the410/WAVE5'

def _from_txt_to_indice_JJA(data):
    """
    takes in list of floats, with every float consisting of yyyymmdd, starting from 1950, for JJA (summer=92 days)
    returns array of indices corresponding to summer days, for n years (e.g. indice 0 would be 1950-06-01)   
    """
    out_array = []
    for i, day in enumerate(data):
        if int(str(int(day))[4:6]) == 6:
            indice = (((int(str(int(day))[:4]) - 1950) * 92 ) + ((int(str(int(day))[4:6]) - 6) * 30) + (int(str(int(day))[6:])))
            #print(indice, int(str(int(day))[:4]) , (int(str(int(day))[4:6])), (int(str(int(day))[6:])))
            out_array.append(indice)
        elif int(str(int(day))[4:6]) != 6:
            indice = (((int(str(int(day))[:4]) - 1950) * 92 ) + 30 + ((int(str(int(day))[4:6]) - 7) * 31) + (int(str(int(day))[6:])))
            #print(indice, int(str(int(day))[:4]) , (int(str(int(day))[4:6])), (int(str(int(day))[6:])))
            out_array.append(indice)
    return np.array(out_array, dtype=int)

def DDdata_txt_to_timeseries(filename:str):
    """Takes in the filename of DDdata.txt from Fabio, returns  xarray with DDday timeseries
    WARNING: DDday data and dates are from 1950-2022"""
    #read in ddday data, convert to indices
    txt = pd.read_csv(f'{path_data}/{filename}', header = None)[0].to_list()
    indices = _from_txt_to_indice_JJA(txt)
    #read in timestamps
    times = Dataset(fr"{path_data}/old_era5/1950_2022_stream_250hPa_lonlat_midlats_JJA.nc")["time"]
    dates = netCDF4.num2date(times, units = times.units, calendar=times.calendar)
    #to get correct time format
    time= xr.open_dataset(fr"{path_data}/old_era5/1950_2022_stream_250hPa_lonlat_midlats_JJA.nc")["time"]
    #convert
    np_array = np.zeros_like(dates)
    np_array[indices]=1 #DDdays get value 1
    xr_array = xr.DataArray(data=np_array, dims=["time"], 
             coords=dict(time=time), 
             attrs=dict(description=f"DDdays {filename[-8:-4]} binary", units="1 for DDday, 0 if not"))
    return xr_array

def create_event_length_per_year(DD_series_xarray):
    """
    Creates two nested lists, 1) for each year a list with event lengths per event and 2) for each year a list with DDday indices for that year (e.g. 2 = day 2 of that year)
    IN: Xarray with timeseries of DDdays consiting of 0's and 1's, as returned by DDdata_txt_to_timeseries()
    OUT: year_lengths:list of lists, year_indices:list of lists
    """
    ## I want for each year a list with event lengths
    ##DD_series_xarray is an xarray with 0 and 1s, corresponding with whether that day is a DDday or not. (SUMMER ONLY)
    ## DD_series_xarray is the output from DDdata_txt_to_timeseries()
    
    ##returns:
    ##year_lenghts: for each year, a list with the event lenghts per event. len(year_lenghts[year])=nr of events that year
    ## year_indices:for each year, a list with the indice of the indice of the day of that summer, for each ddday 

    diff = np.diff(DD_series_xarray)
    year_lengths = []
    year_indices = []

    x = DD_series_xarray.time.dt.year
    yearsn = np.unique(np.unique(DD_series_xarray.time.dt.year))

    for year in yearsn: #yearsn
        summer_days=DD_series_xarray[np.where(x==year)]
        year_events = []
        event_indices=[]
        event_length = 0
        for i, day in enumerate(summer_days):
            if i == 91:
                "end of summer"
                if event_length>0:
                    "end of event"
                    year_events.append(event_length)
                    event_length=0
            elif day==1:
                "(start of) event"
                event_length+=1
                event_indices.append(i)
            elif day==0:
                if event_length>0:
                    "end of event"
                    year_events.append(event_length)
                    event_length=0
        year_lengths.append(year_events)
        year_indices.append(event_indices)
    return year_lengths, year_indices
