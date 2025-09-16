import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from netCDF4 import Dataset,date2num, num2date
import os
import netCDF4
import cartopy.crs as ccrs
from scipy.stats import ttest_ind
import matplotlib.patches as mpatches
import sys

sys.path.append('/scistor/ivm/the410/GolfVijf/')

from GolfVijf.thresholding import *
from GolfVijf.utils import covariance_timeseries2
from GolfVijf.processing import load_and_subset_data, compute_trend_pattern, remove_spatmean_variability, spatial_subset, subdomains

path_data='/scistor/ivm/the410/WAVE5'

### Loading streamfunction data
var = "STREAM250" #"T2M" #"STREAM250"
longname =  "Streamfunction 250 hpa" #"Streamfunction 250 hpa" "2m surface temperature"
preprocessing = "seasonality_only" #"all", 'interannual_only', 'seasonality_only'"
unit = "m2/s" #"degrees C"

# Determining locations and consistent color coding
sf_extreme_locs =[(58,-150),(42,-125),(41,-86),(56,-60),(52,-20),(47,3),(52,30),(39.5,65),(45,100),(38,177)] # lat, lon

subdomains_ = {
             'circumglobal': (35,70,-180,180),
        'Both': (35,70,-100,110),
'US_Atl':(35,70,-100,0),
        'EURASIA': (35,70,15,110)}

colors_dict  = {'US_Atl':'lawngreen',
        'EURASIA': 'yellow',
             'Both': 'blue',
             'circumglobal': 'black'}


def load_ERA5_sst_data(lower_year:int = 1940, upper_year:int = 2019, time_aggr:str="weekly",
                      path_data:str='/scistor/ivm/the410/WAVE5'):
    '''
    loads in ERA5 sst data, cuts in given the period, de-seasonalises and aggregates given the time_aggr
    returns aggregated xarray
    '''
    
    print(lower_year, upper_year, time_aggr)
    
    remove_seasonality_sst = True
    detrending_how_sst = "None"
    
    
    SSTs = xr.open_dataset(f"{path_data}/SST_era5_NHExt_0.25degr_19400101-20240229_JJA.nc")
    
    SSTs = SSTs["sst"].sel(time=SSTs.time.dt.year<=upper_year) #because last years are corrupted still... 
    SSTs = SSTs.sel(time=SSTs.time.dt.year>=lower_year) #because last years are corrupted still... 
    
    SSTs_weekly = SSTs.resample(time="1W").mean(skipna=True)
    
    if remove_seasonality_sst:
        sst_weekly_ = SSTs_weekly.sel(time=SSTs_weekly.time.dt.month.isin([6,7,8])) #to just select JJA
        ## calculate weekly mean, for each gridpoint 
        SSTs_weekly_mean = SSTs_weekly.groupby(SSTs_weekly.time.dt.week).mean(dim="time")
        #to resample to just the summer
        SSTs_weekly_mean = SSTs_weekly_mean.sel(week=SSTs_weekly_mean.week.isin([22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,35]) )
        #deseasanlise by removing the weekly mean
        SSTs_weekly = sst_weekly_.groupby(sst_weekly_.time.dt.week) - SSTs_weekly_mean
        
    if time_aggr == 'weekly':
            SSTs_aggregated = SSTs_weekly.astype("float32")
    elif time_aggr == 'yearly':
            SSTs_aggregated = SSTs_weekly.groupby("time.year").mean(dim="time").astype("float32")
    elif time_aggr == "monthly":
            SSTs_aggregated = SSTs_weekly.groupby("time.month").mean(dim="time")
            SSTs_aggregated = SSTs_aggregated.where(
                SSTs_aggregated.month.isin([6,7,8]), drop=True).astype("float32")
            
    return SSTs_aggregated



def load_ERA5_stream_data(stream_azonal:bool=True, stream_extended:bool=True,
                          lower_year:int = 1940, upper_year:int = 2019, time_aggr:str="weekly",
                          path_data:str='/scistor/ivm/the410/WAVE5'):

    #LOAD AND/OR AGGREGATE THE CORRECT STREAMFUNCTION DATA
    preprocessing = "seasonality_only" 

    if not stream_extended:
        #load the data
        stream = load_and_subset_data(variable = "STREAM250", months = [6,7,8], subdomain = 'midlat', ndays = 1).astype("float32")
        stream_removed_seasonality = remove_spatmean_variability(stream, how=preprocessing)
        
        ## to slice the data 
        stream_removed_seasonality_sliced = stream_removed_seasonality.sel(time=stream_removed_seasonality.time.dt.year>=lower_year)
        stream_removed_seasonality_sliced = stream_removed_seasonality_sliced.sel(time=stream_removed_seasonality_sliced.time.dt.year<=upper_year)

        #remove grid point anomalies 
        spatial_mean = stream_removed_seasonality_sliced.mean(axis=0)
        stream_anoms = stream_removed_seasonality_sliced - spatial_mean
        aggregated_stream = stream_anoms.resample(time="1W").mean(dim="time")
        aggregated_stream = aggregated_stream.sel(time=aggregated_stream.time.dt.month.isin([6,7,8]))

    if stream_extended:
        stream_extended = load_and_subset_data(variable = "STREAM250", months = [6,7,8], subdomain = 'nhext', ndays = 1).astype("float32")
        stream_extended_removed_seasonality = remove_spatmean_variability(stream_extended, how=preprocessing)

        stream_removed_seasonality_sliced = stream_extended_removed_seasonality.sel(time=stream_extended_removed_seasonality.time.dt.year>=lower_year)
        stream_removed_seasonality_sliced = stream_removed_seasonality_sliced.sel(time=stream_removed_seasonality_sliced.time.dt.year<=upper_year)

        if stream_azonal:
            zonal_mean = stream_removed_seasonality_sliced.mean(axis=(0,2)) #zonal mean over time
            stream_anoms = stream_removed_seasonality_sliced - zonal_mean
        elif not stream_azonal:
            spatial_mean = stream_removed_seasonality_sliced.mean(axis=0) #gridpoint anomalies
            stream_anoms = stream_removed_seasonality_sliced - spatial_mean
        aggregated_stream = stream_anoms.resample(time="1W").mean(dim="time")
        aggregated_stream = aggregated_stream.sel(time=aggregated_stream.time.dt.month.isin([6,7,8]))

    return aggregated_stream


def load_ERA5_var_data(var, lower_year:int = 1940, upper_year:int = 2019, time_aggr:str="weekly",
                      path_data:str='/scistor/ivm/the410/WAVE5'):
    

    remove_seasonality = True
    detrending_how_t2m = "None"

    subd_dict = {"T2M":"midlat", "OLR":"tropics"}
    assert var in ["T2M", "OLR"]

    #load data
    VAR = load_and_subset_data(variable = var, months = [6,7,8], subdomain =subd_dict[var], ndays = 1).astype("float32")

    #It makes a difference whether you do the timeslicing before or after the seasonality removal
    VAR = VAR.where(VAR.time.dt.year>=lower_year, drop=True)
    VAR = VAR.where(VAR.time.dt.year<=upper_year, drop=True)

    VAR_weekly = VAR.resample(time="1W").mean(skipna=True)

    #remove seasonality
    VAR_weekly_ = VAR_weekly.sel(time=VAR_weekly.time.dt.month.isin([6,7,8])) #to just select JJA
    ## calculate weekly mean, for each gridpoint 
    VAR_weekly_mean = VAR_weekly.groupby(VAR_weekly.time.dt.week).mean(dim="time")
    #to resample to just the summer
    VAR_weekly_mean = VAR_weekly_mean.sel(week=VAR_weekly_mean.week.isin(
        [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,35]) )
    #deseasanlise by removing the weekly mean
    VAR_weekly = VAR_weekly_.groupby(VAR_weekly_.time.dt.week) - VAR_weekly_mean
    
    
    if time_aggr == 'weekly':
        VAR_aggregated = VAR_weekly
    elif time_aggr == 'yearly':
        VAR_aggregated = VAR_weekly.groupby("time.year").mean(dim="time")
    elif time_aggr == "monthly":
        VAR_aggregated = VAR_weekly.groupby("time.month").mean(dim="time")
        VAR_aggregated = VAR_aggregated.where(VAR_aggregated.month.isin([6,7,8]), drop=True)

    return VAR_aggregated


def aggregated_covars_region(region:str, data:xr.DataArray, trend_pattern:xr.DataArray, 
                             lower_year: int = 1940, upper_year: int = 2023, 
                             time_aggr : str = 'weekly'):
    '''
    Takes in trend pattern and data, and calculates daily covariance on the trend pattern for a given region.
    Returns a covariance timeseries, aggregated at specified timescale, for the specified time_period.
    
    Input data should be de-seasonalised, and further anomalies per gridpoint are calculated. 
    
    returns: Xarray timeseries of aggregated  
    '''

    ##
    print(region)
    print(subdomains[region])

    #get values for streamfunction
    subset_region = spatial_subset(data, subdomain=region)
    
    #calculate anomalies per gridpoint
    subset_region_spatial_mean = subset_region.mean(axis=0)
    subset_region_anoms = subset_region - subset_region_spatial_mean
    subset_region = subset_region_anoms

    #caslculate covariance of data with trend pattern
    subset_region_trend = spatial_subset(trend_pattern, subdomain=region)
    subset_region_covars = covariance_timeseries2(subset_region_trend, subset_region, normalize=True)

    # To slice the years of interest
    subset_region_covars_weekly_sliced = subset_region_covars.sel(
        time=subset_region_covars.time.dt.year>=lower_year)
    subset_region_covars_weekly_sliced = subset_region_covars_weekly_sliced.sel(
        time=subset_region_covars_weekly_sliced.time.dt.year<=upper_year)
    
    if time_aggr == 'daily':
#         data_covars = subset_region_covars_weekly_sliced  
        # to keep only summer months
        data_covars = subset_region_covars_weekly_sliced.sel(
            time=subset_region_covars_weekly_sliced.time.dt.month.isin([6,7,8]))
    
    elif time_aggr == "weekly":
        subset_region_covars_aggregated = subset_region_covars_weekly_sliced.resample(time="1W").mean()
        # to keep only summer months
        data_covars = subset_region_covars_aggregated.sel(
            time=subset_region_covars_aggregated.time.dt.month.isin([6,7,8]))
        
    elif time_aggr == "monthly":
        subset_region_covars_aggregated = subset_region_covars_weekly_sliced.resample(time="1M").mean()
        # to keep only summer months
        data_covars = subset_region_covars_aggregated.sel(
            time=subset_region_covars_aggregated.time.dt.month.isin([6,7,8]))
        
    elif time_aggr == "yearly":
        data_covars = subset_region_covars_weekly_sliced.resample(time="1Y").mean()
    
    data_covars.attrs = {"region":region, "time aggregation":time_aggr}
    return data_covars
    
    
def _significance_t_test(subset, data_without_subset):
    '''computes two sided t-test, Welch's
    with nan_policy = omit
    a
    returns t and p values
    
    subset= composites
    data is data without subset
    '''
    
    results = ttest_ind(subset, data_without_subset, 
                        equal_var=False, nan_policy="omit")
    
    return results[0], results[1]
    
def plot_this_VAR_pattern(var_to_plot, VAR_aggregated_data_subset, 
                          VAR_aggregated_positives, VAR_aggregated_positives_mean,
                          week_lag, region, model="ERA5",
                          lower_year=1940, upper_year=2019,
                          sign=True, covar_threshold=0.5,
                         aggregation="weekly", var_to_plot_name=""):
#     sign = True

    year_str= f"{lower_year}_{upper_year}"
    
    if var_to_plot_name == "": 
            var_to_plot_name = var_to_plot 
    
    if model == "ERA5":
        filename=f"{model}_{var_to_plot_name}_{week_lag}_{region}_{covar_threshold}_sign{sign}_{year_str}"
        fname_figure = f"/scistor/ivm/the410/WAVE5/figures/composites_ERA5/{filename}.png"
    else:
        filename=f"{model}_{var_to_plot_name}_{week_lag}_{region}_{covar_threshold}_sign{sign}_{year_str}"
        fname_figure = f"/scistor/ivm/the410/WAVE5/figures/composites_CMIP6/{filename}.png"
        
    if os.path.isfile(fname_figure): #check if plot already exists
        #figure already exists
        print("composite figure already exists")
    else:
        print("composite figure doesn't exists yeat")

        ## FOR VARS T2M, SST, OLR
        assert var_to_plot in ["T2M", "sst", "OLR", "stream", "stream_extended"]
        
        data = VAR_aggregated_positives_mean
        lons = VAR_aggregated_data_subset.longitude
        lats = VAR_aggregated_data_subset.latitude

        sst_title = f"{aggregation} detrended=False, week lag={week_lag}, T-Test={sign}, {year_str}"

        if var_to_plot == "T2M" :
            vmin, vmax = -2, 2
            unit="degr C"
        elif var_to_plot == "sst":
            vmin, vmax = -1, 1
            unit="degr C"
        elif var_to_plot == "OLR":
            vmin, vmax = -60000, 60000
            unit="J m**-2"
        elif var_to_plot == "stream" or var_to_plot == "stream_extended":
            vmin, vmax = -10000000, 10000000
            unit="m2/s"

        if sign:
            t, p = _significance_t_test(VAR_aggregated_positives, VAR_aggregated_data_subset)
#                 data = np.where(p <= 0.05, VAR_aggregated_positives_mean, np.NaN)
            
        title=f"{model} {var_to_plot_name} {lower_year}-{upper_year} {region} \n {sst_title} \n with JJA trend 1979-2023 \n and covar_tresh={covar_threshold}"



        ######
        shading = 'flat'

        cmap = 'RdBu_r'
        fig, ax = plt.subplots(figsize = (20,8), subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})
        im = ax.pcolormesh(lons, lats, data[:-1,:-1], transform = ccrs.PlateCarree(),
                                shading = shading, cmap = cmap, vmin=vmin, vmax=vmax)
        if sign:
            hatch = ax.contourf(lons, lats, p, transform = ccrs.PlateCarree(),
                                levels=[0, 0.05, 1],
                                hatches=["...", ""],
                                colors = 'none')
            artists, labels = hatch.legend_elements(str_format='{:2.1f}'.format)
            artist, label = [artists[0]], ["sign."]
            ax.legend(artist, label, handleheight=2, framealpha=1)

        coords = subdomains[region]
        color=colors_dict[region]
        ax.add_patch(mpatches.Rectangle(xy=[coords[2], coords[0]], width=coords[3]-coords[2], height=coords[1]-coords[0],
                                        facecolor='none', edgecolor=color, linewidth=4,
                                        transform=ccrs.PlateCarree()))

        ax.coastlines()
        ax.set_title(title)
        ax.gridlines(draw_labels = ['left','bottom'])
        fig.colorbar(im, ax=ax, fraction=0.005, pad=0.02, label=f"{unit}")
        plt.show()
        fig.savefig(fname_figure)
        
def create_composites_ERA5(var_name, 
                           region, 
                           daJJA_removed_seasonality,
                           daJJA_1979_None_trend,
                           covar_treshold:int=0.5, 
                           stream_azonal:bool=True,
                           lower_year:int= 1950, upper_year:int=2014,
                           time_aggregation:str="weekly",
                           Sign_masked=False,
                           week_lags=[1,2,3]):
    '''
    
    timeperiod here is also influencing over which period we de-seasonalize the data when loading, 
    via the loading functions 
    
    
    '''
    
    print(region, var_name, lower_year, upper_year, time_aggregation)
    
    # load data
    assert var_name in ["T2M", "OLR", "sst", "stream", "stream_extended"] 
    var_name_units = {"T2M":"degr C", "OLR":"J m**-2", "sst":"degr C", 
                      "stream":"m2/s", "stream_extended":"m2/s"}
    
    
    if var_name in ["T2M", "OLR"]:
        data_var_aggr = load_ERA5_var_data(var_name, 
                                           lower_year=lower_year, upper_year=upper_year, 
                                           time_aggr=time_aggregation)
    elif var_name == "sst":
        data_var_aggr = load_ERA5_sst_data(lower_year=lower_year, upper_year=upper_year, 
                                           time_aggr=time_aggregation)
    elif var_name == "stream":
        data_var_aggr = load_ERA5_stream_data(stream_azonal=stream_azonal,
                                              stream_extended=False,
                          lower_year=lower_year, upper_year=upper_year, 
                                           time_aggr=time_aggregation)
    elif var_name == "stream_extended":
        data_var_aggr = load_ERA5_stream_data(stream_azonal=stream_azonal,
                                              stream_extended=True,
                          lower_year=lower_year, upper_year=upper_year, 
                                           time_aggr=time_aggregation)
            
    # get covariance pattern
    data_covars = aggregated_covars_region(region, daJJA_removed_seasonality, daJJA_1979_None_trend, 
                                           lower_year=lower_year, upper_year=upper_year, 
                                           time_aggr=time_aggregation)
    #
    ## SET DATA AND SELECT JJA ONLY    
    ## Check if this line is needed or not:
    data_var_aggr = data_var_aggr.sel(time=data_var_aggr.time.dt.month.isin([6,7,8]))
    
    ## 
    indices_bool = np.where(data_covars>covar_treshold, 1, 0) #get boolean timeseries
    time_indices_treshold_excedance = data_covars.time[np.where(data_covars>covar_treshold)] #get time indices
    indices_treshold_excedance = np.where(data_covars>covar_treshold)[0] #get raw indices

    out = {}
    pvals = {}
    
    ## TO FILTER FOR DIFFERENT TIME LAGS 
    for week_lag in week_lags:
        indices_filtered_with_lag = []
        reverse_indices_with_lag = []
        for i in indices_treshold_excedance:
            #print(i, i%13)
            if i%13 >= abs(week_lag):
                #print("don't skip")
                indices_filtered_with_lag.append(i-week_lag)
                reverse_indices_with_lag.append(data_covars.time[i-week_lag].values) #get the corresponding time stamp
            else:
                #print("skip")
                continue
         
        VAR_aggregated_positives = data_var_aggr.values[indices_filtered_with_lag] #get Var of weeks where threshold is exceeded
        VAR_aggregated_positives_mean = VAR_aggregated_positives.mean(axis=0)
        
        #to get the all the data without the subset(composite)
        data_var_aggr_subset = data_var_aggr.drop(reverse_indices_with_lag, dim="time") #get the data without positive matches 
            
        
        if Sign_masked:
            ## If significance mask, to calculate e.g. pattern corrs
            t, p = _significance_t_test(VAR_aggregated_positives, data_var_aggr_subset)
            pvals[week_lag]=p
            VAR_aggregated_positives_mean = np.where(p <= 0.05, VAR_aggregated_positives_mean, np.NaN)
        
        
        out_array = np.zeros((1, data_var_aggr.shape[1], data_var_aggr.shape[2])) 
        out_array[0,:,:]=VAR_aggregated_positives_mean
        xr_out = xr.DataArray(
                data=out_array,
                dims=["time", "latitude", "longitude"],
                coords=dict(
                    time=[0],
                    longitude=data_var_aggr.longitude,
                    latitude=data_var_aggr.latitude),
                attrs=dict(
                    description=f"{var_name} composite anomalies",
                    units=var_name_units[var_name]))
        out[week_lag]=[xr_out, len(indices_filtered_with_lag)] #returning also the nr of positive matches
    
        ##
        # to plot:
        var_to_plot_name = ""
        if var_name == "stream" or var_name == "stream_extended":
            var_to_plot_name = f"{var_name}_azonal={stream_azonal}"
        plot_this_VAR_pattern(var_name, data_var_aggr_subset, 
                          VAR_aggregated_positives, VAR_aggregated_positives_mean,
                          week_lag, region, model="ERA5",
                          lower_year=lower_year, upper_year=upper_year,
                          sign=True, covar_threshold=covar_treshold, 
                          var_to_plot_name=var_to_plot_name)
    
    
    return out, pvals


#### Everything below here is for CMIP6 pattern matching on the composites 


from typing import Union


def covariance_timeseries3(pattern: Union[xr.DataArray, np.ndarray], array: Union[xr.DataArray, np.ndarray], latitude_weighting: bool = True, normalize: bool = False) -> Union[xr.DataArray, np.ndarray]:
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
        return covariance[0]
        
    return covariance



def get_sst_pattern_CMIP6(model, region, trend_pattern, 
                          covar_threshold=0.5, 
                          detrending_sst=False, lower_year = 1950, upper_year = 2014, 
                          Sign_masked=False, pvals_era={}):
    '''
    Takes a CMIP6 model, calculates the covariances of the streamfunction with the 
    ERA5 trend pattern from 1979-2023, for specified region. Takes those indices where covar_threshold is exceeded.
    Then takes those SST weeks, with different time lags (1,2,3). Creates composites, plots those. 
    Returns the composite pattern for the different timelags. 
    If Sign_masked=True, it uses the p values of the ERA5 SST pattern for the same period (lower_year, upper_year)
    to mask out the non-significant areas from ERA5.
    
    out is a dictionary with for each time lag a tuple with the sst_pattern and the number of positive matches
    '''
    
    
    ##set variables 
    trend_years = "1979-2023"
    anomalies = True
    time_aggr_covars = 'weekly'
    
    ##
    print(region, trend_years)
    print(subdomains[region])
    print(f"Anomalies is {anomalies}")
    print(model)
    
    var_to_plot = "sst"
    print(f"variable of interest is {var_to_plot}")
    time_aggr_ssts = 'weekly' 
    
    if detrending_sst == True:
        detrending_how_sst = "Global"

    elif detrending_sst == False:
        remove_seasonality_sst = True
        detrending_how_sst = "None"

    ##
    ## get STREAM250 data fro model
    filename = f"regridded_{model}_stream250_1950_2014_JJA_midlats.nc"
    data = xr.open_dataset(f"{path_data}/CMIP6/250/{filename}")["stream"]
    if model in ["CESM2-WACCM", "NorESM2-LM", "NorESM2-MM", "GFDL-CM4", "CanESM5", "CMCC-CM2-SR5", "CESM2-WACCM", "BCC-CSM2-MR",]:
        print("converting calendar")
        data = data.convert_calendar(calendar='standard')
    #remove seasonality
    data_removed_seasonality = remove_spatmean_variability(data, how=preprocessing)
    data_removed_seasonality = data_removed_seasonality.sel(time=data_removed_seasonality.time.dt.year>=lower_year)
 
    #LOAD SST DATA
    if detrending_sst:
        filename = f"detrended_anoms_GlobalTrend_weekly_sst_{model}_MJJA.nc"
        SST_model = xr.open_dataset(f"{path_data}/CMIP6/SST/{filename}")["sst_anoms"]
        data_var_aggr = SST_model.sel(time=SST_model.time.dt.month.isin([6,7,8]))
    if not detrending_sst:
        filename = f"regridded_nhplus_sst_1950_2014_{model}_MJJA.nc"
        fname = f"{path_data}/CMIP6/SST/deseasonalised_weekly_{lower_year}_{upper_year}_{filename}"
        
        if os.path.isfile(fname): #check if file already exists
            print("SST deseasonalized already exists so just loading data")
            data_var_aggr = xr.open_dataset(fname)["tos"]
        else:
            print("SSTs have to be deseasonalized still")
            SSTs = xr.open_dataset(f"{path_data}/CMIP6/SST/{filename}")["tos"]
            if model in ["CESM2-WACCM", "NorESM2-LM", "NorESM2-MM", "GFDL-CM4", "CanESM5", "CMCC-CM2-SR5", "CESM2-WACCM", "BCC-CSM2-MR"]:
                print("converting calendar")
                SSTs = SSTs.convert_calendar(calendar='standard')
            ## still deseasonalize
            SSTs = SSTs.sel(time=SSTs.time.dt.year>=lower_year)
            SSTs = SSTs.sel(time=SSTs.time.dt.year<=upper_year)#because last years are corrupted still... 
            SSTs_weekly = SSTs.resample(time="1W").mean(skipna=True)
            if remove_seasonality_sst:
                sst_weekly_ = SSTs_weekly.sel(time=SSTs_weekly.time.dt.month.isin([6,7,8])) #to just select JJA
                ## calculate weekly mean, for each gridpoint 
                SSTs_weekly_mean = SSTs_weekly.groupby(SSTs_weekly.time.dt.week).mean(dim="time")
                #to resample to just the summer
                SSTs_weekly_mean = SSTs_weekly_mean.sel(week=SSTs_weekly_mean.week.isin([22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,35]) )
                #deseasanlise by removing the weekly mean
                data_var_aggr = sst_weekly_.groupby(sst_weekly_.time.dt.week) - SSTs_weekly_mean
                data_var_aggr.to_netcdf(f"{path_data}/CMIP6/SST/deseasonalised_weekly_{lower_year}_{upper_year}_{filename}")
    


    #get values from CMIP6
    subset_region = spatial_subset(data_removed_seasonality, subdomain=region)

    #calculate anomalies per gridpoint
    if anomalies:
        subset_region_climatology = subset_region.mean(dim="time")
        subset_region_anoms = subset_region - subset_region_climatology
        subset_region = subset_region_anoms

    ## CALCULATE COVARIANCE WITH ERA5 TREND 
    if trend_years == "1979-2023":
        era5_trend = spatial_subset(trend_pattern, subdomain=region) #FROM ERA5
        subset_region_covars = covariance_timeseries2(era5_trend, subset_region, normalize=True)

    if time_aggr_covars == "weekly":
        subset_region_covars_aggregated = subset_region_covars.resample(time="1W").mean(dim="time")
        subset_region_aggregated = subset_region.resample(time="1W").mean(dim="time")
        
    data_covars = subset_region_covars_aggregated.sel(time=subset_region_covars_aggregated.time.dt.month.isin([6,7,8]))
#     data_covars = subset_region_covars_aggregated.sel(time=subset_region_covars_aggregated.time.dt.week.isin([22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,35]))
    
    
    ## plot deze covars en opslaan
    fname_figure=f"{model}_covariances_withERA5pattern_{region}_{covar_threshold}"
    if not os.path.isfile(f"/scistor/ivm/the410/WAVE5/figures/composites_CMIP6/{fname_figure}.png"): #check if plot already exists
        print("covar plot doesn't exists yet")
        data_covars.plot()
        plt.plot(data_covars.time, np.full(data_covars.shape, covar_threshold), label="threshold")
        plt.title(f"{time_aggr_covars}-mean covariances on JJA trend pattern {region} \n {model} and ERA5 \n preprocessing={preprocessing}")
        plt.tight_layout()
        plt.savefig(f"/scistor/ivm/the410/WAVE5/figures/composites_CMIP6/{fname_figure}.png")

    
    # get indices 
    indices_bool = np.where(data_covars>covar_threshold, 1, 0)[:,0] #get boolean timeseries
    time_indices_treshold_excedance = data_covars.time[indices_bool] #get time indices
    indices_treshold_excedance = np.where(data_covars>covar_threshold)[0] #get raw indices

    out = {}
    
    for week_lag in [1,2,3]:
        indices_filtered_with_lag = []
        for i in indices_treshold_excedance:
            #print(i, i%13)
            if i%13 >= abs(week_lag):
                #print("don't skip")
                indices_filtered_with_lag.append(i-week_lag)
            else:
                #print("skip")
                continue
        
#         print(data_var_aggr.time.dt.week)
#         print(data_covars.time.dt.week) #this has one year extra? why )
        
        
        VAR_aggregated_positives = data_var_aggr.values[indices_filtered_with_lag] #get Var of weeks where threshold is exceeded
        VAR_aggregated_positives_mean = VAR_aggregated_positives.mean(axis=0)
        
        ## Here create the p-val mask with ERA5 pvals, and then if Sign_masked=True, 
        #return that instead, also in ERA5         
        if Sign_masked:
            # Get pvals from era and use that for the mask 
            VAR_aggregated_positives_mean = np.where(pvals_era[week_lag] <= 0.05, VAR_aggregated_positives_mean, np.NaN)
        
        #make xarray of VAR_aggregated_positives_mean    
        out_array = np.zeros((1, 521, 1440))
        out_array[0,:,:]=VAR_aggregated_positives_mean
        xr_out = xr.DataArray(
                data=out_array,
                dims=["time", "latitude", "longitude"],
                coords=dict(
                    time=[0],
                    longitude=data_var_aggr.longitude,
                    latitude=data_var_aggr.latitude),
                attrs=dict(
                    description=f"SST composite anomalies, ERA5 sign mask={Sign_masked}",
                    units="degC"))
        out[week_lag]=[xr_out, len(indices_filtered_with_lag)] #returning also the nr of positive matches
    
#         ## to plot:
#         print("now plotting")
#         plot_this_VAR_pattern(var_to_plot, data_var_aggr, 
#                           VAR_aggregated_positives, VAR_aggregated_positives_mean,
#                           detrending_sst, detrending_how_sst, 
#                           week_lag, region, model=model,
#                           lower_year=lower_year, upper_year=upper_year,
#                           sign=True, covar_threshold=covar_threshold)
#         plot_this_VAR_pattern(var_to_plot, data_var_aggr, 
#                           VAR_aggregated_positives, VAR_aggregated_positives_mean,
#                           detrending_sst, detrending_how_sst, 
#                           week_lag, region, model=model,
#                           lower_year=lower_year, upper_year=upper_year,
#                           sign=False, covar_threshold=covar_threshold)
        
        
    return out

def get_sst_pattern_ERA5(region, data_removed_seasonality, trend_pattern,
                        covar_treshold=0.5, detrending_sst=False, 
                        lower_year = 1950, upper_year = 2014, Sign_masked=True):
    '''
    Calculates the covariances for a certain region of the streamfunction with the 
    ERA5 trend pattern from 1979-2023. Takes those indices where covar_threshold is exceeded.
    Then takes those SST weeks, with different time lags (1,2,3). Creates composites, plots those. 
    Returns the composite pattern for the different timelags. 
    If Sign_masked=True, it uses the p values
    to mask out the non-significant areas, and returns that composite AND the pvals for each timelag. 
    
    out is a dictionary with for each time lag a tuple with the sst_pattern and the number of positive matches
    pvals is a dictionary with pvals, if Sign_masked=False, will be empty
    '''
    
    
    ## set constants
    trend_years = "1979-2023"
    anomalies = True
    time_aggr_covars = 'weekly'
    
    if detrending_sst == True:
        detrending_how_sst = "Global"
#         lower_year = 1950 #because that is the timeperiod of the detrended ssts
#         upper_year = 2021

    elif detrending_sst == False:
        remove_seasonality_sst = True
        detrending_how_sst = "None"
    
    ##
    print(region, trend_years)
    print(subdomains[region])
    print(f"Anomalies is {anomalies}")

    #get values for streamfunction
    subset_region = spatial_subset(data_removed_seasonality, subdomain=region)
    #calculate anomalies per gridpoint
    if anomalies:
        subset_region_spatial_mean = subset_region.mean(axis=0)
        subset_region_anoms = subset_region - subset_region_spatial_mean
        subset_region = subset_region_anoms

    if trend_years == "1979-2023":
        subset_region_trend = spatial_subset(trend_pattern, subdomain=region)
        subset_region_covars = covariance_timeseries2(subset_region_trend, subset_region, normalize=True)

    if time_aggr_covars == "weekly":
        # To slice the years to be able to regress on SSTS 
        subset_region_covars_weekly_1950_2021 = subset_region_covars.sel(time=subset_region_covars.time.dt.year>=lower_year)
        subset_region_covars_weekly_1950_2021 = subset_region_covars_weekly_1950_2021.sel(time=subset_region_covars_weekly_1950_2021.time.dt.year<=upper_year)

        subset_region_covars_aggregated = subset_region_covars_weekly_1950_2021.resample(time="1W").mean()
        subset_region_aggregated = subset_region.resample(time="1W").mean(dim="time")
        
    #### get SST values
        
    var_to_plot = "sst"
    print(f"variable of interest is {var_to_plot}")
 
    time_aggr_ssts = 'weekly' 

    if not detrending_sst:
        if os.path.isfile(f"{path_data}/deseasonalised_weekly_SST_era5_NHExt_0.25degr_{lower_year}_{upper_year}_JJA.nc"): #check if file already exists
            print("SST deseasonalized already exists so just loading data")
            SSTs_aggregated = xr.open_dataset(f"{path_data}/deseasonalised_weekly_SST_era5_NHExt_0.25degr_{lower_year}_{upper_year}_JJA.nc")["sst"]
        else:
            SSTs = xr.open_dataset(f"{path_data}/SST_era5_NHExt_0.25degr_19400101-20240229_JJA.nc")
            SSTs = SSTs["sst"].sel(time=SSTs.time.dt.year<=upper_year) #because last years are corrupted still... 
            SSTs = SSTs.sel(time=SSTs.time.dt.year>=lower_year)
            SSTs_weekly = SSTs.resample(time="1W").mean(skipna=True)
            if remove_seasonality_sst:
                sst_weekly_ = SSTs_weekly.sel(time=SSTs_weekly.time.dt.month.isin([6,7,8])) #to just select JJA
                ## calculate weekly mean, for each gridpoint 
                SSTs_weekly_mean = SSTs_weekly.groupby(SSTs_weekly.time.dt.week).mean(dim="time")
                #to resample to just the summer
                SSTs_weekly_mean = SSTs_weekly_mean.sel(week=SSTs_weekly_mean.week.isin([22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,35]) )
                #deseasanlise by removing the weekly mean
                SSTs_aggregated = sst_weekly_.groupby(sst_weekly_.time.dt.week) - SSTs_weekly_mean
                SSTs_aggregated.to_netcdf(f"{path_data}/deseasonalised_weekly_SST_era5_NHExt_0.25degr_{lower_year}_{upper_year}_JJA.nc")


    if detrending_sst:
        print("detrending is True")
        print(detrending_how_sst)
        if detrending_how_sst == "Global":
            '''lowess fit is done on global mean SST values, '''
            #SSTs_weekly = xr.load_dataset("/scistor/ivm/the410/WAVE5/detrended_GlobalTrend_weekly_ssts_nhplus_june2sept.nc")
            '''then anomalies are calculated gridpoint wise'''
            SSTs_aggregated = xr.load_dataset("/scistor/ivm/the410/WAVE5/detrended_GlobalTrend_weekly_ssts_anoms_nhplus_june2sept.nc").astype("float32")
            
    if var_to_plot == "sst":
        time_aggr_var = time_aggr_ssts
#         data_var = SSTs_aggregated
        detrending_var = detrending_sst
        detrending_how_var = detrending_how_sst

    ###
    if time_aggr_covars == "weekly" and time_aggr_var == 'weekly':
        time_aggr = 'weekly'
    else:
        print("time aggregations are not alligned")


#     data_covars = subset_region_covars_aggregated

    ## SET DATA AND SELECT JJA ONLY     
    data_covars = subset_region_covars_aggregated.sel(time=subset_region_covars_aggregated.time.dt.month.isin([6,7,8]))
    data_var_aggr = SSTs_aggregated.sel(time=SSTs_aggregated.time.dt.month.isin([6,7,8]))
    
    ## 
    indices_bool = np.where(data_covars>covar_treshold, 1, 0) #get boolean timeseries
    time_indices_treshold_excedance = data_covars.time[np.where(data_covars>covar_treshold)] #get time indices
    indices_treshold_excedance = np.where(data_covars>covar_treshold)[0] #get raw indices

    out = {}
    pvals = {}
    
    ## TO FILTER FOR DIFFERENT TIME LAGS 
    for week_lag in [1,2,3]:
        indices_filtered_with_lag = []
        reverse_indices_with_lag = []
        for i in indices_treshold_excedance:
            #print(i, i%13)
            if i%13 >= abs(week_lag):
                #print("don't skip")
                indices_filtered_with_lag.append(i-week_lag)
                reverse_indices_with_lag.append(data_covars.time[i-week_lag].values) #get the corresponding time stamp
            else:
                #print("skip")
                continue
        if detrending_sst:
            VAR_aggregated_positives = data_var_aggr.to_array().values[0][indices_filtered_with_lag] #get Var of weeks where threshold is exceeded
        else:
            VAR_aggregated_positives = data_var_aggr.values[indices_filtered_with_lag] #get Var of weeks where threshold is exceeded
        VAR_aggregated_positives_mean = VAR_aggregated_positives.mean(axis=0)
        #make xarray of VAR_aggregated_positives_mean
        
        #to get the all the data without the subset(composite)
        data_var_aggr_subset = data_var_aggr.drop(reverse_indices_with_lag, dim="time") #get the data without positive matches 
       
        
        if Sign_masked:
            ## get pvals for this timelag 
            t, p = _significance_t_test(VAR_aggregated_positives, data_var_aggr_subset)
            pvals[week_lag]=p
            VAR_aggregated_positives_mean = np.where(p <= 0.05, VAR_aggregated_positives_mean, np.NaN)
        
        out_array = np.zeros((1, 521, 1440))
        out_array[0,:,:]=VAR_aggregated_positives_mean
        xr_out = xr.DataArray(
                data=out_array,
                dims=["time", "latitude", "longitude"],
                coords=dict(
                    time=[0],
                    longitude=data_var_aggr.longitude,
                    latitude=data_var_aggr.latitude),
                attrs=dict(
                    description="SST composite anomalies",
                    units="degC"))
        out[week_lag]=[xr_out, len(indices_filtered_with_lag)] #returning also the nr of positive matches
    
        ##
#         # to plot: #--> see other notebook
#         plot_this_VAR_pattern(var_to_plot, data_var_aggr, 
#                           VAR_aggregated_positives, VAR_aggregated_positives_mean,
#                           detrending_var, detrending_how_var, 
#                           week_lag, region, model="ERA5",
#                           lower_year=lower_year, upper_year=upper_year,
#                           sign=True, covar_threshold=covar_treshold)
#         plot_this_VAR_pattern(var_to_plot, data_var_aggr, 
#                           VAR_aggregated_positives, VAR_aggregated_positives_mean,
#                           detrending_var, detrending_how_var, 
#                           week_lag, region, model="ERA5",
#                           lower_year=lower_year, upper_year=upper_year,
#                           sign=False, covar_threshold=covar_treshold)
    
    return out, pvals


models = ['MIROC6',
 'MPI-ESM-1-2-HAM',
 'MPI-ESM1-2-HR',
 'MPI-ESM1-2-LR',
 'MRI-ESM2-0',
 'NorESM2-LM',
 'NorESM2-MM',
 'GFDL-CM4',
 'EC-Earth3-Veg-LR',
 'EC-Earth3-Veg',
 'EC-Earth3',
 'CanESM5',
 'CNRM-ESM2-1',
 'CNRM-CM6-1',
 'CMCC-CM2-SR5',
 'CESM2-WACCM',
 'BCC-CSM2-MR',
 'AWI-ESM-1-1-LR',
 'ACCESS-CM2']

def get_covariances(region, threshold_cov, models, data_removed_seasonality, trend_pattern,
                    lat_weight=True, Sign_masked=False, mask_above_80=True):
    

        
    ##
    counts = []
    sst_patterns_era5, pvals_era = get_sst_pattern_ERA5(region, data_removed_seasonality, trend_pattern,
                                                 covar_treshold=threshold_cov,
                                                 Sign_masked=Sign_masked)# if Sign_masked=False, pvals_era is empty        
    
    if mask_above_80:
        for week in [1,2,3]:
            pvals_era[week][np.where(sst_patterns_era5[week][0].latitude>80), :] = 1 #set pvals above lat=80 to 1 
    
    count=["ERA5"]
    for week_lag in [1,2,3]:
        era5 =  sst_patterns_era5[week_lag][1] #the first is the data, the second the nr of postiive matches
        count.append(era5)    
    counts.append(count)
    
    model_ssts = {}
    for model in models: 
        if model == "CanESM5" or model == "AWI-ESM-1-1-LR":
            print("Model CanESM5/AWI-ESM-1-1-LR do momentarily not have the year 1950??")
            lower_year = 1951
        else:
            lower_year = 1950
        ## here load the data:
        model_ssts[model]= get_sst_pattern_CMIP6(model, region,trend_pattern,
                                                 lower_year=lower_year, covar_threshold=threshold_cov,
                                                 Sign_masked=Sign_masked, pvals_era=pvals_era)
    
    covariances = []

    for model in models:
        covs = [model]
        count = [model]
        cmip6 = np.zeros((1, 521, 1440))
        for week_lag in [1,2,3]:
            era5 =  sst_patterns_era5[week_lag][0] #the first is the data, the second the nr of postiive matches
            #cmip6[0,:,:] = model_ssts[model][week_lag][0]
            cmip6 = model_ssts[model][week_lag][0]
            covs_week_model = covariance_timeseries3(era5, cmip6, normalize=True, latitude_weighting=lat_weight)
            #print(covs_week_model) #for this week and this model 
            covs.append(covs_week_model[0])
            count.append(model_ssts[model][week_lag][1])
        covariances.append(covs)
        counts.append(count)

    region_dataframe = pd.DataFrame(covariances, columns=["model_name", "week1", "week2", "week3"])
    region_dataframe_counts = pd.DataFrame(counts, columns=["model_name", "week1_n", "week2_n", "week3_n"])

    return region_dataframe, region_dataframe_counts


def get_stream_trend_pattern_CMIP6_plus_covar(model, ERA5TREND,
                          lower_year = 1950, upper_year = 2014, 
                          get_covars=True,
                          Sign_masked=False, pvals_era={},
                           preprocessing="seasonality_only"):
    '''
    Takes a CMIP6 model, gets trend pattern of streamfunction 
    calculates the covariances of the this trend witht the 
    ERA5 trend pattern from 1979-2023, for specified region. 
    
    If Sign_masked=True, it uses the p values of the ERA5 pattern for the same period (lower_year, upper_year)
    to mask out the non-significant areas from ERA5, to calculate weighted covariancec
    
    out are the trend as xarray and covariance with ERA5 float
    '''
    
    
    ##set variables 
    trend_years = f"{lower_year}-{upper_year}"
    anomalies = True
    time_aggr_covars = 'weekly'
    
    ##
    print(trend_years)
    print(model)
    
    var_to_plot = "stream"
    print(f"variable of interest is {var_to_plot}")
    
    ##
    ## get STREAM250 data fro model
    filename = f"regridded_{model}_stream250_1950_2014_JJA_midlats.nc"
    data = xr.open_dataset(f"{path_data}/CMIP6/250/{filename}")["stream"]
    if model in ["CESM2-WACCM", "NorESM2-LM", "NorESM2-MM", "GFDL-CM4", "CanESM5", 
                 "CMCC-CM2-SR5", "CESM2-WACCM", "BCC-CSM2-MR", "CMCC-CM2-HR4", 
                 "CMCC-ESM2", 'HadGEM3-GC31-LL', "HadGEM3-GC31-MM", "INM-CM4-8", 
                 "INM-CM5-0", "IPSL-CM5A2-INCA", "TaiESM1", "UKESM1-0-LL"]:
        print("converting calendar")
        if model in ['HadGEM3-GC31-LL', "HadGEM3-GC31-MM", "UKESM1-0-LL"]:
            data = data.convert_calendar(calendar='standard', align_on="year")
        else:
            data = data.convert_calendar(calendar='standard')
    #remove seasonality
    data_removed_seasonality = remove_spatmean_variability(data, how=preprocessing)
    data_removed_seasonality = data_removed_seasonality.sel(time=data_removed_seasonality.time.dt.year>=lower_year)
    data_removed_seasonality = data_removed_seasonality.sel(time=data_removed_seasonality.time.dt.year<=upper_year)
    data_removed_seasonality.attrs["units"]="m2/s"
    ## GET TREND
    CMIP6_trend = compute_trend_pattern(data_removed_seasonality)

    ## plot deze trend en opslaan
    
    fname_figure=f"{model}_stream250_trendpattern_prp={preprocessing}_{lower_year}_{upper_year}"
    if not os.path.isfile(f"/scistor/ivm/the410/WAVE5/figures/stream_trends_CMIP6/{fname_figure}.png"): #check if plot already exists
        print("trend plot doesn't exists yet")

        ######
        shading = 'flat'
        vmin, vmax = -200000, 200000
        cmap = 'RdBu_r'
        fig, ax = plt.subplots(figsize = (20,8), subplot_kw={'projection':ccrs.PlateCarree()}) 
        im = ax.pcolormesh(CMIP6_trend.longitude, CMIP6_trend.latitude, CMIP6_trend.values[0, :-1,:-1], transform=ccrs.PlateCarree(),
                                shading = shading, cmap = cmap, vmin = vmin, vmax = vmax)

        ax.set_title(f'{model} JJA stream250 trend \n preprocessing={preprocessing} {trend_years}')
        ax.gridlines(draw_labels = ['left','bottom'])
        ax.coastlines()
        fig.colorbar(im, ax=ax, fraction=0.005, pad=0.02, label=f"m2/s/y")
        
        plt.tight_layout()
        plt.savefig(f"/scistor/ivm/the410/WAVE5/figures/stream_trends_CMIP6/{fname_figure}.png")

    if not get_covars:
        return CMIP6_trend
    
    if get_covars:
        if Sign_masked:
            ## get pvals of ERA5 for masking BOTH datasets
            print("trendpatterns are masked with era5 pvals")
            CMIP6_trend = np.where(pvals_era <= 0.05, CMIP6_trend, np.NaN)
            ERA5TREND = np.where(pvals_era <= 0.05, ERA5TREND, np.NaN)

        COVARS = {}
        for region in ["circumglobal", "Both", "US_Atl", "EURASIA"]: 
            subset_era_trend = spatial_subset(ERA5TREND, subdomain=region)
            subset_cmip_trend = spatial_subset(CMIP6_trend, subdomain=region)
            covar = covariance_timeseries3(subset_era_trend, subset_cmip_trend, normalize=True, latitude_weighting=True)
            #print(covar)
            COVARS[region] = covar

        return CMIP6_trend, COVARS
