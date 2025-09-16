import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from netCDF4 import Dataset,date2num, num2date
import os
import netCDF4
import cartopy.crs as ccrs
from datetime import date


path_data='/scistor/ivm/the410/WAVE5'

def from_txt_to_dates(data):
    """
    takes in list of floats, with every float consisting of yyyymmdd
    returns array with date in Date format  
    """
    out_array = []
    for i, day in enumerate(data):
        out_array.append(date(int(str(int(day))[:4]), int(str(int(day))[4:6]), int(str(int(day))[6:])))
    return np.array(out_array, dtype=date)

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

def DDdata_txt_to_timeseries(filename):
    """Takes in the filename of DDdata.txt from Fabio, returns  xarray with DDday timeseries
    WARNING: DDday data and dates are from 1950-2022"""
    #read in ddday data, convert to indices
    txt = pd.read_csv(f'{path_data}/{filename}', header = None)[0].to_list()
    indices = from_txt_to_indice_JJA(txt)
    #read in timestamps
    times = Dataset(fr"{path_data}/old_era5/1950_2022_stream_250hPa_lonlat_midlats_JJA.nc")["time"]
    dates = netCDF4.num2date(times, units = times.units, calendar=times.calendar)
    
    #convert
    np_array = np.zeros_like(dates)
    np_array[indices]=1 #DDdays get value 1
    xr_array = xr.DataArray(data=np_array, dims=["time"], 
             coords=dict(time=dates), 
             attrs=dict(description=f"DDdays {filename[-8:-4]} binary", units="1 for DDday, 0 if not"))
    return xr_array

def create_event_length_per_year(DD_series_xarray):
    ## I want for each year a list with event lengths
    ##DD_series_xarray is an xarray with 0 and 1s, corresponding with whether that day is a DDday or not. (SUMMER ONLY)
    ## DD_series_xarray is the output from DDdata_txt_to_timeseries()
    
    ##returns:
    ##year_lenghts: for each year, a list with the event lenghts per event. len(year_lenghts[year])=nr of events that year
    ## year_indices:for each year, a list with the indice of the indice of the day of that summer, for each ddday 

    diff = np.diff(DD_series_xarray)
    year_lengths = []
    year_indices = []

    x = array.time.dt.year
    yearsn = np.unique(np.unique(array.time.dt.year))

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