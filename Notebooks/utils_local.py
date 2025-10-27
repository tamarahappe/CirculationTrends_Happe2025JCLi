import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from netCDF4 import Dataset,date2num, num2date
import os
import netCDF4
import xarray as xr
import cartopy.crs as ccrs
import scipy

def plot_composites(data, lons, lats, title:str, deseasonalized=False, unit="m2/s"):
    import matplotlib as mpl
    cmap = plt.get_cmap('RdBu_r')

    fig, ax = plt.subplots(1,1, figsize=(50,30))
    ax1 = plt.subplot(111, projection=ccrs.PlateCarree())
    
    if deseasonalized: #to adjust the colorbar 
        cs = ax1.pcolormesh(lons, lats, data,
                     transform=ccrs.PlateCarree(), cmap=cmap, norm=mpl.colors.CenteredNorm(), edgecolors = 'None')
    if not deseasonalized:
        cs = ax1.pcolormesh(lons, lats, data,
                         transform=ccrs.PlateCarree(), cmap=cmap, vmin=-40000000, vmax=0, edgecolors = 'None')

    ax1.coastlines()
    ax1.set_title(f"{title}", fontsize=35)
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False

    fig.colorbar(cs, ax=ax1, fraction=0.005, pad=0.02, label=unit)

    plt.show()
    
    
#input variables are bounds, rank_anoms, and rank

def bounding_ranks(ranks:xr.DataArray, rank_anoms:xr.DataArray,bounds:list=[23, 70]):
    '''
    Splits the rank anomalies up into lower, middle, and upper bounds, as given by the thresholds. Bound list
    is flexible.
    INPUTS: bounds = list of int. Have to be between 0 and max(rank), act as thresholds. 
    E.g. 1 bound will split the data into two with (0, bound) and (bound, max)
    ranks = xr.DataArray with days in season ranked (time, lats, lons)
    rank_anoms = xr.DataArray with rank_anoms per day in season (time, lats, lons)
    
    OUTPUT= list of xr.DataArray, with the rank anomalies split for the given bounds
    '''
    #set local variables:
    outs = [] #maybe later this can be a dict? if we want to organize more

    bin_nr = len(bounds) + 1 #count nr of bounds + 1, because the bounds will split the data into nr of bins
    bounds.append(int(np.max(ranks[:,0,0]))+1) #count nr of summerdays/ranks, append to get upper limit
    lower = 1

    #Get info from data Array
    nryears = np.unique(ranks.time.dt.year).shape[0]
    lons=ranks.longitude
    lats=ranks.latitude
    attrs = rank_anoms.attrs
    
    print("starting bin loop")

    #iterate over the bins, to get the slices
    for i in range(bin_nr):
        upper = bounds[i]                         #Set upper limit for this bin
        rank_dim = upper - lower                  #get the nr of ranks in this bin
        attrs["bin bounds"] = f"{lower}-{upper}" #put info in attr dict

        print(f"lower:{lower}, upper:{upper}, rank_dim:{rank_dim}")

        values = np.zeros((lats.shape[0], lons.shape[0], nryears, rank_dim)) #create empty array to store info
        #for each gridpoint, slice the data for this bin
        for lon in range(lons.shape[0]):        
            for lat in range(lats.shape[0]):
                # get the indices of the ranks in this bin
                indices_bin = (ranks[:, lat, lon]>=lower) & (ranks[:, lat, lon]<upper)
                #get the associated rank_anom values and reshape to years, bin dimensions
                values[lat, lon, :, :] = rank_anoms[indices_bin, lat, lon].to_numpy().reshape(nryears, rank_dim)
        
        #create out DataArray with values 
        out_bound = xr.DataArray(
            data=values, dims=["latitude", "longitude", "time", "rank_anom"], 
                 coords=dict(latitude=lats, longitude=lons, time=np.unique(ranks.time.dt.year)), 
                 attrs=attrs)
        outs.append(out_bound) #append this DataArray to outs
        
        lower = bounds[i] #update lower bound, to move along the bounds
        
    return outs


def get_gridpoint_trends(list_of_arrays):
    from scipy.stats import linregress

    lats = list_of_arrays[0].latitude
    lons = list_of_arrays[0].longitude
    
    #create empty arrays to store trend and pvalues 
    trends = np.zeros((lats.shape[0], lons.shape[0], len(list_of_arrays)))
    pvals = np.zeros((lats.shape[0], lons.shape[0], len(list_of_arrays)))

    #for each gridpoint, fit linregression - to test just take 5*5 grid
    for lon in range(lons.shape[0]):        
        for lat in range(lats.shape[0]):
            # get the indices of the ranks in this bin
            for i, out_array in enumerate(list_of_arrays):
                reg_result = linregress(x = list_of_arrays[i].time, y = np.nanmean(list_of_arrays[i].values[lat,lon], axis=1))
                trends[lat, lon, i], pvals[lat, lon, i] = reg_result[0], reg_result[3] #assign trend and pval to corresponding place in array
                
                
    return trends, pvals

from cartopy.mpl.geoaxes import GeoAxes


# Register GeoAxes as a projection
GeoAxes._pcolormesh_patched = False

def plot_trends(data, trends, pvals, lons, lats, 
                vmin=-80000, vmax=80000, unit="m2/s", var="STREAM250_midlat_remove_None_1950_onward",
               var_title="STREAM250 preprocessing=None 1950-2023",
               save_fig=False):
    
    #data is trends
    
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import cartopy.crs as ccrs
    from cartopy.mpl.geoaxes import GeoAxes
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    
    cmap = plt.get_cmap('RdBu_r')
    
    fig, axs = plt.subplots(len(data),1, figsize=(20,8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    axs = axs.flatten()
    
    #if shading = flat
    lats = np.concatenate([lats[[0]] - np.diff(lats)[0], lats], axis = 0) # Adding the sourthern edge 
    lons = np.concatenate([lons, lons[[-1]] + np.diff(lons)[0]], axis = 0)# Adding the eastern edge (only for flat shating)
   
    #vmin, vmax = -80000, 80000
    
    for i in range(len(data)):
        #cs = axs[i].pcolormesh(lons, lats, trends[:,:,i], shading = "flat", 
         #                     transform=ccrs.PlateCarree(), cmap=cmap, norm=mpl.colors.CenteredNorm(), edgecolors = 'None')
        cs = axs[i].pcolormesh(lons, lats, trends[:,:,i], shading = "flat", 
                              transform=ccrs.PlateCarree(), cmap=cmap,  vmin=vmin, vmax=vmax, edgecolors = 'None')

        
        axs[i].coastlines()
        #name=data[i].attrs["long_name"]
        bounds=data[i].attrs["bin bounds"]
        if i == 0:
            axs[i].set_title(f"{var_title} trend for \n \n bin bounds {bounds}", 
                             fontsize=18)
        else:
            axs[i].set_title(f"bin bounds {bounds}", 
                             fontsize=18)
        axs[i].set_xlabel("Longitude")
        axs[i].set_ylabel("Latitude")

        gl = axs[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=2, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        fig.colorbar(cs, ax=axs[i], fraction=0.005, pad=0.03, label=f"{unit}/y")
        
    if save_fig:
        plt.savefig("/scistor/ivm/the410/GolfVijf/figures_revised/Fig4_HQ.png", dpi=300)

    fig.tight_layout()
    plt.show()
    
