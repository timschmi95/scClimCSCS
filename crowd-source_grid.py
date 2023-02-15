import numpy as np
import sys 
import xarray as xr
from pathlib import Path
import datetime as dt
import geopandas as gpd
from pyproj import CRS
from scipy import sparse
import pandas as pd
import copy
import warnings
from pyproj import Proj
import cartopy.crs as ccrs





def grid_crowd_source(crowd_source_data,pop_path,date_sel='20210628',kernel_size_gap_fill=7,min_crowd_count=100):
    """function to grid crowd-sourced data

    Args:
        crowd_source_data (Path,str,pd.DataFrame): dataframe of crowd-source reports
        pop_path (Path,str): path for population data source
        date_sel (str, optional): 'all' for all dates, otherwise '%Y%m%d'. Defaults to '20210628'.
        kernel_size_gap_fill (int, optional): size of kernel (must be odd). Defaults to 7.
        min_crowd_count (int, optional): min count of crowd-source reports. Only used if date_sel='all'. Defaults to 100.

    Returns:
        ds: xr.Dataset of gridded crowd-sourced data
    """

    assert(kernel_size_gap_fill%2 == 1), 'kernel_size_gap_fill must be an odd number'

    if isinstance(crowd_source_data,(str,Path)):
        path = crowd_source_data
        crowd = pd.read_csv(path,sep=',')
    elif isinstance(crowd_source_data,pd.DataFrame):
        crowd = crowd_source_data
    
    #filter out filtered_out points
    crowd = crowd.loc[crowd['FILTEREDOUT'] == 0,:]

    #population pre-processing
    population = pd.read_csv(pop_path,sep=';')
    population['chx'] = np.round(population['E_KOORD']/1000)
    population['chy'] = np.round(population['N_KOORD']/1000)
    pop_sum = population.groupby(['chx','chy']).agg({'B21BTOT': 'sum'})
    chx_range = [2484, 2838]
    chy_range = [1073, 1299]

    chx = np.arange(chx_range[0],chx_range[1])
    chy = np.arange(chy_range[0],chy_range[1])

    pop = np.zeros((len(chy),len(chx)))
    for i in range(len(pop_sum)): #tqdm()
        x_coord = chx==pop_sum.index.get_level_values('chx')[i]
        y_coord = chy==pop_sum.index.get_level_values('chy')[i]
        pop[y_coord, x_coord] = pop_sum.iloc[i]['B21BTOT']


    #crowd source pre processing
    crowd['Time'] = pd.to_datetime(crowd['Time'], format='%Y-%m-%d %H:%M:%S')
    # round to nearest minute
    crowd['Time'] = crowd['Time'].dt.round('min')
    # # select times on 2021-06-28
    # crowd = crowd.loc[crowd['Time'].dt.date == pd.to_datetime('2021-06-28').date()]
    # select times in range (1 hailday: 6UTC-6UTC)

    #create copy of crowd dataframe with all values
    crowd_all = crowd.copy(deep=True)

    if date_sel == 'all':
        crowd['hail_day'] = (crowd['Time'] -pd.Timedelta('6h')).dt.date #use -6h, to get haildays correctly! (1 hailday: 6UTC-6UTC)
        grpy = crowd.groupby('hail_day').ID.count()
        sel_dates = grpy.index[grpy>min_crowd_count] #min *min_crowd_count* reports per day (CH wide)
        dates = [d.strftime('%Y%m%d') for d in sel_dates if d.year >=2017] #only select dates after 2017
        # print(dates)
        # return dates
        # raise ValueError('not implemented yet')
    else:
        dates = [date_sel]

    for date in dates:
        print(date)
        date_plus_one = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime('%Y%m%d')
        datelist_range = [f'{date}0600',f'{date_plus_one}0600']
        crowd = crowd_all.loc[(crowd_all['Time'] >= datelist_range[0]) & (crowd_all['Time'] <= datelist_range[1])]

        # convert crowd size classes to mm todo: check this & find more robust way
        crowd['size_mm'] = np.nan
        crowd['size_mm'][crowd['size']==(6+10)] = 68
        crowd['size_mm'][crowd['size']==(5+10)] = 43
        crowd['size_mm'][crowd['size']==(4+10)] = 32
        crowd['size_mm'][crowd['size']==(3+10)] = 23
        crowd['size_mm'][crowd['size']==(2+10)] = 8
        crowd['size_mm'][crowd['size']==(1+10)] = 5
        crowd['size_mm'][crowd['size_mm']==0] = 0

        # round to nearest km (for later groupby operation)
        crowd['chx'] = np.round((crowd['x'])/1000-600)
        crowd['chy'] = np.round((crowd['y'])/1000-200)

        # group by chx, chy to speed up array loop
        crowd_mean = crowd.groupby(['chx','chy']).agg({'size_mm': 'mean'})
        crowd_count = crowd.groupby(['chx','chy']).agg({'size_mm': 'count'})    


        crowd_sizes = np.zeros((len(chy),len(chx)))
        crowd_sizes[:] = np.nan
        crowd_counts = np.zeros((len(chy),len(chx)))

        # assign grouped point values to array
        for i in range(len(crowd_mean)):
            x_coord = chx-2600==crowd_mean.index.get_level_values('chx')[i]
            y_coord = chy-1200==crowd_mean.index.get_level_values('chy')[i]
            crowd_sizes[y_coord, x_coord] = crowd_mean.iloc[i]['size_mm']
            crowd_counts[y_coord, x_coord] = crowd_count.iloc[i]['size_mm']

        # reporting_fraction = crowd_counts/pop
        # reporting_fraction = np.nan_to_num(reporting_fraction, nan=0.0, posinf=1)
        # reporting_fraction[reporting_fraction<=0] = np.nan
        # print(reporting_fraction.shape)

        crowd_estimate = {}
        crowd_estimate['size'] = crowd_sizes.copy()
        crowd_estimate['count'] = crowd_counts.copy()
        crowd_estimate['chx'] = chx
        crowd_estimate['chy'] = chy

        # don't trust gridpoints with only 1 report, big population and big hail
        # crowd_estimate['size'][(crowd_estimate['count']==1) & (pop>500) & (crowd_estimate['size']>30)] = np.nan

        # gridpoints with high enough population and no reports are assumed to have no hail
        crowd_estimate['size'][(pop>2000) & (np.isnan(crowd_sizes))] = 0

        crowd_estimate['size_interpolated'] = crowd_estimate['size'].copy()

        # kernel marching for filling in the gaps todo: convert to disk kernel with gaussian weighting
        kernel_size = kernel_size_gap_fill # odd number
        kernel_size = kernel_size//2 # kernel is implemented as indexing with +- kernel_size
        for i in range(crowd_estimate['size'].shape[0]): # loop over x
            for j in range(crowd_estimate['size'].shape[1]): # loop over y
                if np.isnan(crowd_estimate['size'][i,j]): # only fill gridpoints with nan
                    if np.sum(crowd_estimate['size'][i-kernel_size:i+kernel_size,j-kernel_size:j+kernel_size]>=0) > 3: # only fill if there are at least 4 valid gridpoints whitinin the kernel
                        crowd_estimate['size_interpolated'][i,j] = np.nanmean(crowd_estimate['size'][i-kernel_size:i+kernel_size,j-kernel_size:j+kernel_size]) # fill with mean of kernel

        # filter out high frequency noise with small kernel
        noisy = crowd_estimate['size_interpolated'].copy()
        denoised = np.zeros((len(chy),len(chx)))
        denoised[:] = np.nan

        kernel_size = 3//2
        for i in range(denoised.shape[0]):  # loop over x
            for j in range(denoised.shape[1]):  # loop over y
                if np.sum(noisy[i-kernel_size:i+kernel_size,j-kernel_size:j+kernel_size]>=0) > 1: # only alter gridpoints with at least 2 valid gridpoints whitinin the kernel
                    denoised[i,j] = np.nanmean(noisy[i-kernel_size:i+kernel_size,j-kernel_size:j+kernel_size]) # fill with mean of kernel
            
        crowd_estimate['denoised'] = denoised.copy()

        #convert to xr.DataSet
        ds = xr.Dataset(data_vars={'h_raw': (['chy', 'chx'], crowd_estimate['size']),
                            'h_grid': (['chy', 'chx'], crowd_estimate['size_interpolated']),
                            'h_smooth': (['chy', 'chx'], crowd_estimate['denoised']),
                            'n_rep': (['chy', 'chx'], crowd_estimate['count'])},
                    coords={'chx': (['chx'], crowd_estimate['chx']*1000),'chy': (['chy'], crowd_estimate['chy']*1000)})

        ds = ds.expand_dims({'time': [dt.datetime.strptime(date, '%Y%m%d')]})
        if date == dates[0]:
            ds_all = ds.copy(deep=True)
        else:
            ds_all = xr.concat([ds_all, ds], dim='time')
            
    ##get lat lon (testing)
    projdef = ccrs.epsg(2056)#.proj4_init
    projdef
    #create meshgrid
    meshX,meshY = np.meshgrid(ds_all.chx, ds_all.chy)
    p = Proj(projdef)
    lon, lat = p(meshX,meshY, inverse=True)
    ds_all=ds_all.assign_coords({'lon':(('chy','chx'),lon)})
    ds_all=ds_all.assign_coords({'lat':(('chy','chx'),lat)})
    
    return ds_all