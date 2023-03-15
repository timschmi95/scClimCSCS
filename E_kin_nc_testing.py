"""
script to calculated monthly netcdf files of daily radar variables 
from 6UTC to 6 UTC

author: Timo Schmid
"""

import sys
import os
import numpy as np
import datetime as dt 
import xarray as xr 
username = 'tschmid' #os.getlogin()
sys.path.append(f'/users/{username}/scClimCSCS')
from get_archived_data import get_combined_max_radar_grid,get_netcdf
import matplotlib.pyplot as plt


varname = 'E_kin'
out_dir = '/scratch/tschmid/data/nc_files'


day_dt = dt.datetime(2021,6,28)
ds = get_netcdf(varname,day_dt)
        
fig,ax=plt.subplots()
ds.E_kin.plot(ax=ax)
fig.savefig('/users/tschmid/test6.png')
        
    # #save netcdf (yearly files)
    # ds_out.to_netcdf(f'{out_dir}/{varname}_6t6_{year}.nc', encoding={varname:{'zlib':True,'complevel':9}})
    # print(f'Saved {day_dt} under {out_dir}')





