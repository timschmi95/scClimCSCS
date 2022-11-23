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
username = os.getlogin()
sys.path.append(f'/users/{username}/scClimCSCS')
from get_archived_data import get_combined_max_radar_grid,get_netcdf

years = [2016]
start_month = 4 #4 (convective season April(4) to Sept(9))
end_month = 4 #9
end_day = 2 #30
varname = 'CZC'
out_dir = '/scratch/tschmid/data/nc_files'
for year in years:
    start_date = dt.datetime(year=year,month=start_month,day=1)
    end_date = dt.datetime(year=year,month=end_month,day=end_day)
    day_dt = start_date
    while (day_dt<=end_date):
        #TODO: fill missing days with nan and give a warning
        #get netcdf
        ds = get_netcdf(varname,day_dt)
        if day_dt == start_date:
            ds_out = ds
        else:
            ds_out = xr.concat([ds_out,ds],dim='time') 
        print(day_dt) 
        day_dt += dt.timedelta(days=1)
        
    #save netcdf (yearly files)
    ds_out.to_netcdf(f'{out_dir}/{varname}_6t6_{year}.nc', encoding={varname:{'zlib':True,'complevel':9}})
    print(f'Saved {day_dt} under {out_dir}')





