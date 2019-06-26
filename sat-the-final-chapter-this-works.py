# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 10:21:20 2019
@author: tjtur
"""

def ltg_plot(highlow,ltg):
    """
    Plots lightning and assigns +/ based on polarity and color codes based
    on ground stikes versus height of intercloud flash
    
    Parameters
    ----------
    highlow : string to say whether we're plotting low or high
        ltg : pandas dataframe containing strike data
        
    Returns
    -------
    Nothing, just makes a scatterplot then exits
                    
    """    
    for st in range(0,len(ltg)):
        lat = ltg.latitude.values[st]
        lon = ltg.longitude.values[st]
        cur = ltg.peakcurrent[st]
        hgt = ltg.icheight[st]    
        size_add = 0
        if hgt == 0:
            col = 'r'
            size_add = 10
            zord = 10
        elif hgt < 10000:
            col = 'm'
            size_add = 5
            zord = 5
        elif hgt < 15000:
            col = 'c'
            zord = 3            
        elif hgt < 20000:
            col = 'b'
            zord = 2 
        else:
            col = 'g'
            zord = 1 
        if cur > 0:
            symb = '+'
        else:
            symb = '_'
        size = 10 + size_add
        if highlow == 'low' and hgt == 0:    
            a.scatter(lon,lat,s=size,marker=symb,c=col,zorder=zord)
            a.set_title('EN Cloud to Ground')
        elif highlow == 'high' and hgt > 0:
            a.scatter(lon,lat,s=size,marker=symb,c=col,zorder=zord)
            a.set_title('EN Intracloud')
    return




import sys
import os

try:
    os.listdir('/var/www')
    windows = False
    sys.path.append('/data/scripts/resources')
    image_dir = os.path.join('/var/www/html/radar','images')
except:
    windows = True
    sys.path.append('C:/data/scripts/resources')
    base_dir = 'C:/data'
    base_gis_dir = 'C:/data/GIS'



from case_data import this_case
event_date = this_case['date']
rda = this_case['rda']
extent = this_case['sat_extent']
shapelist = this_case['shapelist']
case_dir = os.path.join(base_dir,event_date)
radar_dir = os.path.join(case_dir,rda,'netcdf/ReflectivityQC/00.50')
sat_dir = os.path.join(case_dir,'satellite/raw')
ltg_dir = os.path.join(case_dir,'lightning')
image_dir = os.path.join(base_dir,'images',event_date,'satellite')

from my_functions import latlon_from_radar, figure_timestamp, build_html
from custom_cmaps import plts
from gis_layers import shape_mini
import numpy as np
from pyproj import Proj
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import re
import os
import pandas as pd
from datetime import datetime

ltg_D = []
# lightning files obtained from EarthNetworks
ltg_files = os.listdir(ltg_dir)
ltg_csv = os.path.join(ltg_dir,ltg_files[0])
ltg_D = pd.read_csv(ltg_csv, index_col=['time'])
ltg_D.index = [datetime.strptime(x[:-2], '%Y-%m-%dT%H:%M:%S.%f') for x in ltg_D.index]

# Here is a step where we define to bin plots by time
# ----------------------------------------------------------------
# ----------------------------------------------------------------
idx = pd.date_range('2019-03-14 23:29', periods=2, freq='1Min')
dt = idx[1] - idx[0]
# ----------------------------------------------------------------
# ----------------------------------------------------------------

# building a met_info list containing filepaths for all
# radar and satellite (including GLM) products and their associated datetime
# derived from filename convention
met_info = []

radar_files = os.listdir(radar_dir)
#radar_files = os.listdir(radar_stage_dir)
for r in (radar_files):
    rad_info = str.split(r,'.')
    rad_time_str = rad_info[0]
    rad_datetime = datetime.strptime(rad_time_str,"%Y%m%d-%H%M%S")
    #info = [rad_datetime,'r',os.path.join(radar_stage_dir,r)]
    info = [rad_datetime,'r',os.path.join(radar_dir,r)]
    met_info.append(info)

satellite_files = os.listdir(sat_dir)
for s in (satellite_files):
    sat_split = str.split(s,'_')
    dtype = sat_split[1][:3]
    vis_test = sat_split[-5][-3:]
    if dtype == 'GLM':
        g16_dtype = 'g'
    elif vis_test == 'C02':
        g16_dtype = 'vis'
    else:
        g16_dtype = 's'

    sat_time_s = sat_split[3]
    sat_time = sat_time_s[1:-1]
    sat_datetime = datetime.strptime(sat_time, "%Y%j%H%M%S")
    info = [sat_datetime,g16_dtype,os.path.join(sat_dir,s)]
    met_info.append(info)


np_met_info = np.array(met_info)

metdat_D = pd.DataFrame(data=np_met_info[1:,1:],index=np_met_info[1:,0])  # 1st row as the column names
metdat_D.columns = ['data_type', 'file_path']

just_rad = (metdat_D.data_type == 'r')
just_sat = (metdat_D.data_type == 's')
just_vis = (metdat_D.data_type == 'vis')
just_glm = (metdat_D.data_type == 'g')
 
file_sequence = []
for i in range(0,len(idx)):
    new_datetime = idx[i]
    #making progressively larger time slices
    time_slice = (metdat_D.index < new_datetime)
    py_dt = new_datetime.to_pydatetime()
    fig_title,fig_fname_tstr = figure_timestamp(py_dt)

    # the [-1:] slice grabs the latest (last) file to use for plotting
    new_sat = metdat_D[time_slice & just_sat][-1:]
    # file_path.max() is the only way I knew how to make xxx_path a string
    sat_path = new_sat.file_path.max()

    new_rad = metdat_D[time_slice & just_rad][-1:]
    rad_path = new_rad.file_path.max()
    new_glm = metdat_D[time_slice & just_glm][-1:]
    glm_path = new_glm.file_path.max()

    new_vis = metdat_D[time_slice & just_vis][-1:]
    vis_path = new_vis.file_path.max()

    # building a list to define which filepaths to use for each timestep
    # filepaths associated with longer time interval products
    # can show up multiple times in successive timesteps
    new_seq = [new_datetime,sat_path,vis_path,rad_path,glm_path,fig_title,fig_fname_tstr]
    file_sequence.append(new_seq)

#file_sequence = file_sequence[0:2]
plot_vis = True

for fn in range(0,len(file_sequence)):
    new_datetime = file_sequence[fn][0]
    print('DT :' + str(new_datetime))
    sat_file = file_sequence[fn][1]
    print('sat :' + str(sat_file))
    vis_file = file_sequence[fn][2]
    print('vis :' + str(vis_file))

    radar_file = file_sequence[fn][3]
    glm_file = file_sequence[fn][4]
    try:
        G = xr.open_dataset(glm_file)
    except:
        pass
    try:
        VV = xr.open_dataset(vis_file)
    except:
        pass

    fig_title_tstr = file_sequence[fn][-2]
    fig_fname_tstr = file_sequence[fn][-1]

    # obtain lightning slice with time range between current date_range time
    # and the time one time interval ago
    ltg_time_slice = (ltg_D.index > (idx[fn] - dt)) & (ltg_D.index <= idx[fn])
    ltg = ltg_D[ltg_time_slice]

    # process radar file
    data = xr.open_dataset(radar_file)
    dnew2,rlats,rlons,back=latlon_from_radar(data)

    da = dnew2.ReflectivityQC
    ref_arr = da.to_masked_array(copy=True)
    ra_filled = ref_arr.filled()

    # process satellite file
    C = xr.open_dataset(sat_file)
    
    #C02 = C['CMI_C02'].data
    #C02 = np.power(C02, 1/gamma)
    #C02 = C02 * 1.2
    #C03 = C['CMI_C03'].data
    #C03 = np.power(C03, 1/gamma)

    gamma = 1.9
    
    C02 = VV['Rad'].data
    C02 = C02/C02.max()
    C02 = np.power(C02, 1/gamma)
    C08 = C['CMI_C08'].data - 273.15
    C09 = C['CMI_C09'].data - 273.15
    C10 = C['CMI_C10'].data - 273.15
    C13 = C['CMI_C13'].data - 273.15


    test = ['C02','Ref','C13', 'C08','C09','C10']

    test = ['Ref','Ref','C13', 'GLM','ltg_low','ltg_high']
    if plot_vis:
        test = ['C02','Ref','C13', 'GLM','ltg_low','ltg_high']


    # Now that all data arrays are created, we will begin plotting.
    #
    # First, determine satellite projection

    sat_lon = C['goes_imager_projection'].longitude_of_projection_origin
    sat_sweep = C['goes_imager_projection'].sweep_angle_axis
    sat_h = C['goes_imager_projection'].perspective_point_height
    sat_lon = C['goes_imager_projection'].longitude_of_projection_origin
    sat_sweep = C['goes_imager_projection'].sweep_angle_axis
    semi_maj = C['goes_imager_projection'].semi_major_axis
    semi_min = C['goes_imager_projection'].semi_minor_axis
    # The projection x and y coordinates equals the scanning angle (in radians) multiplied by the satellite height
    # See details here: https://proj4.org/operations/projections/geos.html?highlight=geostationary

    p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, a=semi_maj, b=semi_min, sweep=sat_sweep)

    x = C['x'][:] * sat_h
    y = C['y'][:] * sat_h
    if plot_vis:
        x_vis = VV['x'][:] * sat_h
        y_vis = VV['y'][:] * sat_h

    arDict = {}



    #pc = ccrs.PlateCarree()

    # Now perform cartographic transformation for satellite data.
    # That is, convert image projection coordinates (x and y) to longtitudes/latitudess.
    fig, axes = plt.subplots(2,3,figsize=(20,12),subplot_kw={'projection': ccrs.PlateCarree()})
    plt.suptitle(fig_title_tstr)
    font = {'weight' : 'normal', 'size'   : 12}
    plt.titlesize : 24

    XX, YY = np.meshgrid(x, y)
    lons, lats = p(XX, YY, inverse=True)

    if plot_vis:
        visXX, visYY = np.meshgrid(x_vis, y_vis)
        vlons, vlats = p(visXX, visYY, inverse=True)
        arDict['C02'] = {'ar': C02, 'lat':vlats, 'lon':vlons}
    # Create a pyproj geostationary map object


    #arDict['C03'] = {'ar': C03, 'lat':lats, 'lon':lons}
    #arDict['C08'] = {'ar': C08, 'lat':lats, 'lon':lons}
    #arDict['C09'] = {'ar': C09, 'lat':lats, 'lon':lons}
    #arDict['C10'] = {'ar': C10, 'lat':lats, 'lon':lons}
    arDict['C13'] = {'ar': C13, 'lat':lats, 'lon':lons}
    # Don't need to reproject radar data - already got lon/lats from 'latlon_from_radar' function
    arDict['Ref'] = {'ar': ra_filled, 'lat':rlats, 'lon':rlons}
    
    for y,a in zip(test,axes.ravel()):
        a.set_extent(extent, crs=ccrs.PlateCarree())
        for sh in shape_mini:
            a.add_feature(shape_mini[sh], facecolor='none', edgecolor='gray', linewidth=0.5)
            a.tick_params(axis='both', labelsize=8)

        a.set_aspect(1.25)
        if str(y) != 'ltg_low' and str(y) != 'ltg_high' and str(y) != 'GLM':
            lon = arDict[y]['lon']
            lat = arDict[y]['lat']
            arr = arDict[y]['ar']


        if str(y) == 'Ref':
            cs = a.pcolormesh(lat,lon,arr,cmap=plts[y]['cmap'],vmin=plts[y]['vmn'], vmax=plts[y]['vmx'])
            a.set_title(plts[y]['title'])
            #a.set_title('Radar')

        elif str(y) == 'ltg_low':
            if len(ltg) > 0:
                ltg_plot('low',ltg)
            else:
                pass

        elif str(y) == 'ltg_high':
            if len(ltg) > 0:
                ltg_plot('high',ltg)

        elif str(y) == 'GLM':
            try:
                a.scatter(G['flash_lon'], G['flash_lat'], marker='_')
                a.set_title(plts[y]['title'])
                
            except:
                pass

        
        else:
            a.pcolormesh(lon,lat,arr,cmap=plts[y]['cmap'],vmin=plts[y]['vmn'],vmax=plts[y]['vmx'])
            a.set_title(plts[y]['title'])
                
    image_dst_path = os.path.join(image_dir,fig_fname_tstr + '.png')
    plt.savefig(image_dst_path,format='png')

    plt.show()
    plt.close()

try:
    build_html(image_dir)
except:
    pass