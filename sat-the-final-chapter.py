# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 10:21:20 2019

@author: tjtur
"""
def make_cmap(colors, position=None, bit=False):
    """
    Creates colormaps (cmaps) for different products.
    
    Information on cmap with matplotlib
    https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html
    
    Parameters
    ----------
       colors : list of tuples containing RGB values. Tuples must be either:
                - arithmetic (zero to one) - ex. (0.5, 1, 0.75)
                - 8-bit                    - ex. (127,256,192)
     position : ordered list of floats
                None: default, returns cmap with equally spaced colors
                If a list is provided, it must have:
                  - 0 at the beginning and 1 at the end
                  - values in ascending order
                  - a number of elements equal to the number of tuples in colors
          bit : boolean         
                False : default, assumes arithmetic tuple format
                True  : set to this if using 8-bit tuple format
    Returns
    -------
         cmap
                    
    """  
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap

def latlon_from_radar(az,elevation,num_gates):
    """
    Convert radar bin radial coordinates to lat/lon coordinates.
    Adapted from Brian Blaylock code
    
    Parameters
    ----------
          az : numpy array
               All the radials for that particular product and elevation
               Changes from 720 radials for super-res product cuts to 360 radials
   elevation : float
               The radar elevation slice in degrees. Needed to calculate range 
               gate length (gate_len) as projected on the ground using simple
               trigonometry. This is a very crude approximation that doesn't
               factor for terrain, earth's curvature, or standard beam refraction.
   num_gates : integer
               The number of gates in a radial, which varies with 
               elevation and radar product. That is why each product makes 
               an individual call to this function. 
                    
    Returns
    -------
         lat : array like
         lon : array like
        back : I have no idea what this is for. I don't use it.
                    
    """
    rng = None
    factor = math.cos(math.radians(elevation))
    if num_gates <= 334:
        gate_len = 1000.0 * factor
    else:
        gate_len = 250.0 * factor
    rng = np.arange(2125.0,(num_gates*gate_len + 2125.0),gate_len)
    g = Geod(ellps='clrk66')
    center_lat = np.ones([len(az),len(rng)])*dnew2.Latitude
    center_lon = np.ones([len(az),len(rng)])*dnew2.Longitude
    az2D = np.ones_like(center_lat)*az[:,None]
    rng2D = np.ones_like(center_lat)*np.transpose(rng[:,None])
    lat,lon,back=g.fwd(center_lon,center_lat,az2D,rng2D)
    return lat,lon,back

import numpy as np
from pyproj import Proj
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import matplotlib as mpl
#from metpy.plots import colortables
from custom_cmaps import wv_cmap, ref_cmap
import sys
import os
import math
from pyproj import Geod
import pandas as pd
from datetime import datetime
#import matplotlib.pyplot as plt

ltg_dir = 'C:/data/20190601/lightning'
radar_stage_dir = 'C:/data/20190601/KGRR/stage'
sat_source_dir = 'C:/data/20190601/satellite/raw'
sat_stage_dir = 'C:/data/20190601/satellite/stage'

ltg_D = []
radsat_D = []


ltg_files = os.listdir(ltg_dir)
ltg_csv = os.path.join(ltg_dir,ltg_files[0])
ltg_D = pd.read_csv(ltg_csv, sep=',', delimiter=None, header='infer',index_col=1,usecols=[0,1,2,3,5],skip_blank_lines=True)

met_info = []

#f = open('metfiles', 'w')
#f.write('time,datatype,filename\n')

radar_files = os.listdir(radar_stage_dir)
for r in (radar_files):
    rad_info = str.split(r,'_')
    rad_time_str = rad_info[0]
    rad_datetime = datetime.strptime(rad_time_str,"%Y%m%d-%H%M%S")
    #rad_pd_time = datetime.strftime(rad_datetime,"%Y-%m-%dT%H:%M:%S")
    #pd_datetime = pd.to_datetime(rad_pd_time)
    info = [rad_datetime,'r',os.path.join(radar_stage_dir,r)]
    met_info.append(info)


satellite_files = os.listdir(sat_source_dir)
for s in (satellite_files):
    sat_split = str.split(s,'_')
    sat_time_s = sat_split[3]
    sat_time = sat_time_s[1:-1]
    sat_datetime = datetime.strptime(sat_time, "%Y%j%H%M%S")
    info = [sat_datetime,'s',os.path.join(sat_source_dir,s)]
    met_info.append(info)


print(met_info)   

idx = pd.date_range('2019-06-01 22:15', periods=22, freq='2.5Min')

np_met_info = np.array(met_info)

metdat_D = pd.DataFrame(data=np_met_info[1:,1:],index=np_met_info[1:,0])  # 1st row as the column names
metdat_D.columns = ['data_type', 'file_path']

raddat = metdat_D[metdat_D.data_type == 'r']
satdat = metdat_D[metdat_D.data_type == 's']

file_sequence = []
for i in range(0,len(idx)):
    new_datetime = idx[i]
    #print(new_datetime)
    new_sat = satdat[satdat.index < new_datetime][-1:]
    sat_path = new_sat.file_path.max()

    new_rad = raddat[raddat.index < new_datetime][-1:]
    #print(new_rad)
    rad_path = new_rad.file_path.max()
    print(rad_path)

    
    #new_ltg = ltg_D[ltg_D.index < new_datetime][-1:]
    new_seq = [sat_path,rad_path]
    file_sequence.append(new_seq)


base_gis_dir = 'C:/data/GIS'

for ST in ['MI']:
    st = ST.lower()
    counties_dir = 'counties_' + st
    county_reader = 'county_' + st
    counties_shape = 'counties_' + ST + '.shp'
    COUNTIES_ST = 'COUNTIES_' + ST
    counties_shape_path = os.path.join(base_gis_dir,counties_dir,counties_shape)
    county_reader = counties_shape_path
    reader = shpreader.Reader(counties_shape_path)
    counties = list(reader.geometries())
    COUNTIES_ST = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

ir4_colors = [(0,0,0),(255,255,255),(0,0,0), (255,0,0), (255,255,0), (0,255,0), (0,0,255), (191,0,255), (255,255,255),(0,0,0),(120,120,120),(0,0,0)]
ir4_position = [0, 10/166, 35/166, 45/166, 55/166, 65/166, 82/166, 90/166, 95/166, 135.9/166, 136/166, 1]
ir4_cmap=make_cmap(ir4_colors, position=ir4_position,bit=True)
plt.register_cmap(cmap=ir4_cmap)



#sat_base = 'C:/data/20190601/satellite/raw'
#radar_file = 'C:/data/20190601/KGRR/stage/20190601-222645_Refl_00.50.netcdf'
#fname = 'OR_ABI-L2-MCMIPC-M6_G16_s20191522231416_e20191522234189_c20191522234306.nc'
#FILE = os.path.join(sat_base,fname)
#FILE = 'C:/data/satellite/test2/OR_ABI-L2-MCMIPC-M6_G16_s20191660016501_e20191660019273_c20191660019397.nc'
i = 1
last_radar_file = None
last_FILE = None
for fn in range(0,len(file_sequence)):
    FILE = file_sequence[fn][0]
    radar_file = file_sequence[fn][1]
    C = xr.open_dataset(FILE)
    data = xr.open_dataset(radar_file)
    degrees_tilt = data.Elevation
    dnew2 = data.sortby('Azimuth')
    azimuths = dnew2.Azimuth.values
    num_gates = len(dnew2.Gate)
    rlats,rlons,rback=latlon_from_radar(azimuths,degrees_tilt,num_gates)
    da = dnew2.ReflectivityQC
    ref_arr = da.to_masked_array(copy=True)
    ra_filled = ref_arr.filled()

    gamma = 2.0
    
    C02 = C['CMI_C02'].data
    C02 = np.power(C02, 1/gamma)
    C03 = C['CMI_C03'].data
    C03 = np.power(C03, 1/gamma)
    
    C08 = C['CMI_C08'].data - 273.15
    C09 = C['CMI_C09'].data - 273.15
    C10 = C['CMI_C10'].data - 273.15
    C13 = C['CMI_C13'].data - 273.15

    plts = {}
    plts['C02'] = {'cmap':'Greys_r','vmn':0.0,'vmx':1.0,'title':'Channel 2 Visible'}
    plts['C03'] = {'cmap':'Greys_r','vmn':0.0,'vmx':1.0,'title':'Channel 3 Near IR'}
    plts['C08'] = {'cmap':wv_cmap,'vmn':-109.0,'vmx':0.0,'title':'Channel 8 W/V'}
    plts['C09'] = {'cmap':wv_cmap,'vmn':-109.0,'vmx':0.0,'title':'Channel 9 W/V'}
    plts['C10'] = {'cmap':wv_cmap,'vmn':-109.0,'vmx':0.0,'title':'Channel 10 W/V'}
    plts['C13'] = {'cmap':ir4_cmap,'vmn':-110.0,'vmx':56.0,'title':'Channel 13 IR'}
    plts['Ref'] = {'cmap':ref_cmap,'vmn':-30,'vmx':80,'title':'Reflectivity','cbticks':[0,15,30,50,60],'cblabel':'dBZ'}
    
    test = ['C02','Ref','C13', 'C08','C09','C10']
    #test = ['C08','C13']

    # Satellite longitude
    sat_lon = C['goes_imager_projection'].longitude_of_projection_origin
    sat_sweep = C['goes_imager_projection'].sweep_angle_axis
    sat_h = C['goes_imager_projection'].perspective_point_height
    sat_lon = C['goes_imager_projection'].longitude_of_projection_origin
    sat_sweep = C['goes_imager_projection'].sweep_angle_axis
    semi_maj = C['goes_imager_projection'].semi_major_axis
    semi_min = C['goes_imager_projection'].semi_minor_axis
    # The projection x and y coordinates equals the scanning angle (in radians) multiplied by the satellite height
    # See details here: https://proj4.org/operations/projections/geos.html?highlight=geostationary
    x = C['x'][:] * sat_h
    y = C['y'][:] * sat_h

    # Create a pyproj geostationary map object
    p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, a=semi_maj, b=semi_min, sweep=sat_sweep)
    pc = ccrs.PlateCarree()
    # Perform cartographic transformation. That is, convert image projection coordinates (x and y)
    # to latitude and longitude values.
    fig, axes = plt.subplots(2,3,figsize=(11,7),subplot_kw={'projection': ccrs.PlateCarree()})
    XX, YY = np.meshgrid(x, y)
    lons, lats = p(XX, YY, inverse=True)

    arDict = {}
    arDict['C02'] = {'ar': C02, 'lat':lats, 'lon':lons}
    arDict['C03'] = {'ar': C03, 'lat':lats, 'lon':lons}
    arDict['C08'] = {'ar': C08, 'lat':lats, 'lon':lons}
    arDict['C09'] = {'ar': C09, 'lat':lats, 'lon':lons}
    arDict['C10'] = {'ar': C10, 'lat':lats, 'lon':lons}
    arDict['C13'] = {'ar': C13, 'lat':lats, 'lon':lons}
    arDict['Ref'] = {'ar': ra_filled, 'lat':rlats, 'lon':rlons}
    
    extent = [-86.4,-84.3,41.7,43.2]

    for y,a in zip(test,axes.ravel()):
        a.set_extent(extent, crs=ccrs.PlateCarree())
        a.set_aspect(1.25)
        lon = arDict[y]['lon']
        lat = arDict[y]['lat']
        arr = arDict[y]['ar']
        a.add_feature(COUNTIES_ST, facecolor='none', edgecolor='gray')
        if str(y) == 'Ref':
            cs = a.pcolormesh(lat,lon,arr,cmap=plts[y]['cmap'],vmin=plts[y]['vmn'], vmax=plts[y]['vmx'])
        else:
            a.pcolormesh(lon,lat,arr,cmap=plts[y]['cmap'],vmin=plts[y]['vmn'],vmax=plts[y]['vmx'])

    image_dst_path = os.path.join(sat_stage_dir,str(i) + '.png')
    plt.savefig(image_dst_path,format='png')
    i = i + 1
    #print(mosaic_fname[:-4] + ' mosaic complete!')    
    plt.show()
    plt.close()
    last_FILE = FILE
    last_radar_file = radar_file
"""
fig2, axes2 = plt.subplots(2,1,figsize=(11,7),sharex='none',subplot_kw={'projection': ccrs.PlateCarree()})
for z,d in zip(test,axes2.ravel()):
    d.set_extent(extent, crs=ccrs.PlateCarree())
    d.set_aspect(1.25)
    d.add_feature(COUNTIES_ST, facecolor='none', edgecolor='black')
    d.pcolormesh(rlons,rlats,ref_arr, cmap=ref_cmap,vmin=-30,vmax=80)


plt.show()
"""