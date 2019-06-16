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


import numpy as np
from pyproj import Proj
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib as mpl
#from metpy.plots import colortables
from custom_cmaps import wv_cmap
import sys
#import matplotlib.pyplot as plt


ir4_colors = [(0,0,0),(255,255,255),(0,0,0), (255,0,0), (255,255,0), (0,255,0), (0,0,255), (191,0,255), (255,255,255),(0,0,0),(120,120,120),(0,0,0)]
ir4_position = [0, 10/166, 35/166, 45/166, 55/166, 65/166, 82/166, 90/166, 95/166, 135.9/166, 136/166, 1]
ir4_cmap=make_cmap(ir4_colors, position=ir4_position,bit=True)
plt.register_cmap(cmap=ir4_cmap)


FILE = 'C:/data/satellite/test2/OR_ABI-L2-MCMIPC-M6_G16_s20191660016501_e20191660019273_c20191660019397.nc'
C = xr.open_dataset(FILE)

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
plts['C02'] = {'fname': C02, 'cmap':'Greys_r','vmn':0.0,'vmx':1.0,'title':'Channel 2 Visible'}
plts['C03'] = {'fname': C03, 'cmap':'Greys_r','vmn':0.0,'vmx':1.0,'title':'Channel 3 Near IR'}
plts['C08'] = {'fname': C08, 'cmap':wv_cmap,'vmn':-109.0,'vmx':0.0,'title':'Channel 8 W/V'}
plts['C09'] = {'fname': C09, 'cmap':wv_cmap,'vmn':-109.0,'vmx':0.0,'title':'Channel 9 W/V'}
plts['C10'] = {'fname': C10, 'cmap':wv_cmap,'vmn':-109.0,'vmx':0.0,'title':'Channel 10 W/V'}
plts['C13'] = {'fname': C13, 'cmap':ir4_cmap,'vmn':-110.0,'vmx':56.0,'title':'Channel 13 IR'}

test = ['C02','C03','C13', 'C08','C09','C10']


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
extent = [-93.5,-86.5,42,47.5]
for y,a in zip(test,axes.ravel()):
    this_title = plts[y]['title']
    a.set_extent(extent, crs=ccrs.PlateCarree())
    a.set_aspect(1.25)
    a.add_feature(cfeature.STATES, facecolor='none', edgecolor='black')
    a.pcolormesh(lons,lats,plts[y]['fname'], cmap=plts[y]['cmap'],vmin=plts[y]['vmn'],vmax=plts[y]['vmx'])
    #cs = a.pcolormesh(lats,lons,plts[y]['fname'],cmap=plts[y]['cmap'],vmin=plts[y]['cmap'], vmax=plts[y]['vmx'])

plt.show()
