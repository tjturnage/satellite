# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 10:21:20 2019

@author: tjtur
"""
import numpy as np
from pyproj import Proj
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
#from metpy.plots import colortables
import custom_cmaps
#import matplotlib.pyplot as plt


#ir_norm, ir_cmap = colortables.get_with_range('ir_rgbv', -110, 55)
#ir_cmap.set_under('k')
#plt.register_cmap(cmap=ir_cmap)
#file16 = 'C:/data/2018-04-13-meteotsunami/satellite/CH08/G16_C08_20180413_115722.nc'
#C = xr.open_dataset(file16)
FILE = 'C:/data/satellite/test2/OR_ABI-L2-MCMIPC-M6_G16_s20191660016501_e20191660019273_c20191660019397.nc'
C = xr.open_dataset(FILE)
C08 = C['CMI_C08'].data - 273.15
C09 = C['CMI_C09'].data - 273.15
C10 = C['CMI_C10'].data - 273.15
C13 = C['CMI_C13'].data - 273.15
#C13 = C['CMI_C02']
sat_h = C['goes_imager_projection'].perspective_point_height

# Satellite longitude
sat_lon = C['goes_imager_projection'].longitude_of_projection_origin

# Satellite sweep
sat_sweep = C['goes_imager_projection'].sweep_angle_axis

# Satellite height
sat_h = C['goes_imager_projection'].perspective_point_height

# Satellite longitude
sat_lon = C['goes_imager_projection'].longitude_of_projection_origin

# Satellite sweep
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
fig, ax = plt.subplots(1,1,figsize=(14,7),subplot_kw={'projection': ccrs.PlateCarree()})
XX, YY = np.meshgrid(x, y)
lons, lats = p(XX, YY, inverse=True)
ax.set_extent([-90,-82,40,47.5], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.STATES, facecolor='none', edgecolor='black')
ax.pcolormesh(lons, lats, C10, cmap=wv_cmap,vmin=-110,vmax=0)
plt.show()
"""
pc = ccrs.PlateCarree()
lc = ccrs.LambertConformal(central_longitude = -86.0,
                           standard_parallels = (40,45))

ax1 = fig.add_subplot(1, 1, 1, projection=p)
ax1 = fig.add_subplot(2, 1, 2, projection=lc)
ax1.set_extent([-94, -82, 40, 46], crs=lc)
ax1.imshow(rad16, origin='upper',
           extent=(x16.min(), x16.max(), y16.min(), y16.max()), cmap='gray_r', transform=geos16, vmin=0, vmax=40,interpolation='none')


"""