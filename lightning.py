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


def ltg_plot_3d(ltg,i):
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
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(111, projection='3d')
    #ax.grid(False)

    plt.titlesize : 24

    shift = (0*0.02)
    
    xmin = -85.0001 + shift
    xmax = -82.7001 + shift
    ymin = 40.0001 + shift
    ymax = 44.7501 + shift
    xticks = make_ticks(xmin,xmax)
    yticks = make_ticks(ymin,ymax)
    #ax.set_xticks(xticks)
    #ax.set_yticks(yticks)

    ax.set_xlim3d([-86.0 + shift, -83.0 + shift])
    ax.set_ylim3d([41.0 + shift, 44.75 + shift])
    ax.set_zlim3d([0.0, 20000.0])

    for st in range(0,len(ltg)):
        lat = ltg.latitude.values[st]
        lon = ltg.longitude.values[st]
        cur = ltg.peakcurrent[st]
        hgt = ltg.icheight[st]
        #print(hgt)
        if cur < 0:
            m = '_'
            c = mpl.colors.to_rgb(0,0,hgt/25000)
        else:
            m = '+'
            c = mpl.colors.to_rgb(hgt/25000,0,0)
        if hgt > 20000:
            col = 1
        else:
            col = hgt/20000
        
        print(col)
        ax.scatter(lon,lat,hgt, c=c, marker=m)
    image_dst_path = os.path.join('C:/data/images/lightning/',str(i) + '.png')
    plt.savefig(image_dst_path,format='png')
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
from my_functions import build_html, make_ticks
from custom_cmaps import azdv_cmap
event_date = this_case['date']
rda = this_case['rda']
extent = this_case['sat_extent']
shapelist = this_case['shapelist']

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
    image_dir = os.path.join(base_dir,'images',event_date,'satellite')

case_dir = os.path.join(base_dir,event_date)
radar_dir = os.path.join(case_dir,rda,'netcdf/ReflectivityQC/00.50')
sat_dir = os.path.join(case_dir,'satellite/raw')
ltg_dir = os.path.join(case_dir,'lightning')


import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
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
idx = pd.date_range('2019-03-14 22:30', periods=60, freq='1Min')
dt = idx[1] - idx[0]
# ----------------------------------------------------------------
# ----------------------------------------------------------------

# building a met_info list containing filepaths for all
# radar and satellite (including GLM) products and their associated datetime
# derived from filename convention

for i in range(1,len(idx)):
    
    ltg_time_slice = (ltg_D.index > (idx[i-1])) & (ltg_D.index < idx[i])
    ltg = ltg_D[ltg_time_slice]





# Now perform cartographic transformation for satellite data.
# That is, convert image projection coordinates (x and y) to longtitudes/latitudess.

    ltg_plot_3d(ltg,i)
#image_dst_path = os.path.join(image_dir,fig_fname_tstr + '.png')
#plt.savefig(image_dst_path,format='png')

    plt.show()
    plt.close()


try:
    build_html('C:/data/images/lightning/')
except:
    pass
