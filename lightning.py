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

import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

import numpy as np
import os



try:
    os.listdir('/var/www')
    windows = False
    sys.path.append('/data/scripts/resources')
    from case_data import this_case
    image_dir = os.path.join('/var/www/html/radar','images')
except:
    windows = True
    sys.path.append('C:/data/scripts/resources')
    from case_data import this_case
    base_dir = 'C:/data'
    base_gis_dir = 'C:/data/GIS'
    event_date = this_case['date']
    image_dir = os.path.join(base_dir,'images',event_date,'satellite')


from my_functions import build_html


case_dir = os.path.join(base_dir,event_date)
ltg_dir = os.path.join(case_dir,'lightning')


import matplotlib.colors as colors
from my_functions import figure_timestamp
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
#ltg_colors = [(0.8,0,0),(0.8,0,0),(0,0,0.8),(0,0.8,0)]
#ltg_position = [0,0.25,0.75,1]
ltg_colors = [(4/5,0,0),(0,4/5,0),(0,0,4/5)]
ltg_position = [0,0.6,1]
ltg_cmap=make_cmap(ltg_colors, position=ltg_position)
plt.register_cmap(cmap=ltg_cmap)


ltg_D = []
# lightning files obtained from EarthNetworks
ltg_files = os.listdir(ltg_dir)
ltg_csv = os.path.join(ltg_dir,ltg_files[0])
ltg_D = pd.read_csv(ltg_csv, index_col=['time'])
ltg_D.index = [datetime.strptime(x[:-2], '%Y-%m-%dT%H:%M:%S.%f') for x in ltg_D.index]
ltg_hgt = ltg_D['icheight']
vmax_hgt = ltg_hgt.max()
norm=colors.Normalize(vmin=0, vmax=vmax_hgt)

# Here is a step where we define to bin plots by time
# ----------------------------------------------------------------
# ----------------------------------------------------------------
#idx = pd.date_range('2019-03-14 22:30', periods=60, freq='1Min')
idx = pd.date_range('2019-06-01 22:00', periods=90, freq='2Min')
dt = idx[1] - idx[0]
# ----------------------------------------------------------------
# ----------------------------------------------------------------

# building a met_info list containing filepaths for all
# radar and satellite (including GLM) products and their associated datetime
# derived from filename convention

for i in range(1,len(idx)):
    
    ltg_time_slice = (ltg_D.index > (idx[i-1])) & (ltg_D.index < idx[i])
    new_datetime = idx[i]
    py_dt = new_datetime.to_pydatetime()
    fig_title,fig_fname_tstr = figure_timestamp(py_dt)
    ltg = ltg_D[ltg_time_slice]
    hgt = ltg['icheight']
    vmax = hgt.max()

    ltg_n = ltg[(ltg['peakcurrent']<=0)]
    lon_n = np.array(ltg_n['longitude'].tolist())
    lat_n = np.array(ltg_n['latitude'].tolist())
    hgt_n = np.array(ltg_n['icheight'].tolist())
    current_n = np.array(ltg_n['peakcurrent'].tolist())
    #n_size = np.log(np.absolute(current_n)) * 20
    n_size = np.absolute(current_n)/50
    n_col = hgt_n/25000
    vmax_n = 25000

    ltg_p = ltg[(ltg['peakcurrent']>0)]
    lon_p = np.array(ltg_p['longitude'].tolist())
    lat_p = np.array(ltg_p['latitude'].tolist())
    hgt_p = np.array(ltg_p['icheight'].tolist())
    current_p = np.array(ltg_p['peakcurrent'].tolist())
    #p_size = current_p/100    
    p_size = np.absolute(current_p)/50
    #p_size = np.log(np.absolute(current_p)) * 20
    p_col = hgt_p/25000
    vmax_p = 25000
    fig = plt.figure(figsize=(11,7))
    plt.suptitle('Total Lightning\n' + fig_title + '\nMarker size proportional to peak current' )
    plt.titlesize : 24
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlim3d([-86.0, -84.0])
    ax.set_ylim3d([41.75, 42.75])
    ax.set_zlim3d([0.0, 20000.0])
    sc = ax.scatter(lon_n,lat_n,hgt_n, c=n_col, marker="_", s=n_size,depthshade=False,cmap=ltg_cmap)
    ax.scatter(lon_p,lat_p,hgt_p, c=p_col, marker="+", s=p_size,depthshade=False,cmap=ltg_cmap)
    #hgt = np.expand_dims(hgt,axis=0)
    #setthis = np.arange(0,25000,100)
    #setthis = np.expand_dims(setthis,axis=0)
    #plt.imshow(setthis,interpolation='nearest',vmin=0,vmax=25000,cmap=ltg_cmap)
    plt.colorbar(sc, ticks = [],shrink=0.5,aspect=10)
    #fig.colorbar(cbar,shrink=0.5,aspect=5)
    
    
    image_dst_path = os.path.join('C:/data/images/lightning/',fig_fname_tstr + '.png')
    plt.savefig(image_dst_path,format='png')

#
    plt.show()
    plt.close()
#
#
try:
    build_html('C:/data/images/lightning/')
except:
    pass
