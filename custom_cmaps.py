# -*- coding: utf-8 -*-
"""
Creates custom cmaps for matplotlib
https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html

Assumption: You'll import the created cmaps into wdss_create_netcdfs.py

author: thomas.turnage@noaa.gov
Last updated: 28 May 2019
------------------------------------------------

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
from metpy.plots import colortables
from matplotlib.colors import LinearSegmentedColormap

#-------- Begin creating custom color maps --------

#--- Spectrum Width
sw_colors = [(0,0,0),(220,220,255),(180,180,240),(50,50,150),(255,255,0),(255,150,0),(255,0,0),(255,255,255)]
sw_position = [0, 1/40, 5/40, 0.25, 15/40, 0.5, 0.75, 1]
sw_cmap=make_cmap(sw_colors, position=sw_position,bit=True)
plt.register_cmap(cmap=sw_cmap)

#--- Velocity Gradient
vg_colors = [(0, 0, 0),(32,32,32),(128,128,128),(117,70,0),(151,70,0),(186,70,0),(220,132,0),(255,153,0),(119,0,0,),(153,0,0),(187,0,0),
             (221,0,0),(255,0,0),(255,204,204),(255,204,255),(255,255,255),(255,255,255)]
vg_position = [0, 1/15, 2/15, 3/15, 4/15, 5/15, 6/15, 7/15, 8/15, 9/15, 10/15, 11/15, 12/15, 13/15, 14/15, 0.999999, 1 ]
vg_cmap=make_cmap(vg_colors, position=vg_position,bit=True)
plt.register_cmap(cmap=vg_cmap)


#--- Velocity Gradient
vg_colors = [(0, 0, 0),(32,32,32),(128,128,128),(117,70,0),(151,70,0),(186,70,0),(220,132,0),(255,153,0),(119,0,0,),(153,0,0),(187,0,0),
             (221,0,0),(255,0,0),(255,204,204),(255,204,255),(255,255,255),(255,255,255)]
vg_position = [0, 1/15, 2/15, 3/15, 4/15, 5/15, 6/15, 7/15, 8/15, 9/15, 10/15, 11/15, 12/15, 13/15, 14/15, 0.999999, 1 ]
vg_cmap=make_cmap(vg_colors, position=vg_position,bit=True)
plt.register_cmap(cmap=vg_cmap)

#ir_colors = [(31, 31, 31),(31, 31, 31),(0,113,113),(31,255,255),(255,255,31),(255,0,0),(0,0,0),(255,255,255),(145,31,145),(145,31,145)]
#ir_position = [0, 2/16, 5/16, 8/16, 9/16, 10/16, 11/16, 13/16, 15/16, 1]
ir_colors = [(145,31,145),(145,31,145),(255,255,255),(0,0,0),(255,0,0),(255,255,31),(31,255,255),(0,113,113),(31, 31, 31),(31, 31, 31)]
ir_position = [0, 1/16, 3/16, 5/16, 6/16, 7/16, 8/16, 11/16, 14/16, 1]
ir2_cmap=make_cmap(ir_colors, position=ir_position,bit=True)
plt.register_cmap(cmap=ir2_cmap)


"""
color: 50 31 31 31
color: 30 0 113 113
color: 0  31 255 255
color: -30 0 0 115
color: -40 31 241 40
color: -50 255 255 31
color: -60 255 0 0
color: -70 0 0 0
color: -90 255 255 255
color: -110 145 31 145
"""

ir3_colors = [(78,61,158),(177,104,212),(178,178,178),(0,0,0),(199,23,0),(190,201,2),(2,215,8),(2,7,191),(0,190,200),(197, 197, 197),(0, 0, 0)]
ir3_position = [0, 2/16, 3/16, 4/16, 5/16, 6/16, 7/16, 8/16, 9/16, 9.001/16,1]
ir3_cmap=make_cmap(ir3_colors, position=ir3_position,bit=True)
plt.register_cmap(cmap=ir3_cmap)


"""
16 color: 50 0 0 0 
9.001 color: -19.99 197 197 197
9 color: -20 0 190 200
8 color: -30 2 7 191
7 color: -40 2 215 8
6 color: -50 190 201 2 
5 color: -60 199 23 0
4 color: -70 0 0 0
3 color: -80 178 178 178
2 color: -90 177 104 212
color: -110 78 61 158
"""

#wv_colors = [(68,68,68),(0,197,156),(255,255,255),(6,4,121),(254,250,3),(6,4,121),(245, 0, 0),(254, 250, 3)]
#wv_position = [0, 46/256, 76/256, 91/256, 101/256, 125.9/256, 126/256, 1]


wv_colors = [(0,238,234),(0,197,156),(255,255,255),(18,18,172),(241,241,9),(255,12,0),(0, 0, 0)]
wv_position = [0, (109-75)/109,(109-47)/109, (109-30)/109, (109-15.5)/109,108/109,1 ]
wv_cmap=make_cmap(wv_colors, position=wv_position,bit=True)
plt.register_cmap(cmap=wv_cmap)

wv_colors_r = [(254, 250, 3),(245, 0, 0),(6,4,121),(254,250,3),(6,4,121),(255,255,255),(0,121,0),(0,238,234)]
wv_position_r = [0, 130/256, 130.1/256, 155/256, 165/256, 180/256, 210/256, 1]
wv_cmap_r=make_cmap(wv_colors_r, position=wv_position_r,bit=True)
plt.register_cmap(cmap=wv_cmap_r)

"""
units: 256
step: 16
product: WV
256                  254  250    3
126    color: 130    245    0    0
125.9                  6    4  121
101    color: 155    254  250    3      
91     color: 165      6    4  121 
76     color: 180    255  255  255 
46     color: 210      0  121    0
0      color: 256      0  238  234


color: 130 245 0 0  254 250 3
color: 155 254 250 3  6 4 121
color: 165 6 4 121 
color: 180 255 255 255 
color: 210 0 121 0
color: 256 0 238 234

"""


#--- Reflectivity
ref_colors = [(0,0,0),(130,130,130),(95,189,207),(57,201,105),(57,201,105),(0,40,0),(9,94,9),(255,207,0),(255,207,0),(255,207,0),(255,133,0),(255,0,0),(89,0,0),(255,245,255),(225,11,227),(164,0,247),(99,0,214),(5,221,224),(58,103,181),(255,255,255)]
ref_position = [0, 45/110, 46/110, 50/110, 51/110, 65/110, 66/110, 70/110, 71/110, 80/110, 81/110, 90/110, 91/110, 100/110, 101/110, 105/110, 106/110, 107/110, 109/110, 1]
ref_cmap=make_cmap(ref_colors, position=ref_position,bit=True)
plt.register_cmap(cmap=ref_cmap)

#--- Azimuthal Shear / Div Shear
azdv_colors = [(1,1,1),(1,1,1),(0,0,1),(0,0,0.7),(0,0,0),(0.7,0,0),(1,0,0),(1,1,1),(1,1,1)]
azdv_position = [0, 0.001, 0.3, 0.43, 0.5, 0.57, 0.7, 0.999, 1]
azdv_cmap=make_cmap(azdv_colors, position=azdv_position)
plt.register_cmap(cmap=azdv_cmap)

#--- Velocity - need to home grow this so I don't require import from metpy
v_norm, v_cmap = colortables.get_with_range('NWS8bitVel', -40, 40)
v_cmap.set_under('k')
plt.register_cmap(cmap=v_cmap)




#-------- End creating custom color maps --------

