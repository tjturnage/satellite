# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:35:29 2018

@author: Owner
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Data obtained from ftp://ftp-restricted.ncdc.noaa.gov/data/lightning/
ltg_dir = 'C:/data/20190601/lightning'
radar_stage_dir = 'C:/data/20190601/KGRR/stage'
sat_source_dir = 'C:/data/20190601/satellite/raw'
sat_stage_dir = 'C:/data/20190601/satellite/stage'

ltg_D = []
radsat_D = []


ltg_files = os.listdir(ltg_dir)
ltg_csv = os.path.join(ltg_dir,ltg_files[0])

df = None
#df = pd.read_csv(ltg_csv, names = ['flashType', 'time', 'latitude', 'longitude','peakcurrent','icheight','numbersensors','multiplicity'], index_col=['time'])
df = pd.read_csv(ltg_csv, index_col=['time'])
just_times = df.index.to_series()

#df = pd.read_csv('pandas_dataframe_importing_csv/example.csv', index_col=['First Name', 'Last Name'], names=['UID', 'First Name', 'Last Name', 'Age', 'Pre-Test Score', 'Post-Test Score'])
ltg_ar = []
with open(ltg_csv) as ltg:
    for line in ltg:
        this_line = str.splitlines(line)
        els = str.split(str(this_line),',')
        ltg_type = els[0][2:]
        ltg_lat = els[2]
        ltg_lon = els[3]
        dt_str = els[1][0:23]
        try:
            ltg_datetime = datetime.strptime(dt_str,"%Y-%m-%dT%H:%M:%S.%f")
            this_ltg = [ltg_datetime,ltg_type,els[2],els[3],els[5]]
            #print(ltg_datetime, els[0],els[2],els[3])
            ltg_ar.append(this_ltg)
        except:
            pass

ltgdat_D = None
print(ltg_ar)
np_ltg = np.array(ltg_ar)
ltgdat_D = pd.DataFrame(data=np_ltg[1:,1:],index=np_ltg[1:,0])# ltgdat_D = None 
ltgdat_D = pd.DataFrame(data=ltg_ar[1:,1:])  # 1st row as the column names

ltgdat_D.columns = ['type','lat','lon','icheight']    


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

idx = pd.date_range('2019-06-01 22:15', periods=12, freq='5Min')

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
