import pickle, rasterio, os, s3fs, zarr, requests, tempfile, zipfile, glob, os, sys, lzma

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import geopandas as gpd

from time import sleep
from datetime import datetime, date, timedelta
from shapely.geometry import Point
from solarpy import irradiance_on_plane

from scipy.interpolate import griddata

# Define input paths
path_to_data = r"/Users/Guille/Desktop/caiso_power/data/"
path_to_temp = r"/Users/Guille/Desktop/caiso_power/data/temp/"

# Define output paths
path_to_oac = r"/Users/Guille/Desktop/caiso_power/output/actuals/"
path_to_ofc = r"/Users/Guille/Desktop/caiso_power/output/forecasts/"
path_to_oax = r"/Users/Guille/Desktop/caiso_power/output/auxiliary/"
path_to_opd = r"/Users/Guille/Desktop/caiso_power/output/processed_data/"

year      = int(sys.argv[1])
month     = int(sys.argv[2])
day       = int(sys.argv[3])
# Retrive a year long data
_start_date = date(year, month, day)
_end_date   = date(year + 1, 1, 1)
# Define path to files
path = [path_to_oac, path_to_oac, path_to_ofc][forecasts]
# Define Resources to loop over
resources_ = ['Demand', 'Solar', 'Wind']
# Get daylong data request
_delta = timedelta(days = 1)
# Loop over number of days
for i in range((_end_date - _start_date).days):
    # Get date in API request format
    time = f"{_start_date.year:02}{_start_date.month:02}{_start_date.day:02}"

    try:
        # Load CAISO Data
        data_p_ = []
        # Loop Over type of resource
        for j in range(len(resources_)):
            resource = resources_[j]
            # Load resource file for a given data
            pickle_file_name = path + f"{resource}_{time}.pkl"
            with open(pickle_file_name, 'rb') as _f:
                data_p_.append(pickle.load(_f))
            #os.remove(pickle_file_name)
        # Save Predictors and Covariates in the same set
        data_ = []
        data_.append(np.concatenate(data_p_, axis = 1))
        # Load Weather Data
        pickle_file_name = path + f"Weather_{time}.pkl"
        with open(pickle_file_name, 'rb') as _f:
            data_.append(pickle.load(_f))
        #os.remove(pickle_file_name)
        # Save Data in LZMA compression format
        lzma_file_name = path_to_opd + f"ac_{time}.xz"
        with lzma.open(lzma_file_name, "wb") as _f:
            pickle.dump(data_, _f)
        print(lzma_file_name)
    # Prite dates with missing files
    except:
        print('Missing: ', pickle_file_name)
        pass

    # Got to next day
    _start_date += _delta
