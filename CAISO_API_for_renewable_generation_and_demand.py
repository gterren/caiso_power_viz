import pickle, rasterio, os, s3fs, zarr, requests, tempfile, zipfile, glob, os, sys

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

forecasts = int(sys.argv[1])
year      = int(sys.argv[2])
month     = int(sys.argv[3])
day       = int(sys.argv[4])

# GMT: Greenwich Mean Time
def _download_csv(url, zip_file_name = 'temp.zip'):
    print(url)
    # Request file on the url
    response = requests.get(url, stream = True)
    # Downalaod zip file from url
    with open(zip_file_name, "wb") as f:
        for chunk in response.iter_content(chunk_size = 512):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    # unzip file
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall('')
    # Find unzip .csv in temp folder
    csv_file_name = glob.glob('*.csv')[0]
    # Open /csv file
    df_ = pd.read_csv(csv_file_name)
    # Remove temp files
    os.remove(csv_file_name)
    os.remove(zip_file_name)
    return df_

# Process OASIS CAISO renewable energy generation data and save it in pickle files
def _process_renewable_energy_generation(i_data_, path, date, HUBs_):
    # Loop over available renewable resource
    for i in range(len(i_data_['RENEWABLE_TYPE'].unique())):
        # Initialize storage dataframe
        o_data_ = pd.DataFrame()
        # Loop over traiding hub
        for j in range(len(HUBs_)):
            # Get Data from Trading hub
            idx_ = (i_data_['RENEWABLE_TYPE'] == i_data_['RENEWABLE_TYPE'].unique()[i]) & (i_data_['TRADING_HUB'] == HUBs_[j])
            # Sort data from hourly sequence
            idx_prime_       = i_data_['OPR_HR'][idx_].to_numpy()
            idx_prime_prime_ = np.argsort(idx_prime_)
            time_            = idx_prime_[idx_prime_prime_]
            energy_          = i_data_['MW'][idx_].to_numpy()[idx_prime_prime_]
            # Dont save data when a trading hub does not have a resource
            if len(energy_) == 0: break
            # Storage Data in dataframe
            o_data_[HUBs_[j]] = energy_

        # Get renewalbe generation resource
        resource = i_data_['RENEWABLE_TYPE'].unique()[i]
        # Save renewable resource energy generation data in pickle file
        _save_CAISO_data_in_pickle_file(o_data_, path, resource, date)


# Process OASIS CAISO energy demand data and save it in pickle files
def _process_energy_demand(i_data_, path, date, TACs_):
    resource = r'Demand'
    # Initialize storage dataframe
    o_data_ = pd.DataFrame()

    # Loop over TAC areas
    for i in range(len(TACs_)):
        # Get Data from TAC area
        idx_ = i_data_['TAC_AREA_NAME'] == TACs_[i]
        # Sort data from hourly sequence
        idx_prime_       = i_data_['OPR_HR'][idx_].to_numpy()
        idx_prime_prime_ = np.argsort(idx_prime_)
        time_            = idx_prime_[idx_prime_prime_]
        energy_          = i_data_['MW'][idx_].to_numpy()[idx_prime_prime_]
        # Storage Data in dataframe
        o_data_[TACs_[i]] = energy_
    # Save energy demand in pickle file
    _save_CAISO_data_in_pickle_file(o_data_, path, resource, date)

# Save CAISO data in a pickle file
def _save_CAISO_data_in_pickle_file(data_, path, resource, date):
    file_name = path + resource + r'_' + date + r'.pkl'
    # Compose saving name
    with open(file_name, 'wb') as _f:
        pickle.dump(data_, _f, protocol = pickle.HIGHEST_PROTOCOL)


market = ["ACTUAL", "DAM"][forecasts]
path   = [path_to_oac, path_to_ofc][forecasts]

resources_ = ['Demand', 'Solar', 'Wind']
TACs_      = ['CA ISO-TAC', 'MWD-TAC', 'PGE-TAC', 'SCE-TAC', 'SDGE-TAC', 'VEA-TAC']
HUBs_      = ['NP15', 'SP15', 'ZP26']

# Retrive a year long data
_start_date = date(year, month, day)
_end_date   = date(year + 1, 1, 1)
N_days      = (_end_date - _start_date).days

# Timestamp
_delta = timedelta(days = 1)

# Loop over number of days
for i in range(N_days):

    # Get daylong data request
    _end_date = _start_date + _delta

    # Get date in API request format
    start_date = f"{_start_date.year:02}{_start_date.month:02}{_start_date.day:02}"
    end_date   = f"{_end_date.year:02}{_end_date.month:02}{_end_date.day:02}"

    flag = 0
    for resource in resources_:
        file_name = path + resource + '_' + start_date + r'.pkl'
        flag     += int(os.path.isfile(file_name))

    if flag < 3:
        print(_start_date, market)

        try:
            # Define renewable energy generation CAISO url API request
            renewable_url = r"http://oasis.caiso.com/oasisapi/SingleZip?queryname=SLD_REN_FCST&market_run_id={}&resultformat=6&startdatetime={}T07:00-0000&enddatetime={}T07:00-0000&version=1".format(market, start_date, end_date)
            # Define energy demand CAISO url API request
            demand_url = r"http://oasis.caiso.com/oasisapi/SingleZip?resultformat=6&queryname=SLD_FCST&version=1&market_run_id={}&startdatetime={}T07:00-0000&enddatetime={}T07:00-0000".format(market, start_date, end_date)
            # Download renewable generation data
            i_renewable_ = _download_csv(renewable_url, zip_file_name = 'temp.zip')
            # Download energy demand data
            i_demand_ = _download_csv(demand_url, zip_file_name = 'temp.zip')

            _process_renewable_energy_generation(i_renewable_, path, start_date, HUBs_)
            _process_energy_demand(i_demand_, path, start_date, TACs_)
        except:
            print('Error: ', start_date, market)

    else:
        print('Exits: ', start_date, market)

    # Got to next day
    _start_date += _delta

    sleep(6.)
