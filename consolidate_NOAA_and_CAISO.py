import pickle, holidays, pytz, xarray, mgzip, glob, os, lzma, blosc, sys

import numpy as np
import pandas as pd

from time import sleep
from datetime import datetime, date, timedelta

# Define output paths
path_to_oac = r"/Users/Guille/Desktop/caiso_power/output/actuals/"
path_to_ofc = r"/Users/Guille/Desktop/caiso_power/output/forecasts/"
path_to_oax = r"/Users/Guille/Desktop/caiso_power/output/auxiliary/"
path_to_opd = r"/Users/Guille/Desktop/caiso_power/output/processed_data/"

year = int(sys.argv[1])

def _load_pickle_file(pickle_file_name):
    with open(pickle_file_name, 'rb') as _f:
        data_ = pickle.load(_f)
    return data_

# Save chunk of processed date in a blosc file
def _save_data_in_compressed_files(data_, path, year, N_max_samples_per_chunk = 900):

    def __save_compressed_file(data_, path, year, chunk):
        dat_file_name = r'{}{}_{}.dat'.format(path, year, chunk)
        with open(dat_file_name, "wb") as f:
            f.write(blosc.compress(pickle.dumps(data_)))
        print(dat_file_name)

    # Loop over chuncks
    for i in range(len(data_)):
        # Check samples in the chunk
        N_samples = data_[i][0].shape[0]
        # Check if several chunks are necessary
        N_chunks = int(N_samples / N_max_samples_per_chunk) + 1
        # Define number of necessary chunks
        if N_chunks > 1: N_samples = int(np.ceil(N_samples/N_chunks))
        # Save Chunks
        for j in range(N_chunks):
            # Define samples in the j-th chunks
            k = j*N_samples
            l = (j + 1)*N_samples
            data_p_ = [data_[i][0][k:l, ...], data_[i][1][k:l, ...],
                       data_[i][2][k:l, ...], data_[i][3][k:l, ...], data_[i][4][k:l, ...]]

            __save_compressed_file(data_p_, path, year  = _ptz_date.year,
                                                  chunk = f"{i:01}-{j:01}")

# Retrive a year long data
_start_date = datetime(year, 1, 1)
_end_date   = datetime(year + 1, 1, 1)
# Define Resources to loop over
resources_ = ['Demand', 'Solar', 'Wind']
# Get daylong data request
_delta = timedelta(days = 1)
# Defining the timezone
_utc = pytz.timezone('UTC')
_ptz = pytz.timezone('America/Los_Angeles')
# Define date in is corresponding timezone
_utc_date = _utc.localize(_start_date)
# Define holidays calendar
_CA_holidays = holidays.country_holidays('US', subdiv = 'CA')
# Initialize list of sequential data
all_CAISO_ac_ = []
all_CAISO_fc_ = []
all_NOAA_ac_  = []
all_NOAA_fc_  = []
time_pfz_     = []
# List of all processed data
all_data_ = []

dt_prev = -8
flag    = False
# Loop over number of days
for i in range((_end_date - _start_date).days):
    # Get date in API request format
    time = f"{_utc_date.year:02}{_utc_date.month:02}{_utc_date.day:02}"
    # Check time difference between time zones
    _ptz_date = _utc_date.astimezone(_ptz)
    dt_new    = int(_ptz_date.utcoffset().total_seconds()/3600)
    # Rise flag if the number of hours is differe
    if dt_prev != dt_new:
        flag = True

    else:
        try:
            # Load CAISO data looping Over type of resource
            CAISO_ac_ = np.concatenate([_load_pickle_file(path_to_oac + f"{resources_[j]}_{time}.pkl").to_numpy()
                         for j in range(len(resources_))], axis = 1)[:, 1:]
            CAISO_fc_ = np.concatenate([_load_pickle_file(path_to_ofc + f"{resources_[j]}_{time}.pkl").to_numpy()
                         for j in range(len(resources_))], axis = 1)[:, 1:]
            # Load NOAA data
            NOAA_ac_ = _load_pickle_file(path_to_oac + f"Weather_{time}.pkl")[:, 0, ...]
            NOAA_fc_ = _load_pickle_file(path_to_ofc + f"Weather_{time}-00.pkl")[0, ...][7:31, ...]
            # Save consecutive data
            all_CAISO_ac_.append(CAISO_ac_)
            all_CAISO_fc_.append(CAISO_fc_)
            all_NOAA_ac_.append(NOAA_ac_)
            all_NOAA_fc_.append(NOAA_fc_)
            # Define tempora information
            time_pfz_.append(np.array([[_ptz_date.year, _ptz_date.month, _ptz_date.day, _ptz_date.timetuple().tm_yday,
                                        (_ptz_date + timedelta(days = hour/24)).hour, _ptz_date.weekday(), (_ptz_date.timetuple().tm_wday >= 5)*1,
                                        _ptz_date.timetuple().tm_isdst*1, int(_ptz_date in _CA_holidays)]for hour in range(24)]))
        except:
            print('Missing file in ', time)
            flag = True

    # Generate a batch when rise flag and there is data
    if flag and (len(all_CAISO_fc_) > 2):
        # Data in matrix form
        all_CAISO_ac_ = np.concatenate(all_CAISO_ac_, axis = 0)[:dt_prev, ...][:-24 - dt_prev, ...]
        all_CAISO_fc_ = np.concatenate(all_CAISO_fc_, axis = 0)[:dt_prev, ...][:-24 - dt_prev, ...]
        all_NOAA_ac_  = np.concatenate(all_NOAA_ac_, axis = 0)[-dt_prev:, ...][:-24 - dt_prev, ...]
        all_NOAA_fc_  = np.concatenate(all_NOAA_fc_, axis = 0)[:-24, ...]
        time_pfz_     = np.concatenate(time_pfz_, axis = 0)[:dt_prev, :][:-24 -dt_prev, ...]
        # Correct error in the GSI computing in NOAA API
        GSI_         = np.concatenate((all_NOAA_fc_[1:, -1, :], np.zeros((1, all_NOAA_fc_.shape[-1]))), axis = 0)
        all_NOAA_fc_ = np.concatenate((all_NOAA_fc_[:, :-1, :], GSI_[:, np.newaxis, :]), axis = 1)
        # Save together all consecutive chu of data
        all_data_.append([all_CAISO_ac_, all_CAISO_fc_, all_NOAA_ac_, all_NOAA_fc_, time_pfz_])
        # Restart sample lists
        all_CAISO_ac_ = []
        all_CAISO_fc_ = []
        all_NOAA_ac_  = []
        all_NOAA_fc_  = []
        time_pfz_     = []
    # Reset flag
    flag = False
    # Go to next date
    dt_prev    = dt_new
    _utc_date += _delta

_save_data_in_compressed_files(all_data_, path_to_opd, year = _ptz_date.year)
