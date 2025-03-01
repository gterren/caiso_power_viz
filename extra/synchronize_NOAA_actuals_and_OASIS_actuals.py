import pickle, glob, os, sys, lzma, blosc, pytz, holidays

import numpy as np
import pandas as pd

from time import sleep
from datetime import datetime, date, timedelta

# Define input paths
path_to_data = r"/Users/Guille/Desktop/caiso_power/data/"
path_to_temp = r"/Users/Guille/Desktop/caiso_power/data/temp/"

# Define output paths
path_to_oac = r"/Users/Guille/Desktop/caiso_power/output/actuals/"
path_to_ofc = r"/Users/Guille/Desktop/caiso_power/output/forecasts/"
path_to_oax = r"/Users/Guille/Desktop/caiso_power/output/auxiliary/"
path_to_opd = r"/Users/Guille/Desktop/caiso_power/output/processed_data/"

year  = int(sys.argv[1])
month = int(sys.argv[2])
day   = int(sys.argv[3])

def _synchronize_NOAA_and_CAISO_data(X_, Y_, Z_):
    # Concatenate data in the batch
    #X_p_ = np.concatenate(X_, axis = -1)
    X_p_ = np.swapaxes(np.swapaxes(np.concatenate(X_, axis = -1)[:, 0, ...], 0, 2), 0, 1)
    Y_p_ = np.swapaxes(np.concatenate(Y_, axis = -1), 0, 1)
    Z_p_ = np.concatenate(Z_, axis = -1)
    print(X_p_.shape, Y_p_.shape, Z_p_.shape)
    N_hours = Y_p_.shape[1]
    # Reshape as a sequence of continues data
    X_pp_ = np.concatenate([X_p_[..., i] for i in range(X_p_.shape[-1])], axis = -1)
    Y_pp_ = np.concatenate([Y_p_[..., i] for i in range(Y_p_.shape[-1])], axis = -1)
    Z_pp_ = np.concatenate([np.array((Z_p_[0, i], Z_p_[1, i], Z_p_[2, i], j, Z_p_[3, i], Z_p_[4, i], Z_p_[5, i], Z_p_[6, i]))[..., np.newaxis]
                            for i in range(Z_p_.shape[-1]) for j in range(N_hours)], axis = 1)
    print(X_pp_.shape, Y_pp_.shape, Z_pp_.shape)
    # synchronize data from different timezones
    X_ppp_ = X_pp_[..., -dt_new:]
    Y_ppp_ = Y_pp_[:, :dt_new]
    Z_ppp_ = Z_pp_[:, :dt_new]
    print(X_ppp_.shape, Y_ppp_.shape, Z_ppp_.shape)
    # Remove remaining samples to have entire das
    N_samples = Y_ppp_.shape[-1]
    X_pppp_   = X_ppp_[..., :N_samples - (N_hours + dt_new)]
    Y_pppp_   = Y_ppp_[:, :N_samples - (N_hours + dt_new)]
    Z_pppp_   = Z_ppp_[:, :N_samples - (N_hours + dt_new)]
    print(X_pppp_.shape, Y_pppp_.shape, Z_pppp_.shape)
    # Reshape data back in days and day hours shape
    N_days   = X_pppp_.shape[-1]//N_hours
    X_ppppp_ = []
    Y_ppppp_ = []
    Z_ppppp_ = []
    for i in range(N_days):
        j = i*N_hours
        k = (i + 1)*N_hours
        X_ppppp_.append(X_pppp_[..., j:k][..., np.newaxis])
        Y_ppppp_.append(Y_pppp_[..., j:k][..., np.newaxis])
        Z_ppppp_.append(Z_pppp_[..., j:k][..., np.newaxis])
    # Define data in matrix form
    X_ppppp_ = np.concatenate(X_ppppp_, axis = -1)
    Y_ppppp_ = np.concatenate(Y_ppppp_, axis = -1)
    Z_ppppp_ = np.concatenate(Z_ppppp_, axis = -1)
    print(X_ppppp_.shape, Y_ppppp_.shape, Z_ppppp_.shape)
    return [X_ppppp_, Y_ppppp_, Z_ppppp_]

# Save chunk of processed date in a blosc file
def _save_data_in_compressed_files(data_, path, year, N_max_samples_per_chunk = 80):

    def __save_compressed_file(data_, path, year, chunk):
        dat_file_name = r'{}ac_{}_{}.dat'.format(path, year, chunk)
        with open(dat_file_name, "wb") as f:
            f.write(blosc.compress(pickle.dumps(data_)))
        print(dat_file_name)
    # Loop over chuncks
    for i in range(len(data_)):
        # Check samples in the chunk
        N_samples = data_[i][0].shape[-1]
        # Check if several chunks are necessary
        N_chunks = int(N_samples / N_max_samples_per_chunk) + 1
        # Define number of necessary chunks
        if N_chunks > 1: N_samples = int(np.ceil(N_samples/N_chunks))
        # Save Chunks
        for j in range(N_chunks):
            # Define samples in the j-th chunks
            k = j*N_samples
            l = (j + 1)*N_samples
            data_p_ = [data_[i][0][..., k:l], data_[i][1][..., k:l], data_[i][2][..., k:l]]

            __save_compressed_file(data_p_, path, year  = _ptz_date.year,
                                                  chunk = f"{i:01}-{j:01}")

# Remove the orginal files of the compressed files
def _remove_processed_files(_start_date, path, N_days):
    for day in range(N_days):
        time = f"{_start_date.year:02}{_start_date.month:02}{_start_date.day:02}"
        lzma_file_name = r'{}ac_{}.xz'.format(path, time)
        try:
            os.remove(lzma_file_name)
            print(lzma_file_name)
        except:
            pass

        _start_date += timedelta(days = 1)

# Retrive a year long data
_start_date = datetime(year, month, day)
_end_date   = datetime(year + 1, 1, 1)
N_days      = (_end_date - _start_date).days
# Defining the timezone
_utc = pytz.timezone('UTC')
_ptz = pytz.timezone('America/Los_Angeles')
# Define date in is corresponding timezone
_utc_date = _utc.localize(_start_date)
print(_utc_date)
# Define holidays calendar
_CA_holidays = holidays.country_holidays('US', subdiv = 'CA')
# Variables initialization
X_ = []
Y_ = []
Z_ = []
data_p_ = []
dt_prev = -8
flag    = False
# Loop over all possible days in a year
for day in range(N_days):
    # Check time difference between time zones
    _ptz_date = _utc_date.astimezone(_ptz)
    dt_new    = int(_ptz_date.utcoffset().total_seconds()/3600)

    # Rise flag if the number of hours is differe
    if dt_prev != dt_new:
        flag = True

    # Rise flag if files does not exits
    try:
        time = f"{_ptz_date.year:02}{_ptz_date.month:02}{_ptz_date.day:02}"
        lzma_file_name = r'{}ac_{}.xz'.format(path_to_opd, time)
        with lzma.open(lzma_file_name, "rb") as _f:
            data_ = pickle.load(_f)
        print(lzma_file_name)
        X_.append(data_[1][..., np.newaxis])
        Y_.append(data_[0][..., np.newaxis])
        Z_.append(np.array((_ptz_date.year, _ptz_date.month, _ptz_date.day, _ptz_date.timetuple().tm_yday,
                            _ptz_date.weekday(), (_ptz_date.timetuple().tm_wday >= 5)*1, int(_ptz_date in _CA_holidays)))[..., np.newaxis])
    except:
        flag = True
    # Generate a batch when rise flag and there is data
    if flag and (len(X_) > 2):
        # synchronize data
        data_p_.append(_synchronize_NOAA_and_CAISO_data(X_, Y_, Z_))
        # Initilize storage variables for next batch
        Y_ = []
        X_ = []
        Z_ = []
    # Reset flag
    flag = False
    # Go to next date
    dt_prev    = dt_new
    _utc_date += timedelta(days = 1)

_save_data_in_compressed_files(data_p_, path_to_opd, year = _ptz_date.year)
_remove_processed_files(_start_date, path_to_opd, N_days)
