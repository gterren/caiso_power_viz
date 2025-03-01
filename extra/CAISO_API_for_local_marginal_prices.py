import pickle, os, requests, tempfile, zipfile, glob, os, sys

import numpy as np
import pandas as pd

from time import sleep
from datetime import datetime, date, timedelta

# Define input paths
path_to_temp = r"/Users/Guille/Desktop/caiso_power/output/temp/"

# Define output paths
path_to_oax = r"/Users/Guille/Desktop/caiso_power/output/auxiliary/"
path_to_lmp = r"/Users/Guille/Desktop/caiso_power/output/LMPs/"

year   = int(sys.argv[1])
month  = int(sys.argv[2])
day    = int(sys.argv[3])
i_node = int(sys.argv[4])

# California region
lat = 34.41
lon = -119.85

# GMT: Greenwich Mean Time
def _download_csv(url, file_name):
    zip_file_name = file_name + '.zip'
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

# Processing LMP data
def _processing_LMPs(data_):
    # Find node names in the csv
    index_     = data_['LMP_TYPE'] == 'LMP'
    intervals_ = data_['OPR_INTERVAL'][index_].to_numpy()
    values_    = data_['VALUE'][index_].to_numpy()
    return np.array([values_[i] for i in np.argsort(intervals_)])

# Save CAISO data in a pickle file
def _save_CAISO_data_in_pickle_file(data_, file_name):
    # Compose saving name
    with open(file_name, 'wb') as _f:
        pickle.dump(data_, _f, protocol = pickle.HIGHEST_PROTOCOL)


bucket = r'http://oasis.caiso.com/oasisapi/SingleZip?'

data_    = pd.read_csv(path_to_oax + r'ca_node_locations.csv')
nodes_   = data_['node_id']
lat_     = data_['lat']
lon_     = data_['long']
regions_ = data_['region']


market = 'RTM'
query  = 'PRC_INTVL_LMP'
url_1  = r'resultformat=6&queryname={}&'.format(query)

# Retrive a year long data
_start_date = date(year, month, day)
_end_date   = date(2022, 11, 1)
N_days      = (_end_date - _start_date).days
print(_start_date)

# Timestamp
_delta = timedelta(days = 1/24)

# Select ranked node by distnace
idx  = np.argsort((lat_ - lat)**2 + (lon_ - lon)**2)[i_node]
node = nodes_[idx]
lat  = lat_[idx]
lon  = lon_[idx]
print(node, lat, lon)

# Loop over number of days
for i in range(N_days):

    # Get date in API request format
    start_date  = f"{_start_date.year:02}{_start_date.month:02}{_start_date.day:02}"
    _start_date = datetime.strptime(start_date,"%Y%m%d").replace(hour = 0)

    # Define saving name
    file_name = path_to_lmp + market + r'_{}'.format(node) +  r'_({},{})_'.format(lat, lon) + start_date + r'.pkl'
    print(file_name)
    # Check if node-data exits for that date
    if not os.path.isfile(file_name):
        print(_start_date, market)

        try:
            # Initilize storage variable
            data_p_ = []
            for i in range(24):
                # Get hourlong data request
                _end_date = _start_date + _delta
                # Define date interval for url requester
                start_date = f"{_start_date.year:02}{_start_date.month:02}{_start_date.day:02}T{_start_date.hour:02}"
                end_date   = f"{_end_date.year:02}{_end_date.month:02}{_end_date.day:02}T{_end_date.hour:02}"
                # Define url request for real-time 5-minutes LMPs
                url_2 = r'version=3&startdatetime={}:00-0000&enddatetime={}:00-0000&'.format(start_date, end_date)
                # Define url request for real-time 5-minutes LMPs
                url = url_1 + url_2 + r'market_run_id={}&node={}'.format(market, node)
                # Download real-time 5-minutes LMPs
                INTVL_ = _download_csv(bucket + url, file_name = '{}-temp'.format(year))
                data_  = _processing_LMPs(INTVL_)
                data_p_.append(data_[..., np.newaxis])
                # Got to next day
                _start_date += _delta

                sleep(5.)
            # Dump Downloaded node-data into a pickle file
            _save_CAISO_data_in_pickle_file(np.concatenate(data_p_, axis = -1), file_name)

        except:
            print('Missing:', start_date, node)
            sleep(5.)

            start_date  = f"{_start_date.year:02}{_start_date.month:02}{_start_date.day:02}"
            _start_date = datetime.strptime(start_date,"%Y%m%d").replace(hour = 0) + timedelta(days = 1)

    else:
        print('Exits: ', start_date, market)

        # Got to next day
        _start_date += timedelta(days = 1)
