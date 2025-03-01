import pickle, requests, zipfile, glob, os, sys

import numpy as np
import pandas as pd

from time import sleep, time
from datetime import datetime, timedelta

import geopandas as gpd
from shapely.geometry import Point

# Path to folder where ca_node_locations.csv file is
path_to_oax = r"/Users/Guille/Desktop/caiso_power/output/auxiliary/"
# Path to folder where file processed .pkl files are stored
path_to_lmp = r"/Users/Guille/Desktop/caiso_power/output/LMPs/"

path_to_map  = r"/Users/Guille/Desktop/extreme_scenarios/data/"
path_to_data = r"/Users/Guille/Desktop/caiso_power/data/"

_CAISO = gpd.read_file(path_to_data + r"maps/CAISO/Balancing_Authority_Areas_in_CA.shp")
_US    = gpd.read_file(path_to_map + r"tl_2022_us_state/tl_2022_us_state.shp")
print(_CAISO)

_CAISO_prime_ = _CAISO.iloc[[1, 7]].to_crs("EPSG:4326")

# California region to rank nodes with respect to
lat = 34.41
lon = -119.85
# Specify LMPs market
market = 'RTM'
# Time intervals in downloading batches
dt = 29

i_node = int(sys.argv[1])
# Request specific parameters
#year   = int(sys.argv[1])
#month  = int(sys.argv[2])
#i_node = int(sys.argv[3])
year_init  = 2020
month_init = 1
day_init   = 1

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

# Process LMPs request to the needed form
def _processing_data(INTVL_, data_):
    # Get only LMPs
    index_ = INTVL_['LMP_TYPE'] == 'LMP'
    # Get information associated with LMPs
    dates_     = INTVL_['OPR_DT'][index_].to_numpy()
    intervals_ = INTVL_['OPR_INTERVAL'][index_].to_numpy()
    values_    = INTVL_['VALUE'][index_].to_numpy()
    hours_     = INTVL_['OPR_HR'][index_].to_numpy()
    #print(intervals_.shape, values_.shape)
    # Loop over dates in the request
    for date in np.sort(np.unique(dates_)):
        # Select samples from a unique date
        idx_1_ = dates_ == date
        LMPs_ = []
        # Process LMPs by hour
        for hour in np.sort(np.unique(hours_[idx_1_])):
            # Get all data from that hour
            idx_2_ = hours_[idx_1_] == hour
            # Sort date by interval
            idx_3_ = np.argsort(intervals_[idx_1_][idx_2_])
            LMPs_.append(values_[idx_1_][idx_2_][idx_3_][:, np.newaxis])

        # Save only if all LMPs are available
        if len(LMPs_) == 24:
            data_[date] = np.concatenate(LMPs_, axis = 1)

    return data_

# Save CAISO data in a pickle file
def _save_CAISO_data_in_pickle_file(data_, file_name):
    # Compose saving name
    with open(file_name, 'wb') as _f:
        pickle.dump(data_, _f, protocol = pickle.HIGHEST_PROTOCOL)

bucket = r'http://oasis.caiso.com/oasisapi/SingleZip?resultformat=6'

data_    = pd.read_csv(path_to_oax + r'ca_node_locations.csv')
nodes_   = data_['node_id']
lat_     = data_['lat']
lon_     = data_['long']
regions_ = data_['region']
N_nodes  = data_.shape[0]
print('No. available nodes: ', N_nodes)

idx_ = []
# Loop over Grid coordinates
for i in range(nodes_.shape[0]):
    if _CAISO_prime_.contains(Point([lon_[i], lat_[i]])).any():
        idx_.append(i)
print(len(idx_))
nodes_   = nodes_[idx_].values.tolist()
lat_     = lat_[idx_].values.tolist()
lon_     = lon_[idx_].values.tolist()
regions_ = regions_[idx_].values.tolist()
# Select node
node   = nodes_[i_node]
lat    = lat_[i_node]
lon    = lon_[i_node]
region = regions_[i_node]
print(node, region, lat, lon)
# Define url request for real-time 5-minutes LMPs
query  = 'PRC_INTVL_LMP'
query  = r'&queryname={}&version=3'.format(query)
market = '&market_run_id={}&node={}'.format(market, node)

file_name = path_to_lmp + region + r'_' + node + r'.pkl'
print(file_name)

# Check if node-data exits for that date
if not os.path.isfile(file_name):
    # Start desired node dictionary
    data_         = {}
    data_['info'] = [node, region, lat, lon]
    year  = year_init
    month = month_init
    day   = day_init
else:
    data_ = pd.read_pickle(file_name)
    date  = list(data_.keys())[1]
    print('Exits from: ', date[:4], date[5:7], date[-2:])
    date  = list(data_.keys())[-1]
    year  = int(date[:4])
    month = int(date[5:7])
    day   = int(date[-2:])
    print('until: ', year, month, day)

_start_date = datetime(year, month, day)

N_requests = int((datetime.now() - _start_date).days/dt)
print('No. requests: ', N_requests)

for i in range(N_requests):
    print('Pulling request No.', i + 1)

    try:
        t_init = time()
        _end_date  = _start_date + timedelta(dt)
        # Define date interval for url requester
        start_date = f"{_start_date.year:02}{_start_date.month:02}{_start_date.day:02}T00"
        end_date   = f"{_end_date.year:02}{_end_date.month:02}{_end_date.day:02}T23"
        period     = r'&startdatetime={}:00-0000&enddatetime={}:00-0000&'.format(start_date, end_date)
        # Download real-time 5-minutes LMPs
        url    = bucket + query + period + market
        INTVL_ = _download_csv(url, zip_file_name = 'temp_{}.zip'.format(i_node))
        # Processing requested data
        data_ = _processing_data(INTVL_, data_)
        # Save requested processed data
        _save_CAISO_data_in_pickle_file(data_, file_name)
        # Go to the next request
        _start_date = _end_date
        sleep(5.)
        print(time() - t_init)
    except:
       print('Error: ', url)
