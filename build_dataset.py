

import sys
import numpy as np
from obspy import UTCDateTime
from time import time

from aux_functions import download_data


start_time_string = sys.argv[1]
end_time_string = sys.argv[2]
base_url = sys.argv[3]
phase = sys.argv[4]

# start_time_string = '2017-01-01'
# end_time_string = '2017-01-03'
# base_url = 'NCEDC'
# phase = 'P'

data_path = 'data_' + phase.lower() + '/'




start_time = UTCDateTime(start_time_string)
end_time = UTCDateTime(end_time_string)

time_window = 60
request_all_channels = True
# base_url = 'NCEDC'

waves = []
times = []

stride = 86400
st = start_time

timer_start = time()

while st < end_time:

    et = st + stride
    try:
        w, t = download_data(st, et, time_window, request_all_channels, base_url, phase)
        waves = waves + w
        times = times + t
    except:
        pass
    st += stride

    

    elapsed_seconds = time() - timer_start
    elapsed_download = et - start_time
    remaining_download = end_time - et

    estimated_eta = remaining_download * elapsed_seconds / (elapsed_download + 0.000001)
    print(str(estimated_eta) + ' seconds remaining')


waves = np.asarray(waves, dtype=np.float32)
times = np.asarray(times, dtype=np.float32)

np.save(data_path + 'waves_' + base_url + '_' + start_time_string + '_' + end_time_string + '.npy', waves)
np.save(data_path + 'times_' + base_url + '_' + start_time_string + '_' + end_time_string + '.npy', times)
