
from collections import defaultdict
import numpy as np
from obspy.clients.fdsn import Client

def create_requests(catalog, time_window, phase, request_all_channels = False):

    requests = []
    event_times = []


    for event in catalog:

        origin = event.origins[0]

        requests_ = {}
        for pick in event.picks:

            if pick.evaluation_mode == 'manual' and pick.evaluation_status != 'preliminary' and pick.evaluation_status != 'rejected':
                nc = pick.waveform_id.network_code
                sc = pick.waveform_id.station_code
                lc = pick.waveform_id.location_code
                cc = pick.waveform_id.channel_code
                if not cc[0] in 'HECDFG':
                    continue
                if not cc[2] in 'ZNE':
                    continue

                if request_all_channels:
                    cc = cc[:-1] + '?'
                st = pick.time - time_window
                et = pick.time + time_window
                requests_[pick.resource_id] = (nc, sc, lc, cc, st, et)


        for arrival in origin.arrivals:

            if arrival.phase.lower() == phase.lower():
                if arrival.pick_id in requests_:
                    request = requests_[arrival.pick_id]
                    requests.append(request)

    return requests



def download_waves_(client, bulk_requests, time_window):
    
    waves = []
    times = []

    st = client.get_waveforms_bulk(bulk_requests)

    traces = []
    
    for trace in st.traces:
        if (trace.stats.sampling_rate >= 100) and (len(trace.data) == (trace.stats.sampling_rate * 2 * time_window)):
            traces.append(trace)

    traces.sort(key=lambda x:x.stats.starttime)
    bulk_requests.sort(key = lambda x: x[4])

    nested_dict = lambda: defaultdict(nested_dict)

    trace_tree = nested_dict()

    for trace in traces:
        if trace.stats.channel in trace_tree[trace.stats.network][trace.stats.station][trace.stats.location].keys():
            trace_tree[trace.stats.network][trace.stats.station][trace.stats.location][trace.stats.channel].append(trace) 
        else:
            trace_tree[trace.stats.network][trace.stats.station][trace.stats.location][trace.stats.channel] = [trace] 


    bulk_requests_ = []
    traces_ = []
    traces_by_channel = {
        'Z' : [],
        'N' : [],
        'E' : []
    }

    times_by_channel = {
        'Z' : [],
        'N' : [],
        'E' : []
    }


    for request in bulk_requests:

        channel_names = trace_tree[request[0]][request[1]][request[2]].keys()

        orientations = [cn[2] for cn in channel_names]

        if len(channel_names) != 3:
            continue

        if not 'Z' in orientations:
            continue

        channel_name_assign = {
            channel_names[0]: '',
            channel_names[1]: '',
            channel_names[2]: ''
        }

        for channel_name in channel_names:
            assigned = False
            if 'Z' == channel_name[-1]:
                channel_name_assign[channel_name] = 'Z'
                assigned = True
            elif 'N' == channel_name[-1]:
                channel_name_assign[channel_name] = 'N'
                assigned = True
            elif 'E' == channel_name[-1]:
                channel_name_assign[channel_name] = 'E'
                assigned = True
            elif '1' in orientations and '2' in orientations:
                if '1' == channel_name[-1]:
                    channel_name_assign[channel_name] = 'N'
                    assigned = True
                elif '2' == channel_name[-1]:
                    channel_name_assign[channel_name] = 'E'
                    assigned = True
            elif '2' in orientations and '3' in orientations:
                if '2' == channel_name[-1]:
                    channel_name_assign[channel_name] = 'N'
                    assigned = True
                elif '3' == channel_name[-1]:
                    channel_name_assign[channel_name] = 'E'
                    assigned = True
            
            if assigned == False:
                print(channel_names)

        traces_by_channel_ = {}
        times_by_channel_ = {}
        
        for channel_name in channel_names:

            for trace in trace_tree[request[0]][request[1]][request[2]][channel_name]:
                if abs(request[4] - trace.stats.starttime) < 0.1:
                    traces_by_channel_[channel_name_assign[channel_name]] = trace
                    times_by_channel_[channel_name_assign[channel_name]]  = np.argmin(abs(trace.times(reftime=(request[4] + time_window))))

        if len(traces_by_channel_.keys()) == 3:
            for channel_name in channel_names:
                traces_by_channel[channel_name_assign[channel_name]].append(traces_by_channel_[channel_name_assign[channel_name]])
                times_by_channel[channel_name_assign[channel_name]].append(times_by_channel_[channel_name_assign[channel_name]])


    for i in range(len(traces_by_channel['Z'])):
        if i >= len(traces_by_channel['Z']) or i >= len(traces_by_channel['N']) or i >= len(traces_by_channel['E']):
            break


        time_threshold = 0.0001
        d1 = abs(traces_by_channel['Z'][i].stats.starttime - traces_by_channel['N'][i].stats.starttime)
        d2 = abs(traces_by_channel['Z'][i].stats.starttime - traces_by_channel['E'][i].stats.starttime)
        d3 = abs(traces_by_channel['N'][i].stats.starttime - traces_by_channel['E'][i].stats.starttime)

        if d1 < time_threshold and d2 < time_threshold and d3 < time_threshold:

            data_z = traces_by_channel['Z'][i].data.copy()
            data_n = traces_by_channel['N'][i].data.copy()
            data_e = traces_by_channel['E'][i].data
            arrival_time = times_by_channel['Z'][i]


            if len(data_z) > (100 * 2 * time_window):


                arrival_time = arrival_time * float((100 * 2 * time_window)) / float(len(data_z))
                
                sampled_coordinates = np.linspace(0, len(data_z) - 1, 100 * 2 * time_window)
                data_z = np.interp(sampled_coordinates, range(len(data_z)), data_z)
                data_n = np.interp(sampled_coordinates, range(len(data_n)), data_n)
                data_e = np.interp(sampled_coordinates, range(len(data_e)), data_e)

                sampled_coordinates = sampled_coordinates.astype(np.float32)
                data_z = data_z.astype(np.float32)
                data_n = data_n.astype(np.float32)
                data_e = data_e.astype(np.float32)

                

            waves.append([data_z, data_n, data_e])
            times.append(arrival_time)


    return waves, times    

def download_waves(client, bulk_requests, time_window, batch_size = 1000):
    
    waves = []
    times = []

    for i in range(0, len(bulk_requests), batch_size):
        w, t = download_waves_(client, bulk_requests[i:i + batch_size], time_window)
        waves = waves + w
        times = times + t

    return waves,times


def download_data(start_time, end_time, time_window, request_all_channels, base_url, phase):

    client = Client(base_url)

    cat = client.get_events(starttime=start_time, endtime=end_time, includearrivals=True)

    bulk_requests = create_requests(cat, time_window, phase, request_all_channels=request_all_channels)


    waves, times = download_waves(client, bulk_requests, time_window)

    return waves,times

