import numpy as np
from utils.svoboda_data_utils import *
import pandas as pd
from pynwb import NWBHDF5IO
import pickle
import numpy as np
import glob

#%% load session utils
def load_session_data(session_ind, trial_type_dict):
    '''
    now we load all trials
    Note that the loaded/save data contain neurons from all recorded areas, including those not in the area of interest

    return:

    shape of returned data:
    spike_data: N x T x K
    behavior_data: K x T x D
    area_ind_list: N 
    trial_type: K

    '''
    files = sorted(glob.glob('/work/hdd/bdye/jxia4/data/loaded_before_unbalanced_is_ALM_is_left/session_ind_*.pickle'))
    session_ind_list = [int(file.split('_')[-1].split('.')[0]) for file in files]
    if session_ind in session_ind_list:
        with open(files[session_ind_list.index(session_ind)], 'rb') as f:
            return pickle.load(f)

    path = '/work/hdd/bdye/jxia4/data/tables_and_infos/'

    session_info = pd.read_csv(path + 'session_info.csv')
    file_name = session_info['session_name'][session_ind]
    print(session_info.iloc[session_ind])

    if session_info.iloc[session_ind]['opto_onset']>0.5:
        opto_timing = 'late'
    else:
        opto_timing = 'early'


    io = NWBHDF5IO(path + file_name, mode="r")
    nwbfile = io.read()

    # get the spike train
    spike_data_combine_all, n_trial_all, licks_dict, behavior_dict, units, ids_sort_by_area, opto_type_sorted_dict, sorted_area, area_value, area_value_dict, is_left, is_ALM = main_get_spike_trains(nwbfile)
    
    spike_data_unbalanced = []
    trial_type_unbalanced = []

    for type in ['hit', 'miss', 'ignore']:

        if (n_trial_all[type]['left'] == 0) and (n_trial_all[type]['right'] == 0):
            continue

        spike_data = spike_data_combine_all[type]
        trial_type_ind = [trial_type_dict[type + '_left']] * n_trial_all[type]['left'] + [trial_type_dict[type + '_right']] * n_trial_all[type]['right']
        spike_data_unbalanced.append(spike_data)
        trial_type_unbalanced.append(trial_type_ind)


    for type in ['opto_hit', 'opto_miss', 'opto_ignore']:
        
        if (n_trial_all[type]['left'] == 0) and (n_trial_all[type]['right'] == 0):
            continue
        
        spike_data = spike_data_combine_all[type]
        spike_data_unbalanced.append(spike_data)

        trial_outcome = type.split('_')[1] # type is hit, miss, or ignore
        opto_trial_type = opto_type_sorted_dict[trial_outcome] # a list of np array with 1 element

        for opto_trial in opto_trial_type[:n_trial_all[type]['left']]:
            trial_type_name = 'opto_' + opto_timing + '_' + str(opto_trial[0]) + '_' + trial_outcome + '_left'
            trial_type_unbalanced.append([trial_type_dict[trial_type_name]])

        for opto_trial in opto_trial_type[n_trial_all[type]['left']:]:
            trial_type_name = 'opto_' + opto_timing + '_' + str(opto_trial[0]) + '_' + trial_outcome + '_right'
            trial_type_unbalanced.append([trial_type_dict[trial_type_name]])
            
    spike_data_unbalanced = np.concatenate(spike_data_unbalanced, axis=2)
    trial_type_unbalanced = np.concatenate(trial_type_unbalanced)
    
    #load region_info_summary.pkl
    with open(path + 'region_info_summary.pkl', 'rb') as f:
        [brain_region_list, session_by_region, session_by_region_n, junk] = pickle.load(f)

    # turn the area_value (area ind created by sorting within a session) into a list of area index (consistent across sessions)
    area_value = np.array(area_value)
    area_ind_list = np.zeros_like(area_value)
    for area, value in area_value_dict.items():
        area_ind = np.where(brain_region_list == area)[0][0]
        area_ind_list[area_value==value] = area_ind

    # get behavior data as K x T X D
    behavior_data_concat_dict = {'jaw': [], 'nose': [], 'tongue': []}

    for name in ['jaw', 'nose', 'tongue']:
        for trial_outcome in ['hit', 'miss', 'ignore', 'opto_hit', 'opto_miss', 'opto_ignore']:
            for side in ['left', 'right']:
                if n_trial_all[trial_outcome][side] == 0:
                    continue
                behavior_data_concat_dict[name].append(behavior_dict[trial_outcome][side][name])
        
        behavior_data_concat_dict[name] = np.concatenate(behavior_data_concat_dict[name], axis=0)

    behavior_unbalanced = np.concatenate((behavior_data_concat_dict['jaw'], behavior_data_concat_dict['nose'], behavior_data_concat_dict['tongue']), axis=2)

    with open('/work/hdd/bdye/jxia4/data/loaded_before_unbalanced_is_ALM_is_left/session_ind_{}.pickle'.format(session_ind), 'wb') as f:
        pickle.dump([spike_data_unbalanced, behavior_unbalanced, area_ind_list, is_left, is_ALM, trial_type_unbalanced], f)

    print('shape of spike_data_unbalanced: ')
    print(spike_data_unbalanced.shape)
    
    print('shape of behavior_unbalanced: ')
    print(behavior_unbalanced.shape)

    print('shape of trial_type_unbalanced: ')
    print(trial_type_unbalanced.shape)

    return spike_data_unbalanced, behavior_unbalanced, area_ind_list, is_left, is_ALM, trial_type_unbalanced


#%% lump area utils
def lump_area_with_nameoi(nameoi, brain_region_list, area_ind_list):
    '''
    nameoi: a phrase that is used to identify list of areas
    brain_region_list: a list of brain regions
    area_ind_list: a list of area indices for each neuron

    return:
    area_ind_lump_list: a list of area indices to be lumped
    area_ind_list_updated: a list of updated area indices for each neuron
    '''
    area_ind_list_updated = area_ind_list.copy()
    area_ind_lump_list = [] 
    for area_ind, area_name in enumerate(brain_region_list):
        if nameoi in area_name:
            area_ind_lump_list.append(area_ind)
            area_ind_list_updated[area_ind_list==area_ind] = area_ind_lump_list[0]

    return area_ind_lump_list, area_ind_list_updated


def lump_area_with_area_name_list(area_name_list, brain_region_list, area_ind_list):
    '''
    area_name_list: a list of area names to be lumped
    brain_region_list: a list of brain regions
    area_ind_list: a list of area indices for each neuron

    return:
    area_ind_lump_list: a list of area indices to be lumped
    area_ind_list_updated: a list of updated area indices for each neuron
    '''
    area_ind_list_updated = area_ind_list.copy()
    area_ind_lump_list = [] 
    
    for area_name in area_name_list:
        #print(area_name)
        area_ind = np.where(brain_region_list == area_name)[0][0]
        area_ind_lump_list.append(area_ind)
        area_ind_list_updated[area_ind_list==area_ind] = area_ind_lump_list[0]

    return area_ind_lump_list, area_ind_list_updated



def update_area_ind_list_and_areaoi_ind(area_ind_list, areaoi, brain_region_list, is_ALM):
    areaoi_ind = np.zeros((len(areaoi),), dtype=int)
    area_ind_list_updated = area_ind_list.copy()
    
    #non-ALM
    nameoi = 'motor area'
    area_ind_lump_list_non_ALM, area_ind_list_updated = lump_area_with_nameoi(nameoi, brain_region_list, area_ind_list)
    areaoi_ind[0] = area_ind_lump_list_non_ALM[0]

    #ALM
    area_ind = len(brain_region_list)-1
    area_ind_list_updated[is_ALM==1] = area_ind
    areaoi_ind[1] = area_ind

    #Orbital area
    nameoi = 'Orbital area'
    area_ind_lump_list_orbital, area_ind_list_updated = lump_area_with_nameoi(nameoi, brain_region_list, area_ind_list_updated)
    areaoi_ind[2] = area_ind_lump_list_orbital[0]
    
    #Thalamus
    area_name_list = ['Lateral dorsal nucleus of thalamus',
                      'Mediodorsal nucleus of thalamus',
                      'Central lateral nucleus of the thalamus',
                      'Central medial nucleus of the thalamus',
                      'Paracentral nucleus',
                      'Parafascicular nucleus',
                      'Lateral posterior nucleus of the thalamus',
                      'Posterior complex of the thalamus',
                      'Reticular nucleus of the thalamus',
                      'Ventral anterior-lateral complex of the thalamus',
                      'Ventral medial nucleus of the thalamus',
                      'Ventral posterolateral nucleus of the thalamus',
                      'Ventral posterolateral nucleus of the thalamus, parvicellular part',
                      'Ventral posteromedial nucleus of the thalamus',
                      'Ventral posteromedial nucleus of the thalamus, parvicellular part']

    area_ind_lump_list_thalamus, area_ind_list_updated = lump_area_with_area_name_list(area_name_list, brain_region_list, area_ind_list_updated)
    areaoi_ind[3] = area_ind_lump_list_thalamus[0]

    #Striatum
    area_name_list = ['Striatum', 'Caudoputamen']
    area_ind_lump_list_striatum, area_ind_list_updated = lump_area_with_area_name_list(area_name_list, brain_region_list, area_ind_list_updated)
    areaoi_ind[4] = area_ind_lump_list_striatum[0]

    #Pallidum
    area_name_list = ['Pallidum', 'Globus pallidus, external segment']
    area_ind_lump_list_pallidum, area_ind_list_updated = lump_area_with_area_name_list(area_name_list, brain_region_list, area_ind_list_updated)
    areaoi_ind[5] = area_ind_lump_list_pallidum[0]

    #Midbrain
    areaoi_ind[6] = np.where(brain_region_list == 'Midbrain')[0][0]

    #Midbrain reticular nucleus
    areaoi_ind[7] = np.where(brain_region_list == 'Midbrain reticular nucleus')[0][0]

    #Pedunculopontine nucleus
    areaoi_ind[8] = np.where(brain_region_list == 'Pedunculopontine nucleus')[0][0]

    #Substantia nigra, reticular part
    areaoi_ind[9] = np.where(brain_region_list == 'Substantia nigra, reticular part')[0][0]

    #Superior colliculus, motor related
    nameoi = 'Superior colliculus, motor related'
    area_ind_lump_list_SCm, area_ind_list_updated = lump_area_with_nameoi(nameoi, brain_region_list, area_ind_list_updated)
    areaoi_ind[10] = area_ind_lump_list_SCm[0]

    #Pons
    areaoi_ind[11] = np.where(brain_region_list == 'Pontine reticular nucleus')[0][0]

    #Cerebellar nuclei
    area_name_list = ['Fastigial nucleus',
                      'Interposed nucleus']

    area_ind_lump_list_CbN, area_ind_list_updated = lump_area_with_area_name_list(area_name_list, brain_region_list, area_ind_list_updated)
    areaoi_ind[12] = area_ind_lump_list_CbN[0]

    #Cerebellar cortex
    area_name_list = ['Lingula (I)',
                      'Lobule II',
                      'Lobule III',
                      'Lobules IV-V',
                      'Nodulus (X)',
                      'Paramedian lobule',
                      'Pyramus (VIII)',
                      'Uvula (IX)',
                      'Simple lobule']
    area_ind_lump_list_CbC, area_ind_list_updated = lump_area_with_area_name_list(area_name_list, brain_region_list, area_ind_list_updated)
    areaoi_ind[13] = area_ind_lump_list_CbC[0]

    #Medulla
    area_name_list = ['Facial motor nucleus',
                      'Gigantocellular reticular nucleus',
                      'Hypoglossal nucleus',
                      'Intermediate reticular nucleus',
                      'Lateral reticular nucleus, magnocellular part',
                      'Magnocellular reticular nucleus',
                      'Medulla',
                      'Medullary reticular nucleus, ventral part',
                      'Medullary reticular nucleus, dorsal part',
                      'Paragigantocellular reticular nucleus, lateral part',
                      'Parvicellular reticular nucleus']

    area_ind_lump_list_Medulla, area_ind_list_updated = lump_area_with_area_name_list(area_name_list, brain_region_list, area_ind_list_updated)
    areaoi_ind[14] = area_ind_lump_list_Medulla[0]

    return areaoi_ind, area_ind_list_updated, [area_ind_lump_list_non_ALM, area_ind_lump_list_orbital, area_ind_lump_list_thalamus, area_ind_lump_list_striatum, area_ind_lump_list_pallidum, area_ind_lump_list_SCm, area_ind_lump_list_CbN, area_ind_lump_list_CbC, area_ind_lump_list_Medulla]