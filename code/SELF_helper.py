import os
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt

from params import SSD_DIR, CSV_DIR, SPEC_NR_SECONDS, \
    DATA_DIR, SPEC_PARAM_DIR, SSD_PARAM_DIR

def get_participant_list(aspect, condition):
    """ Function to get the list of participants available for the given
    aspect (data/ssd/spec_param) and condition (ec/eo)

    Parameters
    ----------
        aspect (str): 'data', 'ssd', 'sensor_param' or 'ssd_param'
        condition (str): 'ec' or 'eo'

    Returns
    -------
        subjects_selected (list): list of participants available for the given
                                   aspect and condition

    """

    # read participant list
    participants = pd.read_csv(f"{CSV_DIR}name_match.csv")
    subjects = participants['INDI_ID'].tolist()
    subjects = np.array(subjects)
    print(subjects)

    if aspect == 'data':
        data_exists = np.zeros((len(subjects),), dtype='bool')
        for i_sub, subject in enumerate(subjects):
            _fname = f'{DATA_DIR}/{subject}_{condition}.set'
            if os.path.exists(_fname):
                data_exists[i_sub] = True
        subjects_selected = subjects[np.where(data_exists)[0]].tolist()
    
    elif aspect == 'ssd':
        ssd_exists = np.zeros((len(subjects),), dtype='bool')
        for i_sub, subject in enumerate(subjects):
            ssd_fname = f"{SSD_DIR}/{subject}_ssd_filters_{condition}.csv"
            if os.path.exists(ssd_fname):
                ssd_exists[i_sub] = True
        subjects_selected = subjects[np.where(ssd_exists)[0]].tolist()
    
    elif aspect == 'sensor_param':
        sparam_exists = np.zeros((len(subjects),), dtype='bool')
        for i_sub, subject in enumerate(subjects):
            sparam_fname = f"{SPEC_PARAM_DIR}/{subject}_{condition}.csv"
            if os.path.exists(sparam_fname):
                sparam_exists[i_sub] = True
        subjects_selected = subjects[np.where(sparam_exists)[0]].tolist()

    elif aspect == 'ssd_param':
        ssd_param_exists = np.zeros((len(subjects),), dtype='bool')
        for i_sub, subject in enumerate(subjects):
            ssd_param_fname = f"{SSD_PARAM_DIR}/{subject}_{condition}.csv"
            if os.path.exists(ssd_param_fname):
                ssd_param_exists[i_sub] = True
        subjects_selected = subjects[np.where(ssd_param_exists)[0]].tolist()

    subjects = np.sort(subjects_selected)

    return subjects_selected