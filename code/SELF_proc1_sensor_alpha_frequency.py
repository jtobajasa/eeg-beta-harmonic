""" Data: compute center frequency for each EEG subject"""
import os
import mne
import numpy as np
import pandas as pd
import fooof

from params import EPO_DIR, \
    SPEC_FMIN, SPEC_FMAX, \
    SPEC_NR_SECONDS, SPEC_PARAM_DIR, \
    SPEC_NR_PEAKS, ALPHA_FMAX, ALPHA_FMIN
from SELF_helper import get_participant_list

def process_1sub(subject, condition):
    # Make sure output directory exists
    os.makedirs(SPEC_PARAM_DIR, exist_ok=True)
    spec_file = SPEC_PARAM_DIR / f"{subject}_{condition}.csv"

    # Load raw data
    file_name = EPO_DIR / f"{subject}_{condition}-raw.fif"
    raw = mne.io.read_raw_fif(file_name, preload=True)
    raw.set_eeg_reference('average')

    if subject == "sub-032478":
        # this participant has potentially wrong-labeled channel names
        return
    
    # pick midline channels
    raw.pick_types(eeg=True)
    midline_channels = [ch for ch in raw.ch_names if 'z' in ch]
    raw.pick_channels(midline_channels)
    front_channels = [ch for ch in raw.ch_names if 'F' in ch]
    raw.drop_channels(front_channels)

    # compute Power Spectra Density (PSD)
    psd = raw.compute_psd(
        fmin=SPEC_FMIN,
        fmax=SPEC_FMAX,
        n_fft=int(SPEC_NR_SECONDS * raw.info['sfreq']),
        n_overlap=int(raw.info['sfreq']),
        method='welch',
    )
    psd_data, freqs = psd.get_data(return_freqs=True)

    # fit spec param
    fm = fooof.FOOOFGroup(max_n_peaks=SPEC_NR_PEAKS)
    fm.fit(freqs, psd_data)

    alpha_cfs = []
    alpha_pws = []
    rsq_list = []

    for res in fm.group_results:
        peaks = res.peak_params
        rsq_list.append(res.r_squared)

        if peaks.size == 0:
            # no peaks detected
            alpha_cfs.append(np.nan)
            alpha_pws.append(np.nan)
            continue

        # Select peaks within alpha band
        mask = (peaks[:,0] >= ALPHA_FMIN) & (peaks[:,0] <= ALPHA_FMAX)

        if not np.any(mask):
            # no alpha peak detected
            alpha_cfs.append(np.nan)
            alpha_pws.append(np.nan)
        else:
            sub = peaks[mask] # select only alpha peaks
            idx = np.argmax(sub[:,1]) # index of max amplitude peak
            alpha_cfs.append(sub[idx,0]) # center frequency
            alpha_pws.append(sub[idx,1]) # peak amplitude
        
    # average over channels
    peak = np.nanmean(alpha_cfs)
    amp = np.nanmean(alpha_pws)
    rsq = np.nanmean(rsq_list)

    # create dataframe with data
    df_subject = pd.Series(
        data={
            'subject': subject,
            'peak_frequency': peak,
            'peak_amplitude': amp,
            'rsq': rsq
        }
    )
    df_subject.to_csv(spec_file)

    return peak

if __name__ == "__main__":

    for condition in ['EO', 'EC']:
        subjects = get_participant_list('data', condition)
        for subject in subjects:
            print(f"Processing subject {subject} - condition {condition}")
            peak = process_1sub(subject, condition)
            print(f'Done subject {subject} - condition {condition}')
            print(f'Peak frequency: {peak} Hz')