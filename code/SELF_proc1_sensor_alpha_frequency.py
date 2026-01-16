""" Data: Compute center frequency for each EEG subject."""

import os
import mne
import numpy as np
import pandas as pd
import fooof as f

from params import ALPHA_FMIN, DATA_DIR, RESULTS_DIR, \
    SPEC_FMIN, SPEC_FMAX, \
    SPEC_NR_SECONDS, SPEC_PARAM_DIR, \
    SPEC_NR_PEAKS, ALPHA_FMAX, ALPHA_FMIN
from SELF_helper import get_participant_list

def process_1sub(subject, condition):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SPEC_PARAM_DIR, exist_ok=True)
    spec_file = f"{SPEC_PARAM_DIR}/{subject}_{condition}.csv"

    file_name = f"{DATA_DIR}/{subject}_{condition}.set"
    data = mne.io.read_raw_eeglab(file_name, preload=True)
    data.set_eeg_reference("average")

    print("parte 1 done")

    if subject == "sub-032478":
        # this participant has potentially wrong-labeled channel names
        return
    
    # pick midline channels
    data.pick_types(eeg=True)
    midline_channels = [ch for ch in data.ch_names if "z" in ch]
    data.pick_channels(midline_channels)
    front_channels = [ch for ch in data.ch_names if "F" in ch]
    data.drop_channels(front_channels)

    # compute Power Spectra Density (PSD)
    psd, freqs = mne.time_frequency.psd_welch(
        data,
        fmin=SPEC_FMIN,
        fmax=SPEC_FMAX,
        n_fft=int(SPEC_NR_SECONDS * data.info["sfreq"]),
        n_overlap=data.info["sfreq"],
    )

    # fit spec param
    fm = f.FOOOFGroup(max_n_peaks=SPEC_NR_PEAKS)
    fm.fit(freqs, psd)
    alpha_bands = f.analysis.get_band_peak_fg(fm, [ALPHA_FMIN, ALPHA_FMAX])

    peak = np.nanmean(alpha_bands[:, 0])
    amp = np.nanmean(alpha_bands[:, 1])
    rsq = np.mean([fm.get_results()[i][2] for i in range(len(data.ch_names))]) # R squared of the fits

    # create dataframe with data
    df_subject = pd.Series(
        data={"subject": subject,
              "peak_frequency": peak,
              "peak_amplitude": amp,
              'rsq': rsq}
    )

    df_subject.to_csv(spec_file)

    return peak

if __name__ == "__main__":
    for condition in ['ec', 'eo']:
        print(f"Processing condition: {condition}")
        subjects = get_participant_list('data', condition)
        for i_sub, subject in enumerate(subjects):
            peak = process_1sub(subject, condition)
            print(subject, peak)
