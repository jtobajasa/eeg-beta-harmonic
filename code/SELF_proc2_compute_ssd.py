""" Data: compute SSD filters for all subjects and save them."""
import os
import mne
import numpy as np
import pandas as pd
import ssd

from helper import get_participant_list
from params import EPO_DIR, SSD_DIR, SPEC_PARAM_DIR, SSD_WIDTH, SNR_THRESHOLD, CSV_DIR

os.makedirs(SSD_DIR, exist_ok=True)

def process_1sub(subject, condition):
    # Output directories
    ssd_filters_fname = SSD_DIR / f"{subject}_ssd_filters_{condition}.csv"
    raw_ssd_file = SSD_DIR / f"{subject}_{condition}_raw.fif"
    ssd_patterns_fname = SSD_DIR / f"{subject}_ssd_patterns_{condition}.csv"

    # Read spec param
    spec_file = SPEC_PARAM_DIR / f"{subject}_{condition}.csv"
    df = pd.read_csv(spec_file, index_col=0)
    peak_alpha = df.T["peak_frequency"].to_numpy("float32")[0]
    peak_amp = df.T["peak_amplitude"].to_numpy("float32")[0]

    # Load data
    file_name = EPO_DIR / f"{subject}_{condition}-raw.fif"
    raw = mne.io.read_raw_fif(file_name, preload=True)
    raw.pick_types(eeg=True)
    raw.set_montage('standard_1020')

    if np.isnan(peak_alpha):
        # no alpha peak detected
        print(f"{subject} {condition}: no alpha peak detected, skipping SSD.")
        return
    
    if peak_amp < SNR_THRESHOLD:
        # low SNR, skip SSD
        print(f"{subject} {condition}: low SNR ({peak_amp:.3f}), skipping SSD.")
        return
    
    print(f"Running SSD for {subject} {condition} with peak at {peak_alpha:.2f} Hz")
    filters, patterns = ssd.run_ssd(raw, peak=peak_alpha, band_width=SSD_WIDTH)

    # Save filters and patterns as CSV
    df_filters = pd.DataFrame(filters.T, columns=raw.ch_names)
    df_filters.to_csv(ssd_filters_fname, index=False)

    df_patterns = pd.DataFrame(patterns.T, columns=raw.ch_names)
    df_patterns.to_csv(ssd_patterns_fname, index=False)

    # Create raw_ssd file with first components
    nr_components = 4
    raw_ssd = ssd.apply_filters(raw, filters[:, :nr_components])
    raw_ssd.save(raw_ssd_file, overwrite=True)

    return

if __name__ == "__main__":

    df_names = pd.read_csv(CSV_DIR / "name_match.csv")
    all_subjects = df_names.INDI_ID.to_list()

    for condition in ["EO", "EC"]:
        print(f"\n=== Condition: {condition} ===")

        valid_subjects = []
        for subject in all_subjects:
            spec_file = SPEC_PARAM_DIR / f"{subject}_{condition}.csv"
            if spec_file.exists():
                valid_subjects.append(subject)

        print(f" Subjects with sensor params for {condition}: {len(valid_subjects)}")

        # Opcional: solo para pruebas
        # valid_subjects = valid_subjects[:2]

        for subject in valid_subjects:
            print(f"Processing subject {subject}")
            process_1sub(subject, condition)
            print(f"Finished {subject} {condition}")