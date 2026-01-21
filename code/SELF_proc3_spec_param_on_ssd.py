import os
import mne
import numpy as np
import pandas as pd
import fooof
import matplotlib.pyplot as plt

from params import CSV_DIR, SSD_DIR, SPEC_NR_PEAKS, \
    SPEC_NR_SECONDS, SSD_PARAM_DIR, ALPHA_FMIN, ALPHA_FMAX, SNR_THRESHOLD
from SELF_helper import get_participant_list, percentile_spectrum

# Make sure output directory exists
os.makedirs(SSD_PARAM_DIR, exist_ok=True)

def process_1sub(subject, condition):
    "Spectral parametrization on SSD components to obtain alpha and beta peaks"

    spec_file = SSD_PARAM_DIR / f"{subject}_{condition}.csv"

    ssd_fname = SSD_DIR / f"{subject}_{condition}_raw.fif"
    raw_ssd = mne.io.read_raw_fif(ssd_fname, preload=True)

    band = (ALPHA_FMIN, ALPHA_FMAX)

    # Percentile spectrum on first SSD channel
    psd_perc, freq = percentile_spectrum(
        raw_ssd,
        band=band,
        nr_lines=4,
        i_chan=0,
        nr_seconds=SPEC_NR_SECONDS,
    )

    # pick higher percentile
    fm = fooof.FOOOF(max_n_peaks=SPEC_NR_PEAKS)
    fm.fit(freqs=freq, power_spectrum=psd_perc[0])

    # 1) Extract alpha peak
    peaks = fm.peak_params_ # shape: (n_peaks, 3) columns: CF, PW, BW

    if peaks.size == 0:
        # no peaks detected
        return
    
    # Filter peaks in alpha band
    mask_alpha = (peaks[:, 0] >= ALPHA_FMIN) & (peaks[:, 0] <= ALPHA_FMAX)
    
    sub_alpha = peaks[mask_alpha]
    idx_alpha = np.argmax(sub_alpha[:,1]) # index of max power in alpha band
    alpha_cf = sub_alpha[idx_alpha, 0]
    alpha_amp = sub_alpha[idx_alpha, 1]

    if alpha_amp < SNR_THRESHOLD:
        print(f"{subject} {condition}: low SNR ({alpha_amp:.3f}), skipping.")
        return
    
    # 2) Corrected spectrum for aperiodic (looking for maximum)
    psd_corr = fm.power_spectrum - fm._ap_fit

    # Find alpha frequency (max in alpha band)
    idx_start = np.argmin(np.abs(freq - ALPHA_FMIN))
    idx_end = np.argmin(np.abs(freq - ALPHA_FMAX))
    idx_max = np.argmax(psd_corr[idx_start:idx_end]) + idx_start
    alpha_peak = freq[idx_max]

    # Find beta frequency (2nd harmonic ~ 2*alpha_peak)
    idx_start_beta = np.argmin(np.abs(freq - 2*ALPHA_FMIN))
    idx_end_beta = np.argmin(np.abs(freq - 2*ALPHA_FMAX))
    idx_max_beta = np.argmax(psd_corr[idx_start_beta:idx_end_beta]) + idx_start_beta
    beta_peak = freq[idx_max_beta]

    # 3) Save results by subject
    df_subject = pd.Series(
        data={
            "subject": subject,
            "alpha_peak": alpha_peak,
            "beta_peak": beta_peak
        }
    )
    df_subject.to_csv(spec_file)
    plt.close('all')

if __name__ == "__main__":

    for condition in ['EO', 'EC']:
        print(f"\n=== Condition: {condition} ===")

        subjects_ssd = get_participant_list('ssd', condition)
        print(f"  Subjects with SSD for {condition}: {len(subjects_ssd)}")

        for subject in subjects_ssd:
            print(f"Processing subject {subject} - condition {condition}")
            process_1sub(subject, condition)
            print(f'Done subject {subject} - condition {condition}')

        # Complile all *_{condition}.csv
        subjects_param = get_participant_list('ssd_param', condition)
        print(f"  Subjects with SSD params for {condition}: {len(subjects_param)}")

        dfs = []
        for subject in subjects_param:
            spec_file = SSD_PARAM_DIR / f"{subject}_{condition}.csv"
            df = pd.read_csv(spec_file, index_col=0)
            dfs.append(df)

        if len(dfs) == 0:
            print(f"  No SSD param files found for {condition}, skipping group CSV.")
            continue

        df_all = pd.concat(dfs, axis=1).T
        df_all.columns = ("subject", "alpha_peak", "beta_peak")

        out_csv = CSV_DIR / f"ssd_param_all_{condition}.csv"
        df_all.to_csv(out_csv, index=False)
        print(f"  Saved group SSD params to {out_csv}")