import os
import mne
import pandas as pd
import numpy as np

from params import DATA_DIR, CSV_DIR, RAW_DIR, EPO_DIR

# Ensure directory to save epochs exists
os.makedirs(EPO_DIR, exist_ok=True)

# Read the CSV file containing subject information
df = pd.read_csv(CSV_DIR / "name_match.csv")
subjects = df.INDI_ID

# S200 EO, S210 EC
cond_list = {200: "EO", 210: "EC"}

for i_sub, subject in enumerate(subjects):
    print(f"Processing subject {i_sub}: {subject}")

    initial_name, bids_name = df.iloc[i_sub]

    # Route to .vhdr file
    file_type = "vhdr"
    vhdr_path = RAW_DIR / initial_name / "RSEEG" / f"{initial_name}.{file_type}"

    if not vhdr_path.exists():
        print(f"[WARNING] File not found: {vhdr_path}")
        continue

    # Skip subjects with preprocessed data
    already_done = True
    for trigger, cond_label in cond_list.items():
        out_fif = EPO_DIR / f"{subject}_{cond_label}-raw.fif"
        if not out_fif.exists():
            already_done = False
            break

    if already_done:
        print(f"Subject {subject} already processed. Skipping.")
        continue

    # Load raw EEG data
    raw = mne.io.read_raw_brainvision(vhdr_path, eog=['VEOG'])
    raw.load_data()
    raw.filter(0.5, None)

    # Extract events from annotations
    events, event_id = mne.events_from_annotations(raw)

    # Process each condition
    for trigger, cond_label in cond_list.items():

        out_fif = EPO_DIR / f"{subject}_{cond_label}-raw.fif"
        if out_fif.exists():
            continue

        # Seleccionamos solo los eventos de ese trigger (código 200 o 210)
        condA = events[events[:, 2] == trigger]

        if condA.size == 0:
            print(f"[WARNING] {subject}: no events with code {trigger} for {cond_label}, skip condition.")
            continue

        # Detectamos bloques separados por >15000 muestras
        idx, = np.where(np.diff(condA[:, 0]) > 15000)
        condB = np.vstack((condA[0], condA[idx + 1]))

        # Epocamos en tramos de 0 a 60 s
        epo = mne.Epochs(raw, condB, tmin=0, tmax=60, baseline=None)
        epo.load_data()

        # Si no hay épocas, saltamos
        if len(epo) == 0:
            print(f"[AVISO] {subject} {cond_label}: 0 épocas válidas, salto condición.")
            continue

        data = epo.get_data()                   # (n_epochs, n_channels, n_times)
        data = np.transpose(data, (1, 0, 2))    # (n_channels, n_epochs, n_times)
        data = data.reshape(data.shape[0], -1)  # (n_channels, n_epochs * n_times)

        # Si por cualquier razón no hay muestras, también saltamos
        if data.shape[1] == 0:
            print(f"[AVISO] {subject} {cond_label}: epochs vacíos (0 muestras), salto condición.")
            continue

        raw2 = mne.io.RawArray(data, raw.info)

        print(f"  Guardando {out_fif}")
        raw2.save(out_fif, overwrite=True)