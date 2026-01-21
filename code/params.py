from pathlib import Path

# parameters and paths
DATA_DIR = Path.home() / "Documents" / "data_resting"
BASE_DIR = Path.home() / "Documents" / "GitHub" / "eeg-beta-harmonic"

RAW_DIR = DATA_DIR / "raw"
CSV_DIR = BASE_DIR / "csv"
RESULTS_DIR = DATA_DIR / "results/"

EPO_DIR = DATA_DIR / "epo_from_raw"

# spectral parametrization
SPEC_PARAM_DIR = RESULTS_DIR / "sensor_param"
SSD_PARAM_DIR = RESULTS_DIR / "ssd_param"
SSD_DIR = RESULTS_DIR / "ssd"

ALPHA_FMIN = 8
ALPHA_FMAX = 13

BETA_FMIN = 16
BETA_FMAX = 30

FRAC_DEVIATION = 0.05

SPEC_FMIN = 2
SPEC_FMAX = 35
SPEC_NR_PEAKS = 5
SPEC_NR_SECONDS = 3
SSD_WIDTH = 2
SNR_THRESHOLD = 0.5

FIG_DIR = BASE_DIR / "figures"
FIG_WIDTH = 8
