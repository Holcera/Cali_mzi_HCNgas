# MZI Calibration and Data Processing

Originally developed in 2022. Migrated to GitHub in 2025 for archival purposes.

This repository contains Python scripts for processing and analyzing data from Mach-Zehnder Interferometer (MZI) experiments. The scripts are designed to preprocess raw data, normalize signals, and extract key parameters such as dispersion and peak frequencies.

### `DAQ_v0.5nm_HCN.py`

- Processes MZI data at a scan velocity of 0.5 nm/s.
- Key functionalities:
  - Reads raw data from CSV files.
  - Normalizes MZI and gas cell signals.
  - Identifies peaks and valleys in the data.
  - Fits dispersion curves and calculates MZI parameters (D1, D2, D3).
  - Visualizes data with matplotlib.

### `DAQ_v1nm_HCN.py`

- Processes MZI data at a scan velocity of 1 nm/s.
- Key functionalities:
  - Similar to `DAQ_v0.5nm_HCN.py`.

### `Preprocess_data.py`

- Handles initial preprocessing of raw data.
- Key functionalities:
  - Reads raw CSV data.
  - Renames columns for clarity.
  - Visualizes raw signals using matplotlib.

## Dependencies

The scripts rely on the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `hvplot`
- `holoviews`
- `scipy`
- `bokeh`

## Potential Issues

- The precision of the HCN gas cell may not be sufficient, which could lead to significant errors in the disperison.