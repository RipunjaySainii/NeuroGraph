"""
Process EEG data from the PhysioNet EEG Motor Movement/Imagery Dataset.
Subject 1, Run 4.
Calculates a 64x64 Pearson cross-correlation matrix using numpy and exports it to eeg_adjacency_matrix.csv.
"""

from pathlib import Path
import mne
from mne.datasets import eegbci
import numpy as np


def main():
    # 1. Download/fetch PhysioNet EEG Motor Movement/Imagery Dataset (Subject 1, Run 4)
    subject = 1
    runs = [4]
    file_paths = eegbci.load_data(subject, runs)
    edf_path = file_paths[0]

    # 2. Read raw EDF data
    raw = mne.io.read_raw_edf(edf_path, preload=True)

    # Extract raw EEG signals: shape (n_channels, n_times) -> (64, 20000)
    data = raw.get_data()

    # 3. Calculate 64x64 Pearson cross-correlation matrix using numpy
    corr_matrix = np.corrcoef(data)

    if corr_matrix.shape != (64, 64):
        raise ValueError(f"Expected 64x64 matrix, but got {corr_matrix.shape}")

    # 4. Export matrix to 'eeg_adjacency_matrix.csv' without headers or row indices
    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / "eeg_adjacency_matrix.csv"

    np.savetxt(output_path, corr_matrix, delimiter=",")
    print(f"Adjacency matrix successfully exported to {output_path} (shape: {corr_matrix.shape})")


if __name__ == "__main__":
    main()
