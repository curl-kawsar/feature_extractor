# Audio Feature Extractor

This script extracts audio features from WAV files using librosa and saves them as NumPy arrays.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create an `audio_dataset/` folder in the same directory as the script:
```bash
mkdir audio_dataset
```

3. Place your `.wav` audio files in the `audio_dataset/` folder.

## Usage

Run the script:
```bash
python audio_feature_extractor.py
```

## Output

The script will create four NumPy array files and one CSV file:

- `mfcc.npy` - MFCC features, shape: (num_samples, 13, 300)
- `zcr.npy` - Zero Crossing Rate, shape: (num_samples, 300)
- `rmse.npy` - Root Mean Square Energy, shape: (num_samples, 300)  
- `chroma.npy` - Chroma STFT features, shape: (num_samples, 12, 300)
- `audio_features.csv` - Statistical summaries (mean, std, min, max, median) of all features

All features are padded or truncated to exactly 300 frames for consistency. The CSV file contains statistical summaries for easier analysis and visualization.

## Features Extracted

- **MFCC (Mel-Frequency Cepstral Coefficients)**: 13 coefficients that capture the spectral characteristics of audio
- **Zero Crossing Rate (ZCR)**: Rate at which the signal changes from positive to negative or vice versa
- **Root Mean Square Energy (RMSE)**: Measure of the energy or loudness of the audio signal
- **Chroma STFT**: 12-dimensional feature representing the energy distribution across musical pitch classes