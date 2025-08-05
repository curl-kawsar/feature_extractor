#!/usr/bin/env python3
"""
Audio Feature Extractor

This script loads all .wav files from an audio_dataset/ folder and extracts
various audio features using librosa. The features are then padded/truncated
to a fixed length and saved as NumPy arrays.

Features extracted:
- MFCC (13 coefficients)
- Zero Crossing Rate (ZCR)
- Root Mean Square Energy (RMSE)
- Chroma STFT

Output files:
- mfcc.npy: shape (num_samples, 13, 300)
- zcr.npy: shape (num_samples, 300)
- rmse.npy: shape (num_samples, 300)
- chroma.npy: shape (num_samples, 12, 300)
- audio_features.csv: statistical summaries of all features
"""

import os
import glob
import numpy as np
import pandas as pd
import librosa
from typing import Tuple, List
import warnings

# Suppress librosa warnings
warnings.filterwarnings('ignore')

def pad_or_truncate(feature: np.ndarray, target_length: int = 300) -> np.ndarray:
    """
    Pad or truncate a feature array to the target length.
    
    Args:
        feature: Input feature array
        target_length: Target length (default: 300)
    
    Returns:
        Feature array with shape (..., target_length)
    """
    if feature.shape[-1] > target_length:
        # Truncate
        return feature[..., :target_length]
    elif feature.shape[-1] < target_length:
        # Pad with zeros
        pad_width = target_length - feature.shape[-1]
        if feature.ndim == 1:
            return np.pad(feature, (0, pad_width), mode='constant', constant_values=0)
        else:
            return np.pad(feature, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    else:
        return feature

def extract_features(audio_path: str, sr: int = 22050) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract audio features from a single audio file.
    
    Args:
        audio_path: Path to the audio file
        sr: Sample rate (default: 22050)
    
    Returns:
        Tuple of (mfcc, zcr, rmse, chroma) features
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Extract MFCC features (13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Extract Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]  # Remove extra dimension
        
        # Extract Root Mean Square Energy
        rmse = librosa.feature.rms(y=y)[0]  # Remove extra dimension
        
        # Extract Chroma STFT
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Pad or truncate all features to 300 frames
        mfcc = pad_or_truncate(mfcc, 300)
        zcr = pad_or_truncate(zcr, 300)
        rmse = pad_or_truncate(rmse, 300)
        chroma = pad_or_truncate(chroma, 300)
        
        return mfcc, zcr, rmse, chroma
        
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None, None, None, None

def compute_feature_statistics(features: np.ndarray) -> dict:
    """
    Compute statistical summaries of features.
    
    Args:
        features: Feature array
    
    Returns:
        Dictionary with statistical measures
    """
    if features.ndim == 1:
        # For 1D features (ZCR, RMSE)
        return {
            'mean': np.mean(features),
            'std': np.std(features),
            'min': np.min(features),
            'max': np.max(features),
            'median': np.median(features)
        }
    else:
        # For 2D features (MFCC, Chroma) - compute stats across time dimension
        return {
            'mean': np.mean(features, axis=1),  # Mean across time for each coefficient
            'std': np.std(features, axis=1),    # Std across time for each coefficient
            'min': np.min(features, axis=1),    # Min across time for each coefficient
            'max': np.max(features, axis=1),    # Max across time for each coefficient
            'median': np.median(features, axis=1)  # Median across time for each coefficient
        }

def process_audio_dataset(dataset_path: str = "audio_dataset/") -> None:
    """
    Process all .wav files in the dataset folder and save extracted features.
    
    Args:
        dataset_path: Path to the audio dataset folder
    """
    # Check if dataset folder exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset folder '{dataset_path}' not found!")
        print("Please create the folder and add your .wav files to it.")
        return
    
    # Find all .wav files
    wav_files = glob.glob(os.path.join(dataset_path, "*.wav"))
    
    if not wav_files:
        print(f"No .wav files found in '{dataset_path}'")
        return
    
    print(f"Found {len(wav_files)} .wav files")
    
    # Initialize lists to store features
    all_mfcc = []
    all_zcr = []
    all_rmse = []
    all_chroma = []
    
    # Initialize list to store CSV data
    csv_data = []
    
    # Process each audio file
    for i, wav_file in enumerate(wav_files):
        print(f"Processing {i+1}/{len(wav_files)}: {os.path.basename(wav_file)}")
        
        mfcc, zcr, rmse, chroma = extract_features(wav_file)
        
        if mfcc is not None:
            all_mfcc.append(mfcc)
            all_zcr.append(zcr)
            all_rmse.append(rmse)
            all_chroma.append(chroma)
            
            # Compute statistics for CSV
            mfcc_stats = compute_feature_statistics(mfcc)
            zcr_stats = compute_feature_statistics(zcr)
            rmse_stats = compute_feature_statistics(rmse)
            chroma_stats = compute_feature_statistics(chroma)
            
            # Create row for CSV
            row = {'filename': os.path.basename(wav_file)}
            
            # Add ZCR statistics (1D feature)
            for stat_name, stat_value in zcr_stats.items():
                row[f'zcr_{stat_name}'] = stat_value
            
            # Add RMSE statistics (1D feature)
            for stat_name, stat_value in rmse_stats.items():
                row[f'rmse_{stat_name}'] = stat_value
            
            # Add MFCC statistics (2D feature - 13 coefficients)
            for stat_name, stat_values in mfcc_stats.items():
                for i, stat_value in enumerate(stat_values):
                    row[f'mfcc_{i}_{stat_name}'] = stat_value
            
            # Add Chroma statistics (2D feature - 12 coefficients)
            for stat_name, stat_values in chroma_stats.items():
                for i, stat_value in enumerate(stat_values):
                    row[f'chroma_{i}_{stat_name}'] = stat_value
            
            csv_data.append(row)
        else:
            print(f"Skipping {wav_file} due to processing error")
    
    if not all_mfcc:
        print("No valid features extracted. Please check your audio files.")
        return
    
    # Convert lists to NumPy arrays
    mfcc_array = np.array(all_mfcc)      # Shape: (num_samples, 13, 300)
    zcr_array = np.array(all_zcr)        # Shape: (num_samples, 300)
    rmse_array = np.array(all_rmse)      # Shape: (num_samples, 300)
    chroma_array = np.array(all_chroma)  # Shape: (num_samples, 12, 300)
    
    # Print shapes for verification
    print(f"\nExtracted features from {len(all_mfcc)} files:")
    print(f"MFCC shape: {mfcc_array.shape}")
    print(f"ZCR shape: {zcr_array.shape}")
    print(f"RMSE shape: {rmse_array.shape}")
    print(f"Chroma shape: {chroma_array.shape}")
    
    # Save arrays to disk
    print("\nSaving feature arrays...")
    np.save('mfcc.npy', mfcc_array)
    np.save('zcr.npy', zcr_array)
    np.save('rmse.npy', rmse_array)
    np.save('chroma.npy', chroma_array)
    
    # Save CSV file with feature statistics
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv('audio_features.csv', index=False)
        print("- audio_features.csv")
    
    print("Feature extraction completed successfully!")
    print("Saved files:")
    print("- mfcc.npy")
    print("- zcr.npy")
    print("- rmse.npy")
    print("- chroma.npy")
    print("- audio_features.csv")

def main():
    """Main function to run the feature extraction."""
    print("Audio Feature Extractor")
    print("=" * 50)
    
    # Process the audio dataset
    process_audio_dataset()

if __name__ == "__main__":
    main()