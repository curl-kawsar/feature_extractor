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
- audio_features_combined.npz: contains all features in one file
  - mfcc: shape (num_samples, 13, 300)
  - zcr: shape (num_samples, 300)
  - rmse: shape (num_samples, 300)
  - chroma: shape (num_samples, 12, 300)
  - filenames: list of processed audio filenames
- audio_features.csv: statistical summaries of all features
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Tuple, List
import warnings
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

# Try to import librosa with proper error handling
try:
    print("Loading librosa (this may take a moment on first run)...")
    import librosa
    print("Librosa loaded successfully!")
except Exception as e:
    print(f"Error importing librosa: {e}")
    print("Please try installing/reinstalling librosa and its dependencies:")
    print("pip uninstall librosa")
    print("pip install librosa==0.10.1")
    sys.exit(1)

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
        # Load audio file with better error handling
        print(f"  Loading audio: {os.path.basename(audio_path)}")
        y, sr_actual = librosa.load(audio_path, sr=sr, duration=None)
        
        # Check if audio loaded successfully
        if len(y) == 0:
            print(f"  Warning: Empty audio file {audio_path}")
            return None, None, None, None
        
        print(f"  Loaded {len(y)} samples at {sr_actual}Hz")
        
        # Extract MFCC features (13 coefficients)
        print("  Extracting MFCC...")
        mfcc = librosa.feature.mfcc(y=y, sr=sr_actual, n_mfcc=13, hop_length=512)
        
        # Extract Zero Crossing Rate
        print("  Extracting ZCR...")
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)[0]  # Remove extra dimension
        
        # Extract Root Mean Square Energy
        print("  Extracting RMSE...")
        rmse = librosa.feature.rms(y=y, hop_length=512)[0]  # Remove extra dimension
        
        # Extract Chroma STFT
        print("  Extracting Chroma...")
        chroma = librosa.feature.chroma_stft(y=y, sr=sr_actual, hop_length=512)
        
        print(f"  Feature shapes before padding: MFCC:{mfcc.shape}, ZCR:{zcr.shape}, RMSE:{rmse.shape}, Chroma:{chroma.shape}")
        
        # Pad or truncate all features to 300 frames
        mfcc = pad_or_truncate(mfcc, 300)
        zcr = pad_or_truncate(zcr, 300)
        rmse = pad_or_truncate(rmse, 300)
        chroma = pad_or_truncate(chroma, 300)
        
        print(f"  Feature shapes after padding: MFCC:{mfcc.shape}, ZCR:{zcr.shape}, RMSE:{rmse.shape}, Chroma:{chroma.shape}")
        
        return mfcc, zcr, rmse, chroma
        
    except KeyboardInterrupt:
        print(f"\nProcessing interrupted by user.")
        return None, None, None, None
    except Exception as e:
        print(f"  Error processing {audio_path}: {str(e)}")
        print(f"  Error type: {type(e).__name__}")
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
    successful_files = []
    try:
        for i, wav_file in enumerate(wav_files):
            print(f"\nProcessing {i+1}/{len(wav_files)}: {os.path.basename(wav_file)}")
            
            mfcc, zcr, rmse, chroma = extract_features(wav_file)
            
            if mfcc is not None:
                all_mfcc.append(mfcc)
                all_zcr.append(zcr)
                all_rmse.append(rmse)
                all_chroma.append(chroma)
                successful_files.append(wav_file)
                
                # Compute statistics for CSV
                print("  Computing statistics...")
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
                    for coef_idx, stat_value in enumerate(stat_values):
                        row[f'mfcc_{coef_idx}_{stat_name}'] = stat_value
                
                # Add Chroma statistics (2D feature - 12 coefficients)
                for stat_name, stat_values in chroma_stats.items():
                    for coef_idx, stat_value in enumerate(stat_values):
                        row[f'chroma_{coef_idx}_{stat_name}'] = stat_value
                
                csv_data.append(row)
                print(f"  Successfully processed {os.path.basename(wav_file)}")
            else:
                print(f"  Skipping {wav_file} due to processing error")
    
    except KeyboardInterrupt:
        print(f"\n\nProcessing interrupted by user. Processed {len(all_mfcc)}/{len(wav_files)} files.")
        if len(all_mfcc) == 0:
            print("No files were processed successfully. Exiting.")
            return
    
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
    
    # Combine all features into a single dictionary
    combined_features = {
        'mfcc': mfcc_array,
        'zcr': zcr_array,
        'rmse': rmse_array,
        'chroma': chroma_array,
        'filenames': [os.path.basename(f) for f in successful_files]
    }
    
    # Save combined features to a single file
    print("\nSaving combined feature file...")
    np.savez('audio_features_combined.npz', **combined_features)
    
    # Save CSV file with feature statistics
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv('audio_features.csv', index=False)
        print("- audio_features.csv")
    
    print("Feature extraction completed successfully!")
    print("Saved files:")
    print("- audio_features_combined.npz (contains all 4 feature types)")
    print("- audio_features.csv")

def main():
    """Main function to run the feature extraction."""
    print("Audio Feature Extractor")
    print("=" * 50)
    
    # Process the audio dataset
    process_audio_dataset()

if __name__ == "__main__":
    main()