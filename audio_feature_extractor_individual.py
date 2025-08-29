#!/usr/bin/env python3
"""
Audio Feature Extractor - Individual Files & Emotion-wise Organization

This script loads all .wav files from emotion-based folders in audio_dataset/ and extracts
various audio features using librosa. Features are saved as individual .npy files
organized by emotion, plus emotion-wise CSV files with statistics.

Folder structure:
audio_dataset/
  ├── angry/
  ├── happy/
  ├── normal/
  └── sad/

Output structure:
audio_features/
  ├── angry/
  │   ├── angry-1.npy (contains all 4 features for this file)
  │   └── angry-2.npy
  ├── happy/
  │   └── ...
  ├── normal/
  │   └── ...
  ├── sad/
  │   └── ...
  ├── angry_features.csv
  ├── happy_features.csv
  ├── normal_features.csv
  └── sad_features.csv

Features extracted for each file:
- MFCC (13 coefficients)
- Zero Crossing Rate (ZCR)
- Root Mean Square Energy (RMSE)
- Chroma STFT

Each .npy file contains a dictionary with keys:
- 'mfcc': shape (13, 300)
- 'zcr': shape (300,)
- 'rmse': shape (300,)
- 'chroma': shape (12, 300)
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
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
        print(f"    Loading audio: {os.path.basename(audio_path)}")
        y, sr_actual = librosa.load(audio_path, sr=sr, duration=None)
        
        # Check if audio loaded successfully
        if len(y) == 0:
            print(f"    Warning: Empty audio file {audio_path}")
            return None, None, None, None
        
        print(f"    Loaded {len(y)} samples at {sr_actual}Hz")
        
        # Extract MFCC features (13 coefficients)
        print("    Extracting MFCC...")
        mfcc = librosa.feature.mfcc(y=y, sr=sr_actual, n_mfcc=13, hop_length=512)
        
        # Extract Zero Crossing Rate
        print("    Extracting ZCR...")
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)[0]  # Remove extra dimension
        
        # Extract Root Mean Square Energy
        print("    Extracting RMSE...")
        rmse = librosa.feature.rms(y=y, hop_length=512)[0]  # Remove extra dimension
        
        # Extract Chroma STFT
        print("    Extracting Chroma...")
        chroma = librosa.feature.chroma_stft(y=y, sr=sr_actual, hop_length=512)
        
        print(f"    Feature shapes before padding: MFCC:{mfcc.shape}, ZCR:{zcr.shape}, RMSE:{rmse.shape}, Chroma:{chroma.shape}")
        
        # Pad or truncate all features to 300 frames
        mfcc = pad_or_truncate(mfcc, 300)
        zcr = pad_or_truncate(zcr, 300)
        rmse = pad_or_truncate(rmse, 300)
        chroma = pad_or_truncate(chroma, 300)
        
        print(f"    Feature shapes after padding: MFCC:{mfcc.shape}, ZCR:{zcr.shape}, RMSE:{rmse.shape}, Chroma:{chroma.shape}")
        
        return mfcc, zcr, rmse, chroma
        
    except KeyboardInterrupt:
        print(f"\nProcessing interrupted by user.")
        return None, None, None, None
    except Exception as e:
        print(f"    Error processing {audio_path}: {str(e)}")
        print(f"    Error type: {type(e).__name__}")
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

def save_individual_features(features_dict: Dict, output_path: str, filename: str) -> None:
    """
    Save individual audio file features to .npy file.
    
    Args:
        features_dict: Dictionary containing all features
        output_path: Directory to save the file
        filename: Base filename (without extension)
    """
    os.makedirs(output_path, exist_ok=True)
    npy_filename = os.path.join(output_path, f"{filename}.npy")
    np.save(npy_filename, features_dict)
    print(f"    Saved: {npy_filename}")

def process_emotion_folder(emotion_path: str, emotion_name: str, output_base_path: str) -> List[Dict]:
    """
    Process all .wav files in a specific emotion folder.
    
    Args:
        emotion_path: Path to emotion folder
        emotion_name: Name of emotion (angry, happy, etc.)
        output_base_path: Base path for output files
    
    Returns:
        List of dictionaries containing feature statistics for CSV
    """
    print(f"\n{'='*60}")
    print(f"Processing {emotion_name.upper()} emotion files")
    print(f"{'='*60}")
    
    # Find all .wav files in emotion folder
    wav_files = glob.glob(os.path.join(emotion_path, "*.wav"))
    
    if not wav_files:
        print(f"No .wav files found in {emotion_path}")
        return []
    
    print(f"Found {len(wav_files)} .wav files in {emotion_name} folder")
    
    # Create output directory for this emotion
    emotion_output_path = os.path.join(output_base_path, emotion_name)
    
    # Initialize list to store CSV data
    csv_data = []
    successful_count = 0
    
    # Process each audio file
    for i, wav_file in enumerate(wav_files):
        filename_without_ext = os.path.splitext(os.path.basename(wav_file))[0]
        print(f"\n  Processing {i+1}/{len(wav_files)}: {filename_without_ext}")
        
        mfcc, zcr, rmse, chroma = extract_features(wav_file)
        
        if mfcc is not None:
            # Create features dictionary for this file
            features_dict = {
                'mfcc': mfcc,      # Shape: (13, 300)
                'zcr': zcr,        # Shape: (300,)
                'rmse': rmse,      # Shape: (300,)
                'chroma': chroma,  # Shape: (12, 300)
                'emotion': emotion_name,
                'filename': os.path.basename(wav_file)
            }
            
            # Save individual .npy file
            save_individual_features(features_dict, emotion_output_path, filename_without_ext)
            
            # Compute statistics for CSV
            print("    Computing statistics...")
            mfcc_stats = compute_feature_statistics(mfcc)
            zcr_stats = compute_feature_statistics(zcr)
            rmse_stats = compute_feature_statistics(rmse)
            chroma_stats = compute_feature_statistics(chroma)
            
            # Create row for CSV
            row = {
                'filename': os.path.basename(wav_file),
                'emotion': emotion_name
            }
            
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
            successful_count += 1
            print(f"    Successfully processed {filename_without_ext}")
        else:
            print(f"    Skipping {wav_file} due to processing error")
    
    print(f"\n  {emotion_name.upper()} Summary: {successful_count}/{len(wav_files)} files processed successfully")
    return csv_data

def process_audio_dataset(dataset_path: str = "audio_dataset/", output_path: str = "audio_features/") -> None:
    """
    Process all emotion folders in the dataset and save features individually and by emotion.
    
    Args:
        dataset_path: Path to the audio dataset folder
        output_path: Path to save the extracted features
    """
    # Check if dataset folder exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset folder '{dataset_path}' not found!")
        print("Please create the folder and add your emotion subfolders with .wav files.")
        return
    
    # Get list of emotion folders
    emotion_folders = [f for f in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, f))]
    
    if not emotion_folders:
        print(f"No emotion folders found in '{dataset_path}'")
        print("Expected folders: angry, happy, normal, sad")
        return
    
    print(f"Found emotion folders: {', '.join(emotion_folders)}")
    
    # Create output base directory
    os.makedirs(output_path, exist_ok=True)
    
    # Process each emotion folder
    all_emotion_data = {}
    
    try:
        for emotion in emotion_folders:
            emotion_path = os.path.join(dataset_path, emotion)
            csv_data = process_emotion_folder(emotion_path, emotion, output_path)
            
            if csv_data:
                all_emotion_data[emotion] = csv_data
                
                # Save emotion-specific CSV
                df = pd.DataFrame(csv_data)
                csv_filename = os.path.join(output_path, f"{emotion}_features.csv")
                df.to_csv(csv_filename, index=False)
                print(f"  Saved CSV: {csv_filename}")
    
    except KeyboardInterrupt:
        print(f"\n\nProcessing interrupted by user.")
        return
    
    # Create combined CSV with all emotions
    if all_emotion_data:
        print(f"\n{'='*60}")
        print("Creating combined CSV file...")
        all_csv_data = []
        for emotion, csv_data in all_emotion_data.items():
            all_csv_data.extend(csv_data)
        
        combined_df = pd.DataFrame(all_csv_data)
        combined_csv_filename = os.path.join(output_path, "all_emotions_features.csv")
        combined_df.to_csv(combined_csv_filename, index=False)
        print(f"Saved combined CSV: {combined_csv_filename}")
    
    print(f"\n{'='*60}")
    print("FEATURE EXTRACTION COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print("Generated files:")
    print(f"📁 Output folder: {output_path}")
    
    for emotion in emotion_folders:
        emotion_path = os.path.join(output_path, emotion)
        if os.path.exists(emotion_path):
            npy_files = glob.glob(os.path.join(emotion_path, "*.npy"))
            print(f"  📁 {emotion}/ - {len(npy_files)} .npy files")
            csv_file = os.path.join(output_path, f"{emotion}_features.csv")
            if os.path.exists(csv_file):
                print(f"  📄 {emotion}_features.csv")
    
    if os.path.exists(os.path.join(output_path, "all_emotions_features.csv")):
        print(f"  📄 all_emotions_features.csv (combined)")
    
    print(f"\n🎵 Each .npy file contains:")
    print("   - 'mfcc': MFCC features (13, 300)")
    print("   - 'zcr': Zero Crossing Rate (300,)")
    print("   - 'rmse': Root Mean Square Energy (300,)")
    print("   - 'chroma': Chroma features (12, 300)")
    print("   - 'emotion': emotion label")
    print("   - 'filename': original filename")

def load_individual_features(npy_file_path: str) -> Dict:
    """
    Load features from an individual .npy file.
    
    Args:
        npy_file_path: Path to .npy file
    
    Returns:
        Dictionary containing features
    """
    try:
        features = np.load(npy_file_path, allow_pickle=True).item()
        return features
    except Exception as e:
        print(f"Error loading {npy_file_path}: {e}")
        return None

def main():
    """Main function to run the feature extraction."""
    print("Audio Feature Extractor - Individual Files & Emotion-wise")
    print("=" * 60)
    
    # Process the audio dataset
    process_audio_dataset()
    
    print("\n" + "=" * 60)
    print("USAGE EXAMPLE:")
    print("To load features from an individual file:")
    print("  import numpy as np")
    print("  features = np.load('audio_features/angry/angry-1.npy', allow_pickle=True).item()")
    print("  mfcc = features['mfcc']")
    print("  zcr = features['zcr']")
    print("  rmse = features['rmse']")
    print("  chroma = features['chroma']")
    print("=" * 60)

if __name__ == "__main__":
    main()