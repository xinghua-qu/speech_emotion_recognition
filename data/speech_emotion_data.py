import librosa
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor
from pathlib import Path
import joblib  # For efficient caching

import shutil


class RAVDESSDataset(Dataset):
    """Dataset class for RAVDESS dataset with improved data I/O."""

    SR = 16000  # Sample rate
    MAX_LENGTH = 30*SR  # Max length of waveform
    CACHE_DIR = Path("cache")  # Directory to store cached files

    def __init__(self, root_dir: str, model_name='openai/whisper-large-v3'):
        self.root_dir = Path(root_dir)
        self.filepaths = self._build_file_list()
        self.model_name = model_name
        self.processor = self._load_pretrained_processor()
        # 调用这个函数来清理缓存
        self.clear_cache()
        
    def clear_cache(self):
        if RAVDESSDataset.CACHE_DIR.exists():
            for filename in RAVDESSDataset.CACHE_DIR.iterdir():
                try:
                    filename.unlink(missing_ok=True)  # Python 3.8+
                except FileNotFoundError:
                    pass  # If the file does not exist, continue to the next file

    def _load_pretrained_processor(self):
        """Load the pre-trained Whisper model."""
        return AutoFeatureExtractor.from_pretrained(self.model_name)

    def _build_file_list(self):
        """Build a list of file paths for the audio files."""
        filepaths = []
        for actor_id in range(1, 24):
            actor_folder = self.root_dir / f"Actor_{str(actor_id).zfill(2)}"
            if actor_folder.exists():
                for file in actor_folder.iterdir():
                    if file.suffix == ".wav":
                        filepaths.append(file)
        return filepaths

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        filepath = self.filepaths[idx]
        cache_file = self.CACHE_DIR / f"{filepath.stem}.pkl"

        if not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR)

        if cache_file.exists():
            # Load preprocessed data from cache
            waveform, feature, label = joblib.load(cache_file)
        else:
            # Process and cache the data
            waveform, feature, label = self._process_file(filepath)
            joblib.dump((waveform, feature, label), cache_file)

        return waveform, feature, label

    def _process_file(self, filepath):
        """Process the audio file."""
        try:
            waveform, _ = librosa.load(filepath, sr=self.SR)
            waveform_padded = self._pad_trim_waveform(waveform)
            feature = self.processor(waveform_padded, sampling_rate=self.SR, return_tensors="pt").input_features
            feature = torch.squeeze(feature)
        except IOError as e:
            print(f"Error loading {filepath}: {e}")
            raise
        label = self._extract_label_from_filename(filepath)
        return waveform_padded, feature, label

    def _pad_trim_waveform(self, waveform):
        """Pad or trim the waveform to the maximum length."""
        if len(waveform) > self.MAX_LENGTH:
            return waveform[:self.MAX_LENGTH]
        return np.pad(waveform, (0, self.MAX_LENGTH - len(waveform)), 'constant')

    def _extract_label_from_filename(self, filepath: Path):
        """Extract the label from the filename."""
        filename = filepath.name
        emotion_code = int(filename.split('-')[2])
        emotion_code = int(emotion_code - 1)
        return emotion_code
