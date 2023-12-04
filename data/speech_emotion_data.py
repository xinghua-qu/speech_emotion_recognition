import librosa
import numpy as np
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor

class RAVDESSDataset(Dataset):
    """Dataset class for RAVDESS dataset."""

    SR = 16000  # Sample rate
    MAX_LENGTH = 3000  # Max length of waveform

    def __init__(self, root_dir: str, model_name='openai/whisper-large-v3'):
        self.root_dir = Path(root_dir)
        self.filepaths = self._build_file_list()
        self.model_name = model_name
        self.processor = self._load_pretrained_processor()

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
        try:
            waveform, _ = librosa.load(filepath, sr=self.SR)
            waveform = self._pad_trim_waveform(waveform)
            feature = self.processor(waveform, sampling_rate=self.SR, return_tensors="pt").input_features
        except IOError as e:
            print(f"Error loading {filepath}: {e}")
            raise
        label = self._extract_label_from_filename(filepath)
        return feature, label

    def _pad_trim_waveform(self, waveform):
        """Pad or trim the waveform to the maximum length."""
        if len(waveform) > self.MAX_LENGTH:
            return waveform[:self.MAX_LENGTH]
        return np.pad(waveform, (0, self.MAX_LENGTH - len(waveform)), 'constant')

    def _extract_label_from_filename(self, filepath: Path):
        """Extract the label from the filename."""
        filename = filepath.name
        emotion_code = int(filename.split('-')[2])
        return emotion_code

if __name__ == "__main__":
    # Test dataset usage
    dataset = RAVDESSDataset('/home/ec2-user/teddy_workspace/data/speech_emotion/Speech', model_name='openai/whisper-large-v3')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in dataloader:
        features, labels= batch
        # Process batches here
        print(features.shape, labels)
        break