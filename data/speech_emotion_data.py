import os
# import torchaudio
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

class RAVDESSDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.filepaths = []
        self._build_file_list()

    def _build_file_list(self):
        # Iterate through the actor folders
        for actor_id in range(1, 24):
            actor_folder = os.path.join(self.root_dir, f"Actor_{str(actor_id).zfill(2)}")
            if os.path.exists(actor_folder):
                # List all files in the actor folder
                for file in os.listdir(actor_folder):
                    if file.endswith(".wav"):
                        self.filepaths.append(os.path.join(actor_folder, file))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        try:
            waveform, sample_rate = librosa.load(filepath, sr=None)
            if sample_rate != 16000:
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000) 
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            raise
        label = self.extract_label_from_filename(filepath)
        return torch.from_numpy(waveform), sample_rate, label

    def pad_trim_waveform(self, waveform, max_length):
        if len(waveform) > max_length:
            waveform = waveform[:max_length]
        elif len(waveform) < max_length:
            padding = max_length - len(waveform)
            waveform = np.pad(waveform, (0, padding), 'constant')
        return waveform

    def extract_label_from_filename(self, filepath):
        # Implement logic to extract the label from the filename
        filename = os.path.basename(filepath)
        parts = filename.split('-')
        emotion_code = int(parts[2])
        return emotion_code


def custom_collate_fn(batch):
    waveforms, sample_rates, labels = zip(*batch)

    # 计算每个音频的长度
    lengths = [len(waveform) for waveform in waveforms]

    # 将所有音频填充到批处理中最长的音频长度
    waveforms_padded = pad_sequence(waveforms, batch_first=True, padding_value=0)

    # 创建一个 mask，用于在后续处理中标识原始音频长度和填充部分
    max_length = waveforms_padded.shape[1]
    masks = [torch.zeros(max_length, dtype=torch.bool) for _ in range(len(waveforms))]
    for i, length in enumerate(lengths):
        masks[i][:length] = 1

    masks = torch.stack(masks, dim=0)

    sample_rates = torch.tensor(sample_rates)
    labels = torch.tensor(labels)

    return waveforms_padded, sample_rates, labels, masks

if __name__ == "__main__":
    # Usage
    dataset = RAVDESSDataset('/home/ec2-user/teddy_workspace/data/speech_emotion/Speech')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

    for batch in dataloader:
        waveforms, sample_rates, labels, masks= batch
        # Process batches here
        print(waveforms.shape, sample_rates.shape, labels, masks.shape)
        break