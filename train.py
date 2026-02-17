from pathlib import Path
import pandas as pd

import modal
import torch
from torch.utils.data import Dataset
import torchaudio
import torch.nn as nn
import torchaudio.transforms as T

app = modal.App("audio-cnn")

# Modal image definition for remote training environment.
# - Based on minimal Debian image
# - Installs Python and system dependencies
# - Downloads and extracts ESC-50 dataset
# - Copies audio data to /opt/esc50-data
# - Adds local model.py source
image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])
         .run_commands([
             "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
             "cd /tmp && unzip esc50.zip",
             "mkdir -p /opt/esc50-data",
             "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
             "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
         ])
         .add_local_python_source("model"))


# Volume for storing input data (ESC-50 audio files)
volume = modal.Volume.from_name("esc50-data", create_if_missing=True)
# Volume for storing training outputs
modal_volume = modal.Volume.from_name("esc50-data", create_if_missing=True)

class ESC50Dataset(Dataset):
    """
    PyTorch Dataset for ESC-50 environmental audio dataset.
    Inputs:
        - data_dir: Directory containing audio files
        - metadata_file: CSV file with metadata (filename, category, fold, etc.)
        - split: 'train' or 'test' (determines which folds to use)
        - transform: Optional transform to apply to waveform (mel spectrogram configs)
    Outputs:
        - __getitem__: Returns (spectrogram, label) tuple for each sample
    """
    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        # Filter metadata based on split
        # The ESC-50 dataset has 5 folds (each with 400 samples, total 2000).
        # We use the 5th fold for validation (20% of data), and the other 4 folds for training (80%).
        if split == 'train':
            self.metadata = self.metadata[self.metadata['fold'] != 5]
        else:
            self.metadata = self.metadata[self.metadata['fold'] == 5]

        # Prepare class mappings
        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(self.class_to_idx)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        Output: int
        """
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.
        Input: idx (int) - index of the sample
        Output: (spectrogram, label)
            - spectrogram: Tensor (or waveform if no transform)
            - label: int (class index)
        """
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row['filename']

        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Apply transform if provided
        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform

        return spectrogram, row['label']

@app.function(
    image=image,
    gpu="A10",
    volumes={
        "/data": volume,    # Mounts input data volume at /data
        "/models": modal_volume  # Mounts output volume at /models
    },
    timeout=60 * 60 * 3  # 3 hours (in seconds)
)
def train():
    esc50_dir = Path("/opt/esc50-data")

    train_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050, 
            n_fft=1024, 
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB(),
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80)
    )

    val_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050, 
            n_fft=1024, 
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB()
    )

    train_dataset = ESC50Dataset(
        data_dir=esc50_dir, metadata_file=esc50_dir / "meta" / "esc50.csv", split="train", transform=train_transform)
    
    val_dataset = ESC50Dataset(
        data_dir=esc50_dir, metadata_file=esc50_dir / "meta" / "esc50.csv", split="val", transform=val_transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

@app.local_entrypoint()
def main():
    train.remote()
