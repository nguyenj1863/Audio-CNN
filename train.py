from pathlib import Path
import pandas as pd

import modal
import torch
from torch.utils.data import Dataset
import torchaudio

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
    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        if split == 'train':
            self.metadata = self.metadata[self.metadata['fold'] != 5]
        else:
            self.metadata = self.metadata[self.metadata['fold'] == 5]
        
        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(self.class_to_idx)
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row['filename']
        
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.transform:
            spectogram = self.transform(waveform)
        else:
            spectogram = waveform
        
        return spectogram, row['label']

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
    print("training")

@app.local_entrypoint()
def main():
    train.remote()
