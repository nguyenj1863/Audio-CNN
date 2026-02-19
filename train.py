# Standard library imports
from pathlib import Path

# Third-party imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

# Modal imports
import modal

# Local imports
from model import AudioCNN

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



def mixup_data(x, y):
    """
    Applies Mixup augmentation to a batch of data.
    Inputs:
        x: Tensor of input data (batch_size, ...)
        y: Tensor of labels (batch_size,)
    Outputs:
        mixed_x: Mixed input tensor (batch_size, ...)
        y_a: Original labels (batch_size,)
        y_b: Shuffled labels (batch_size,)
        lam: Mixup interpolation coefficient (float)
    """
    lam = np.random.beta(0.2, 0.2)

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the mixup loss given predictions and mixed targets.
    Inputs:
        criterion: Loss function (CrossEntropyLoss)
        pred: Model predictions (batch_size, num_classes)
        y_a: Original labels (batch_size,)
        y_b: Shuffled labels (batch_size,)
        lam: Mixup interpolation coefficient (float)
    Output:
        loss: Scalar mixup loss (float)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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
    # Path to the ESC-50 dataset directory
    esc50_dir = Path("/opt/esc50-data")

    # Data augmentation and preprocessing for training
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

    # Preprocessing for validation (no augmentation)
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

    # Create training and validation datasets
    train_dataset = ESC50Dataset(
        data_dir=esc50_dir, metadata_file=esc50_dir / "meta" / "esc50.csv", split="train", transform=train_transform)
    
    val_dataset = ESC50Dataset(
        data_dir=esc50_dir, metadata_file=esc50_dir / "meta" / "esc50.csv", split="val", transform=val_transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Set device and initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(num_classes=len(train_dataset.classes))
    model.to(device)

    # Training hyperparameters and optimizer
    num_epochs = 100
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.002,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1
    )

    best_accuracy = 0.0

    print("Starting training")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # Training loop for one epoch
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)

            # Apply mixup augmentation with 70% probability
            if np.random.random() > 0.7:
                data, target_a, target_b, lam = mixup_data(data, target)
                output = model(data)
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            else:
                output = model(data)
                loss = criterion(output, target)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_epoch_loss = epoch_loss / len(train_dataloader)

        # Validation after each epoch
        model.eval()

        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            pass

@app.local_entrypoint()
def main():
    train.remote()
