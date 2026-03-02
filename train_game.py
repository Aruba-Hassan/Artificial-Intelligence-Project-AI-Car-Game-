import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ================= CONFIG =================
DATASET_DIR = "dataset"        # dataset/left right pause
SAMPLE_RATE = 16000
N_MFCC = 13
MAX_LEN = 60
BATCH_SIZE = 16
EPOCHS = 100
LR = 0.001

LABELS = {"left": 0, "right": 1, "pause": 2}
device = torch.device("cpu")
print("Device:", device)

# ================= DATASET =================
class VoiceDataset(Dataset):
    def __init__(self):
        self.samples = []
        for label, idx in LABELS.items():
            folder = os.path.join(DATASET_DIR, label)
            for f in os.listdir(folder):
                if f.lower().endswith(".wav"):
                    self.samples.append((os.path.join(folder, f), idx))
        np.random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        y, _ = librosa.load(path, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T

        if mfcc.shape[0] > MAX_LEN:
            mfcc = mfcc[:MAX_LEN]
        else:
            mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))

        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label)

dataset = VoiceDataset()
print("Total samples:", len(dataset))

train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_ds, val_ds = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ================= MODEL =================
class VoiceLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=13,
            hidden_size=96,
            num_layers=2,
            batch_first=True,
            dropout=0.35
        )
        self.fc = nn.Linear(96, 3)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

model = VoiceLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

os.makedirs("model", exist_ok=True)
best_val = float("inf")

print("\nTraining started...\n")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            val_loss += criterion(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1:03d} | Train {train_loss:.3f} | Val {val_loss:.3f} | Acc {acc:.2f}%")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "model/voice_lstm.pth")

print("\nTRAINING COMPLETE — model/voice_lstm.pth SAVED")
