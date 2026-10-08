import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.utils.class_weight import compute_class_weight

# ================= SETTINGS =================
DATASET = "VoiceDataset"
COMMANDS = ["left", "right", "pause"]
SR = 16000
N_MFCC = 13
MAX_LEN = 110
EPOCHS = 60           # full training epochs
BATCH_SIZE = 32
LR = 0.001
AUG_NOISE = 0.005

# ================= MODEL =================
class VoiceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_MFCC, 128, batch_first=True)
        self.fc = nn.Linear(128, len(COMMANDS))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ================= LOAD DATA =================
X, y = [], []

for idx, cmd in enumerate(COMMANDS):
    folder = os.path.join(DATASET, cmd)
    for file in os.listdir(folder):
        if not file.endswith(".wav"):
            continue

        path = os.path.join(folder, file)
        audio, _ = librosa.load(path, sr=SR)

        # ================= AUGMENTATION =================
        if cmd == "right":
            augmented_audios = [
                audio,
                librosa.effects.pitch_shift(audio, sr=SR, n_steps=1),
                librosa.effects.time_stretch(audio, rate=np.random.uniform(0.9, 1.1)),
                audio + AUG_NOISE * np.random.randn(len(audio))
            ]
        else:
            augmented_audios = [audio]

        # ================= MFCC PROCESSING =================
        for aug_audio in augmented_audios:
            aug_audio = aug_audio / (np.max(np.abs(aug_audio)) + 1e-6)
            mfcc = librosa.feature.mfcc(y=aug_audio, sr=SR, n_mfcc=N_MFCC, n_fft=512, hop_length=160).T
            if mfcc.std() < 1e-6:
                mfcc = mfcc - mfcc.mean()
            else:
                mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
            if mfcc.shape[0] < MAX_LEN:
                mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))
            else:
                mfcc = mfcc[-MAX_LEN:]
            X.append(mfcc.astype(np.float32))
            y.append(idx)

# ================= TENSORS =================
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# ================= CLASS WEIGHTS =================
weights = compute_class_weight(class_weight="balanced", classes=np.unique(y.numpy()), y=y.numpy())
weights = torch.tensor(weights, dtype=torch.float32)

# ================= DATA SPLIT =================
val_size = int(0.15 * len(X))
train_size = len(X) - val_size
train_dataset, val_dataset = random_split(TensorDataset(X, y), [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ================= MODEL, LOSS, OPTIMIZER =================
device = torch.device("cpu")
model = VoiceModel().to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ================= TRAINING =================
print("🚀 Training started for all epochs...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0

    for batch_idx, (xb, yb) in enumerate(train_loader, 1):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        total_loss += loss.item()
        correct += (out.argmax(1) == yb).sum().item()

        # print batch progress every 10 batches
        if batch_idx % 10 == 0 or batch_idx == len(train_loader):
            print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f}")

    train_loss = total_loss / len(train_loader)
    train_acc = correct / len(train_dataset) * 100

    # ================= VALIDATION =================
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            val_loss += loss.item()
            val_correct += (out.argmax(1) == yb).sum().item()
    val_loss /= len(val_loader)
    val_acc = val_correct / len(val_dataset) * 100

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    # ================= SAVE MODEL EVERY EPOCH =================
    torch.save(model.state_dict(), "voice_model_best.pth")

print("\n✅ Training finished. Model saved as voice_model_best.pth")
