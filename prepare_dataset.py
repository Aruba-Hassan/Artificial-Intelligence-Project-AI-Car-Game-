import os
import librosa
import numpy as np
import torch
import sys

# ================= SETTINGS =================
DATASET_PATH = r"C:\Users\Dell\CarGame\VoiceDataset"  # <-- change if your folder is elsewhere
COMMANDS = ["left", "right", "pause"]
MAX_LEN = 30  # Max timesteps for LSTM

X = []
y = []

print("Preparing dataset...", flush=True)

# ================= CHECK FOLDERS =================
for command in COMMANDS:
    folder = os.path.join(DATASET_PATH, command)
    if not os.path.exists(folder):
        print(f"ERROR: Folder not found -> {folder}", flush=True)
        sys.exit(1)
    else:
        print(f"Found folder for '{command}': {folder}", flush=True)

# ================= PROCESS FILES =================
for idx, command in enumerate(COMMANDS):
    folder = os.path.join(DATASET_PATH, command)
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    if len(files) == 0:
        print(f"WARNING: No .wav files found in {folder}", flush=True)
        continue
    print(f"Processing {len(files)} files for command '{command}'", flush=True)

    for i, file in enumerate(files):
        file_path = os.path.join(folder, file)
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=16000)
            
            # Extract MFCC (librosa 0.10+ syntax)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T

            # Pad or truncate to MAX_LEN
            if mfcc.shape[0] < MAX_LEN:
                pad_width = MAX_LEN - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_width), (0,0)), mode='constant')
            else:
                mfcc = mfcc[:MAX_LEN, :]

            X.append(mfcc)
            y.append(idx)

        except Exception as e:
            print(f"ERROR loading {file_path}: {e}", flush=True)

# ================= CONVERT TO TENSORS =================
X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.long)

# ================= SAVE DATASET =================
torch.save((X, y), "voice_dataset.pt")
print("Dataset prepared and saved as 'voice_dataset.pt'!", flush=True)
print("X shape:", X.shape, flush=True)
print("y shape:", y.shape, flush=True)
print("Sample labels:", y[:10], flush=True)
