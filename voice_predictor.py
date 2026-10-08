import torch
import torch.nn as nn
import numpy as np
import sounddevice as sd
import librosa

# ================= SETTINGS =================
SR = 16000
DURATION = 1.0           # 1 second per recording to capture full "right"
N_MFCC = 13
MAX_LEN = 110
COMMANDS = ["left", "right", "pause"]
CONFIDENCE_THRESHOLD = 0.50  # lower to catch right reliably
VOLUME_THRESHOLD = 0.01      # ignore silence / low noise

# ================= MODEL =================
class VoiceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_MFCC, 128, batch_first=True)
        self.fc = nn.Linear(128, len(COMMANDS))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ================= LOAD MODEL =================
device = torch.device("cpu")
model = VoiceModel().to(device)
model.load_state_dict(torch.load("voice_model_best.pth", map_location=device))
model.eval()

# ================= RECORD AUDIO =================
def record_audio(duration=DURATION, sr=SR):
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    audio = audio.flatten()

    # Only process if user actually spoke
    if np.max(np.abs(audio)) < VOLUME_THRESHOLD:
        return None
    return audio

# ================= PREDICT COMMAND =================
def predict_command(audio):
    if audio is None:
        return None  # user didn't speak

    # normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # MFCC extraction
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SR, n_mfcc=N_MFCC, n_fft=512, hop_length=160
    ).T

    # standardize
    if mfcc.std() < 1e-6:
        mfcc = mfcc - mfcc.mean()
    else:
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)

    # pad / cut (centered for long words like "right")
    if mfcc.shape[0] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))
    else:
        start = (mfcc.shape[0] - MAX_LEN) // 2
        mfcc = mfcc[start:start + MAX_LEN]

    # convert to tensor
    X = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(X)
        probs = torch.softmax(out, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

        # strict detection: only left/right/pause if confident
        if confidence.item() < CONFIDENCE_THRESHOLD:
            return "invalid"
        return COMMANDS[pred_idx.item()]

# ================= REAL-TIME LOOP =================
print("🎤 Voice command predictor started! Say left, right, or pause.")

try:
    while True:
        audio = record_audio()
        if audio is not None:  # only when user speaks
            command = predict_command(audio)
            print(f"Detected command: {command}")
except KeyboardInterrupt:
    print("\n🛑 Exiting predictor.")
