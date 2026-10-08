import torch
import torch.nn as nn
import sounddevice as sd
import numpy as np
import librosa
import queue
import time

# ================= MODEL =================
SR = 16000
N_MFCC = 13
MAX_LEN = 50
COMMANDS = ["left", "right", "pause"]
CONF_THRESHOLD = 0.75
COOLDOWN = 0.7  # seconds

audio_q = queue.Queue()

class VoiceLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(13, 128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = VoiceLSTM()
model.load_state_dict(torch.load("voice_lstm_best.pth", map_location="cpu"))
model.eval()

last_command_time = 0

# ================= AUDIO =================
def callback(indata, frames, time_info, status):
    audio_q.put(indata.copy())

def extract_mfcc(audio):
    audio = audio.flatten()
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0,0),(0,MAX_LEN-mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    mfcc = mfcc.T
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

# ================= LISTENER =================
def listen_command():
    global last_command_time

    try:
        audio = audio_q.get_nowait()
    except:
        return None

    mfcc = extract_mfcc(audio)

    with torch.no_grad():
        out = model(mfcc)
        probs = torch.softmax(out, dim=1)[0]
        conf, idx = torch.max(probs, 0)

    if conf.item() >= CONF_THRESHOLD:
        now = time.time()
        if now - last_command_time > COOLDOWN:
            last_command_time = now
            return COMMANDS[idx]

    return None

def start_mic():
    stream = sd.InputStream(
        samplerate=SR,
        channels=1,
        callback=callback,
        blocksize=int(SR * 0.6)
    )
    stream.start()
