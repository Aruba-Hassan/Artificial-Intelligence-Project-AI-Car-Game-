import torch
import torch.nn as nn
import numpy as np
import sounddevice as sd
import librosa

SAMPLE_RATE = 16000
N_MFCC = 13
MAX_LEN = 60

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

class VoiceController:
    def __init__(self, model_path):
        self.device = torch.device("cpu")
        self.model = VoiceLSTM().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.buffer = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self.audio_callback
        )
        self.stream.start()

    def audio_callback(self, indata, frames, time, status):
        self.buffer = np.roll(self.buffer, -frames)
        self.buffer[-frames:] = indata[:, 0]

    def listen(self):
        mfcc = librosa.feature.mfcc(
            y=self.buffer, sr=SAMPLE_RATE, n_mfcc=N_MFCC
        ).T

        if mfcc.shape[0] > MAX_LEN:
            mfcc = mfcc[:MAX_LEN]
        else:
            mfcc = np.pad(mfcc, ((0, MAX_LEN-mfcc.shape[0]), (0,0)))

        x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            out = self.model(x)
            pred = out.argmax(1).item()

        return ["left", "right", "pause"][pred]
