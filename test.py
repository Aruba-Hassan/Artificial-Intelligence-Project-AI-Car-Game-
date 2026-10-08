import os
import librosa

DATASET_PATH = r"C:\Users\Dell\CarGame\VoiceDataset"
COMMANDS = ["left", "right", "pause"]

print("Testing dataset...", flush=True)

for command in COMMANDS:
    folder = os.path.join(DATASET_PATH, command)
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}", flush=True)
        continue
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    print(f"{command}: {len(files)} files found", flush=True)

    if len(files) > 0:
        try:
            audio, sr = librosa.load(os.path.join(folder, files[0]), sr=16000)
            print(f"First file '{files[0]}' loaded, shape={audio.shape}, sr={sr}", flush=True)
        except Exception as e:
            print(f"Error loading audio: {e}", flush=True)
