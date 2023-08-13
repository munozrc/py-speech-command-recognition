import librosa
import json
import os

AUDIOS_PATH = "audios"
JSON_PATH = "dataset.json"
SAMPLES_TO_CONSIDER = 22050  # One second worth of audio
NUM_MFCC = 13  # Number of coefficients to extract
N_FFT = 2048  # Interval we consider to apply FFT. Measured in number of samples
HOP_LENGTH = 512  # Sliding window for FFT. Measured in number of samples

data = {
    "mapping": [],
    "labels": [],
    "MFCCs": [],
    "files": []
}

commands_folders = os.listdir(AUDIOS_PATH)
num_commands = len(commands_folders)

for index, command in enumerate(commands_folders):
    command_folder = os.path.join(AUDIOS_PATH, command)
    command_audios = os.listdir(command_folder)
    num_audios = len(command_audios)

    data["mapping"].append(command)
    message = f"Preprocesing '{command}' of {index + 1}/{num_commands}"

    for ax, filename in enumerate(command_audios):
        audio_path = os.path.join(os.path.join(command_folder, filename))
        signal, sample_rate = librosa.load(audio_path)

        if len(signal) < SAMPLES_TO_CONSIDER:
            continue

        signal = signal[:SAMPLES_TO_CONSIDER]
        MFCCs = librosa.feature.mfcc(
            y=signal, sr=sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)

        data["labels"].append(index)
        data["files"].append(audio_path)
        data["MFCCs"].append(MFCCs.T.tolist())
        print(f"[{ax}/{num_audios} - {message}] {filename}")

with open(JSON_PATH, "w") as fp:
    json.dump(data, fp, indent=4)
