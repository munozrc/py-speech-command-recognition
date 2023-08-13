from tensorflow import keras
import sounddevice as sd
import numpy as np
import librosa


class CommandsSpottingService:
    _instance = None
    _model = None
    _mapping = [
        "down",
        "go",
        "left",
        "no",
        "right",
        "stop",
        "up",
        "yes"
    ]

    def __new__(cls, model_path: str):
        if cls._instance is None:
            cls._instance = super(CommandsSpottingService, cls).__new__(cls)
            cls._model = keras.models.load_model(model_path)
        return cls._instance

    def predict(self, filename: str):
        MFCCs = self.preprocess(filename=filename)

        # (number of samples, number time steps, number coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = self._model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        return self._mapping[predicted_index]

    def preprocess(self, filename: str, n_mfcc=13, n_fft=2048, hop_length=512):
        signal, sample_rate = librosa.load(filename)

        SAMPLES_TO_CONSIDER = 22050  # One second worth of audio
        if len(signal) < SAMPLES_TO_CONSIDER:
            return

        # ensure consistency of the length of the signal
        signal = signal[:SAMPLES_TO_CONSIDER]
        MFCCs = librosa.feature.mfcc(
            y=signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return MFCCs.T


if __name__ == "__main__":
    service = CommandsSpottingService(model_path="model.h5")
    print(service.predict("audios/no/1b4c9b89_nohash_3.wav"))
    print(service.predict("audios/stop/0fa1e7a9_nohash_1.wav"))
    print(service.predict("audios/left/1cbcc7a7_nohash_1.wav"))
