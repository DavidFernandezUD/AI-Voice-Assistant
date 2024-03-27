import torch
from wake_word.dataset import Featurizer
import pyaudio
import time


class WakewordEngine:

    def __init__(self, model_path: str, activation_threshold: float = 0.9, sample_rate: int = 44000, channels: int = 1, input_device: int = 0):

        self.activation_threshold = activation_threshold

        self.sample_rate = sample_rate
        self.channels = channels
        self.format = pyaudio.paFloat32
        self.chunksize = 1024
        self.input_device = input_device

        self.audio = pyaudio.PyAudio()

        self.featurizer = Featurizer(16000, length_ms=2000, augment=False)

        self.model = torch.jit.load(model_path)
        self.model.to("cpu")
        self.model.eval()

    def listen(self, seconds: float):
        """
        Records single audio shot of the specified length in .wav fromat.

        Args:
            seconds (float): Duration of the recording.
            path (str): Path the recording will be saved into.
            verbose (bool, optional): Display progress bar. (Default: True)
        """

        self.setream = self.audio.open(
            input_device_index=self.input_device,
            rate=self.sample_rate,
            channels=self.channels,
            format=self.format,
            input=True,
            frames_per_buffer=self.chunksize
        )
        
        while True:

            frames = []
            start_time = time.monotonic()

            while (time.monotonic() - start_time) < seconds:
                data = self.setream.read(self.chunksize, exception_on_overflow=False)
                frames.append(torch.frombuffer(data, dtype=torch.float32))

            buffer = torch.concatenate(frames).unsqueeze(dim=0)
            pred = self.predict(buffer)

            if pred > self.activation_threshold:
                self.setream.close()
                return
        
    def predict(self, waveform: str):

        spec = self.featurizer(waveform, self.sample_rate).unsqueeze(dim=0)
        
        pred = self.model(spec)
        
        return pred


if __name__ == "__main__":

    recorder = WakewordEngine("models/wakeword_model_16KH.pth", sample_rate=48000, channels=1, input_device=7)
    recorder.listen(2)
