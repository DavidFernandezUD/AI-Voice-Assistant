import torch
import torch.nn.functional as F
import torchaudio
from command_classification.dataset import Featurizer
import pyaudio
import wave
import time
import sys
import os


class CommandEngine:

    def __init__(self, model_path: str, sample_rate: int = 44000, channels: int = 1, input_device: int = 0):

        self.CLASSES = [
            'no', 'learn', 'bed', 'marvin', 'zero',
            'six', 'yes', 'eight', 'up', 'on',
            'visual', 'sheila', 'wow', 'stop',
            'seven', 'house', 'nine', 'forward',
            'cat', 'follow', 'right', 'bird',
            'down', 'backward', 'four', 'off',
            'one', 'happy', 'go', 'two', 'dog',
            'five', 'three', 'left', 'tree'   
        ]

        self.sample_rate = sample_rate
        self.channels = channels
        self.format = pyaudio.paInt16
        self.chunksize = 1024
        self.input_device = input_device

        self.audio = pyaudio.PyAudio()


        self.featurizer = Featurizer(sample_rate, 16000, length_ms=2000, augment=False)

        self.model = torch.jit.load(model_path)
        self.model.to("cpu")
        self.model.eval()

    def listen(self, seconds: float, path: str = "./temp.wav", verbose: bool = True) -> str:
        """
        Records single audio shot of the specified length.

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
        
        frames = []
        start_time = time.monotonic()

        while (time.monotonic() - start_time) < seconds:
            data = self.setream.read(self.chunksize, exception_on_overflow=False)
            frames.append(data)

            if verbose:
                self._print_progress_bar((time.monotonic() - start_time) / seconds)

        with wave.open(path, "wb") as file:
            file.setframerate(self.sample_rate)
            file.setnchannels(self.channels)
            file.setsampwidth(self.audio.get_sample_size(self.format))
            file.writeframes(b"".join(frames))
            print()

        pred_class = self.predict(path)
        os.remove(path)

        self.setream.close()

        return pred_class

    def _print_progress_bar(self, progress: float):

        bar_length = 40
        filled_length = int(bar_length * progress)

        bar = "=" * filled_length + "-" * (bar_length - filled_length) 

        sys.stdout.write(f"\rRecording [{bar}] {int(progress * 100)}%")
        sys.stdout.flush()

    def predict(self, path: str):

        audio, _ = torchaudio.load(path)
        spec = self.featurizer(audio).unsqueeze(dim=0)
        
        logits = self.model(spec)
        preds = F.softmax(logits, dim=1)
        class_idx = torch.argmax(preds)
        pred_class = self.CLASSES[class_idx]

        return pred_class
    

if __name__ == "__main__":

    engine = CommandEngine("models/command_model_16KH.pth", sample_rate=44000, channels=1, input_device=7)
    print(engine.listen(2))