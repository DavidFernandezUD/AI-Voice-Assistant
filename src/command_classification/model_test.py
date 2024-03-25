import torch
import torch.nn.functional as F
import torchaudio
from dataset import Featurizer
import pyaudio
import wave
import time
import sys
import os


CLASSES = [
    'no', 'learn', 'bed', 'marvin', 'zero',
    'six', 'yes', 'eight', 'up', 'on',
    'visual', 'sheila', 'wow', 'stop',
    'seven', 'house', 'nine', 'forward',
    'cat', 'follow', 'right', 'bird',
    'down', 'backward', 'four', 'off',
    'one', 'happy', 'go', 'two', 'dog',
    'five', 'three', 'left', 'tree'   
]


class CommandRecognizer:

    def __init__(self, model_path: str, sample_rate: int = 44000, channels: int = 1, input_device: int = 0):

        self.sample_rate = sample_rate
        self.channels = channels
        self.format = pyaudio.paInt16
        self.chunksize = 1024
        self.input_device = input_device

        self.audio = pyaudio.PyAudio()
        self.setream = self.audio.open(
            input_device_index=self.input_device,
            rate=self.sample_rate,
            channels=self.channels,
            format=self.format,
            input=True,
            frames_per_buffer=self.chunksize
        )

        self.featurizer = Featurizer(sample_rate, 16000, length_ms=2000, augment=False)

        self.model = torch.jit.load(model_path)
        self.model.to("cpu")
        self.model.eval()

    def record(self, seconds: float, path: str = "./", verbose: bool = True):
        """
        Records single audio shot of the specified length in .wav fromat.

        Args:
            seconds (float): Duration of the recording.
            path (str): Path the recording will be saved into.
            verbose (bool, optional): Display progress bar. (Default: True)
        """
        
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
        pred_class = CLASSES[class_idx]

        print(pred_class)

    def record_interactive(self, seconds: int, dir: str = "./", verbose: bool = True):
        """
        Interactively records single audio shot of the specified length
        in .wav fromat, and stores them in a specified directory.

        Args:
            seconds (float): Duration of the recordings.
            dir (str): Path the recordings will be saved into.
            verbose (bool, optional): Display progress bar. (Defaulf: True)
        """

        assert os.path.isdir(dir)

        print("Press enter to record and ctrl + c to exit")

        try:
            num_recording = 1
            while True:
                input("> ")
                time.sleep(0.2) # To supress key press sound
                
                path = os.path.join(dir, "temp.wav")
                num_recording += 1

                self.record(seconds, path, verbose)

                self.predict(path)

                os.remove(path)

        except KeyboardInterrupt:
            print("\nexit")


if __name__ == "__main__":

    recorder = CommandRecognizer("models/command_model_16KH.pth", sample_rate=48000, channels=1, input_device=7)
    recorder.record_interactive(seconds=2)
