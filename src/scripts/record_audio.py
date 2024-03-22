"""
Script to record and save audio from input device.

To record a single audio shot, specify the --seconds of the recording
and --path to store the recording.

To interactively record multiple audio shots add --interactive.

When recording, a progress bar will be displayed, and after the recording
finishes, the recorded audio will be stored in the specified directory.
"""


import pyaudio
import wave
import time
import argparse
import sys
import os


class Recorder:

    def __init__(self, sample_rate: int = 44000, channels: int = 1, input_device: int = 0):

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

    def record(self, seconds: float, path: str, verbose: bool = True):
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

    def _print_progress_bar(self, progress: float):

        bar_length = 40
        filled_length = int(bar_length * progress)

        bar = "=" * filled_length + "-" * (bar_length - filled_length) 

        sys.stdout.write(f"\rRecording [{bar}] {int(progress * 100)}%")
        sys.stdout.flush()

    def record_interactive(self, seconds: int, dir: str, verbose: bool = True):
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
                
                path = os.path.join(dir, f"{num_recording}.wav")
                num_recording += 1

                self.record(seconds, path, verbose)
        except KeyboardInterrupt:
            print("\nexit")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--seconds", type=float, default=None, required=True)
    parser.add_argument("--path", type=str, default=None, required=True)
    parser.add_argument("--freq", type=int, default=44000, required=False)
    parser.add_argument("--channels", type=int, default=1, required=False)
    parser.add_argument("--device", type=int, default=0, required=False)
    parser.add_argument("--interactive", default=False, action='store_true', required=False)

    args = parser.parse_args()

    recorder = Recorder(sample_rate=args.freq, channels=args.channels, input_device=args.device)
    
    if args.interactive:
        recorder.record_interactive(seconds=args.seconds, dir=args.path)
    else:
        recorder.record(seconds=args.seconds, path=args.path)

    # Examples:
    # python record_audio.py --interactive --seconds 2 --path data/recordings/ --freq 48000 --device 7
    # python record_audio.py --seconds 2 --path data/recordings/ --freq 44000 --channels 2