import pyaudio
import threading
import time
import torch
import audio_utils as audio_utils


class Listener:

    def __init__(self, sample_rate: int = 44000, buffer_len_ms: int = 1000, input_device: int = 0):

        self.sample_rate = sample_rate
        self.buffer_size = int((buffer_len_ms / 1000.0) * sample_rate)
        self.input_device = input_device
        self.chunk_size = 1024

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input_device_index=self.input_device,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        self.buffer = torch.tensor([[]], dtype=torch.int16)

    def listen(self):
        while True:

            bytes = self.stream.read(self.chunk_size, exception_on_overflow=False)

            array = torch.frombuffer(bytes, dtype=torch.int16)[None, :]
            self.buffer = torch.cat([self.buffer, array], dim=1)

            diff = self.buffer.shape[1] - self.buffer_size
            if diff >= 0:
                self.buffer = self.buffer[0:, diff:]

    def run(self):
        thread = threading.Thread(target=self.listen, daemon=True)
        thread.start()


class ClassifierEngine:

    def __init__(self, model_path: str, freq: float = 1, device: str = "cpu"):
        
        self.freq = freq
        self.device = device

        # Listener
        self.listener = Listener(sample_rate=44000, buffer_len_ms=1000, input_device=7)

        # Model
        self.model = torch.jit.load(model_path)
        self.model.to(device)
        self.model.eval()

    def predict(self, audio):
        with torch.no_grad():
            
            # Convert to suitable format
            audio = (audio.to(torch.float32), self.listener.sample_rate)

            audio = audio_utils.resample(audio, sample_rate=8000)
            audio = audio_utils.pad_truncate(audio, length_ms=1000)
            spec = audio_utils.mel_spectrogram(audio, 64, 400)[None,:,:,:]
            spec = spec.to(self.device)
            
            spec = (spec - spec.mean()) / spec.std()

            # Predict
            pred = self.model(spec)
            pred_class = torch.argmax(pred, dim=1)

            return pred_class.item()
        
    def routine(self):
        period = (1 / self.freq) * 1000
        while True:
            start_time = time.time_ns() / 1000000
            
            if self.listener.buffer.shape[1] > 0:
                print(self.predict(self.listener.buffer))

            end_time = time.time_ns() / 1000000
            
            delta = end_time - start_time
            time.sleep((period - delta) / 1000.0)

    def run(self):

        # Launch listener daemon
        self.listener.run()

        thread = threading.Thread(target=self.routine, daemon=False)
        thread.start()


if __name__ == "__main__":

    # listener = Listener(input_device=7)
    # listener.run()

    engine = ClassifierEngine("models/test_model_script_01.pth")
    engine.run()
