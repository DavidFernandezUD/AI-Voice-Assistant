import pyaudio
import wave
import threading
import time


class Listener:

    def __init__(self, sample_rate: int = 48000, buffer_len: int = 5000, input_device_index: int = 0):
        self.sample_rate = sample_rate
        self.buffer_len = buffer_len
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input_device_index=input_device_index,
            input=True,
            frames_per_buffer=1024
        )

    def listen(self):
        buffer = []
        for _ in range(500):
            # Read audio data from the stream
            data = self.stream.read(1024)
        
            buffer.append(data)
            
            print(_)

        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

        file = wave.open("data/test_recording.wav", "wb")
        file.setnchannels(1)
        file.setframerate(self.sample_rate)
        file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        file.writeframes(b"".join(buffer))
        file.close()

    def run(self):
        thread = threading.Thread(target=self.listen, daemon=False)
        thread.start()


if __name__ == "__main__":

    listener = Listener(input_device_index=7)
    listener.run()
