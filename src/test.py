import pyaudio
import numpy as np

pa = pyaudio.PyAudio()
device_index = 7  # Index corresponding to your Scarlett 2i2 USB device

# Define audio stream parameters
chunk_size = 1024  # Number of frames per buffer
sample_format = pyaudio.paInt16
channels = 2  # Specify the correct number of channels
sample_rate = 48000

try:
    stream = pa.open(format=sample_format,
                     channels=channels,
                     rate=sample_rate,
                     input=True,
                     input_device_index=device_index,
                     frames_per_buffer=chunk_size)

    print("Recording started. Press Ctrl+C to stop.")

    while True:
        # Read audio data from the stream
        data = stream.read(chunk_size)
        
        # Convert binary data to numpy array
        audio_array = np.frombuffer(data, dtype=np.int16)
        
        # Print the first few samples of audio data
        print(audio_array[:10])  # Print the first 10 samples
        
except KeyboardInterrupt:
    print("Recording stopped.")

finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()