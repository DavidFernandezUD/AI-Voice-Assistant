"""
Utility script to list input devices and their indexes
for PyAudio configuration.
"""


import pyaudio


def list_input_devices():
    p = pyaudio.PyAudio()
    device_count = p.get_device_count()

    print("Available input devices:")
    for idx in range(device_count):
        device_info = p.get_device_info_by_index(idx)
        if device_info["maxInputChannels"] > 0:
            print(f"Index: {idx}, Name: {device_info['name']}")


if __name__ == "__main__":

    list_input_devices()
