from wake_word import engine as ww_engine
from command_classification import engine as cmd_engine


class VoiceAssistant:

    def __init__(
            self,
            wakeword_model_path: str,
            command_model_path: str,
            sample_rate: int = 44000,
            channels: int = 1,
            input_device: int = 1
            ):

        self.wakeword_emgine = ww_engine.WakewordEngine(wakeword_model_path, sample_rate=sample_rate, channels=channels, input_device=input_device)
        self.command_engine = cmd_engine.CommandEngine(command_model_path, sample_rate=sample_rate, channels=channels, input_device=input_device)

    def run(self):

        while True:

            self.wakeword_emgine.listen(2)
            pred = self.command_engine.listen(2)

            print(pred)


if __name__ == "__main__":

    assistant = VoiceAssistant("models/wakeword_model_16KH.pth", "models/command_model_16KH.pth", sample_rate=48000, input_device=4)
    assistant.run()
