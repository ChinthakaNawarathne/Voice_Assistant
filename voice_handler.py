import os
import tempfile
import speech_recognition as sr
import edge_tts
import playsound

TTS_VOICE = "en-AU-WilliamNeural"


class VoiceHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.6
        self.recognizer.phrase_threshold = 0.2
        self.recognizer.non_speaking_duration = 0.4

    def listen_to_user(self) -> tuple[str, sr.AudioData | None]:
        with sr.Microphone() as source:
            print("\n🎙️ Listening... Speak now.")
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=8)
                print("🔄 Processing speech...")
                text = self.recognizer.recognize_google(audio)
                print(f"🗣️ You said: {text}")
                return text.strip(), audio
            except (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError):
                return "", None

    def speak(self, text: str):
        print(f"🤖 Assistant: {text}")
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        try:
            edge_tts.Communicate(text, TTS_VOICE).save_sync(tmp.name)
            playsound.playsound(tmp.name, block=True)
        finally:
            os.unlink(tmp.name)
