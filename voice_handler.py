import speech_recognition as sr
import pyttsx3

class VoiceHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        
        # SPEED OPTIMIZATION: Reduce the amount of silence time required before cutting off audio capture
        self.recognizer.pause_threshold = 0.6          # Seconds of silence before a phrase is considered complete
        self.recognizer.phrase_threshold = 0.2         # Minimum seconds of speaking audio to touch
        self.recognizer.non_speaking_duration = 0.4    # Keeping silence windows lean
        
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 190)       # Slightly faster speech rate for snappier responses

    def listen_to_user(self) -> str:
        with sr.Microphone() as source:
            print("\n🎙️ Listening... Speak now.")
            try:
                # Optimized timeouts
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=8)
                print("🔄 Processing speech...")
                text = self.recognizer.recognize_google(audio)
                print(f"🗣️ You said: {text}")
                return text.strip()
            except (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError):
                return ""

    def speak(self, text: str):
        print(f"🤖 Assistant: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()