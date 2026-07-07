import numpy as np
import scipy.signal
from transformers import pipeline

class EmotionTracker:
    def __init__(self):
        self.audio_classifier = pipeline(
            "audio-classification",
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            top_k=1
        )
        self.text_classifier = pipeline(
            "text-classification",
            model="bhadresh-savani/bert-base-uncased-emotion",
            top_k=1
        )
        self.target_sr = 16000

        self.audio_label_map = {
            "angry": "angry", "disgust": "disgusted", "fear": "fearful",
            "happy": "happy", "neutral": "neutral", "sad": "sad", "surprise": "surprised"
        }
        self.text_label_map = {
            "anger": "angry", "fear": "fearful", "joy": "happy", "love": "happy",
            "sadness": "sad", "surprise": "surprised", "neutral": "neutral"
        }

    def _analyze_audio(self, audio_bytes: bytes, sample_rate: int):
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio_array) == 0:
            return "neutral", 0.0
        if sample_rate != self.target_sr:
            num_samples = int(len(audio_array) * self.target_sr / sample_rate)
            audio_array = scipy.signal.resample(audio_array, num_samples)
        result = self.audio_classifier(audio_array, sampling_rate=self.target_sr)
        top = result[0]
        label = self.audio_label_map.get(top["label"].lower(), "neutral")
        return label, top["score"]

    def _analyze_text(self, text: str):
        if not text.strip():
            return "neutral", 0.0
        result = self.text_classifier(text[:512], top_k=1)
        top = result[0]
        label = self.text_label_map.get(top["label"].lower(), "neutral")
        return label, top["score"]

    def analyze(self, audio_bytes: bytes, sample_rate: int, text: str = "") -> tuple[str, float]:
        audio_emoji = "neutral"
        audio_conf = 0.0
        text_emoji = "neutral"
        text_conf = 0.0

        try:
            audio_emoji, audio_conf = self._analyze_audio(audio_bytes, sample_rate)
        except Exception as e:
            print(f"[EmotionTracker] Audio error: {e}")

        try:
            text_emoji, text_conf = self._analyze_text(text)
        except Exception as e:
            print(f"[EmotionTracker] Text error: {e}")

        # Weighted ensemble: prefer audio when confident, otherwise blend
        if audio_conf >= 0.7:
            return audio_emoji, round(audio_conf * 100, 1)
        if text_conf >= 0.7 and text_emoji != "neutral":
            return text_emoji, round(text_conf * 100, 1)
        if audio_conf >= text_conf:
            return audio_emoji, round(audio_conf * 100, 1)
        return text_emoji, round(text_conf * 100, 1)
