from audio_emotion_onnx import OnnxEmotionDetector
from transformers import pipeline


class EmotionTracker:
    def __init__(self, audio_quantization="fp16"):
        self.audio_detector = OnnxEmotionDetector(quantized=audio_quantization)

        self.text_classifier = pipeline(
            "text-classification",
            model="bhadresh-savani/bert-base-uncased-emotion",
            top_k=1,
        )

        self.text_label_map = {
            "anger": "angry", "fear": "fearful", "joy": "happy", "love": "happy",
            "sadness": "sad", "surprise": "surprised", "neutral": "neutral",
        }

    def _analyze_text(self, text: str):
        if not text.strip():
            return "neutral", 0.0
        result = self.text_classifier(text[:512], top_k=1)
        top = result[0]
        label = self.text_label_map.get(top["label"].lower(), "neutral")
        return label, round(top["score"] * 100, 1)

    def analyze(self, audio_bytes: bytes, sample_rate: int, text: str = "") -> tuple[str, float]:
        audio_emoji = "neutral"
        audio_conf = 0.0
        text_emoji = "neutral"
        text_conf = 0.0

        try:
            audio_emoji, audio_conf = self.audio_detector.analyze(audio_bytes, sample_rate)
        except Exception as e:
            print(f"[EmotionTracker] Audio ONNX error: {e}")

        try:
            text_emoji, text_conf = self._analyze_text(text)
        except Exception as e:
            print(f"[EmotionTracker] Text error: {e}")

        if audio_conf >= 70.0:
            return audio_emoji, round(audio_conf, 1)
        if text_conf >= 70.0 and text_emoji != "neutral":
            return text_emoji, round(text_conf, 1)
        if audio_conf >= text_conf:
            return audio_emoji, round(audio_conf, 1)
        return text_emoji, round(text_conf, 1)
