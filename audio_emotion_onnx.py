import numpy as np
from huggingface_hub import hf_hub_download
import onnxruntime as ort

REPO_ID = "prithivMLmods/Speech-Emotion-Classification-ONNX"
TARGET_SR = 16000

QUANT_PATHS = {
    "fp32": "onnx/model.onnx",
    "fp16": "onnx/model_fp16.onnx",
    "int8": "onnx/model_int8.onnx",
    "q4": "onnx/model_q4.onnx",
}

CLASS_MAP = ["angry", "neutral", "disgusted", "fearful",
             "happy", "neutral", "sad", "surprised"]
# 0:ANG  1:CAL→neutral  2:DIS  3:FEA  4:HAP  5:NEU  6:SAD  7:SUR


class OnnxEmotionDetector:
    def __init__(self, model_path=None, quantized="fp16"):
        self._session = None
        self._input_name = None

        if model_path is None:
            filename = QUANT_PATHS.get(quantized, QUANT_PATHS["fp16"])
            model_path = hf_hub_download(repo_id=REPO_ID, filename=filename)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self._session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def analyze(self, audio_bytes: bytes, sample_rate: int):
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio) == 0:
            return "neutral", 0.0

        if sample_rate != TARGET_SR:
            old_len = len(audio)
            new_len = int(old_len * TARGET_SR / sample_rate)
            audio = np.interp(
                np.linspace(0, old_len - 1, new_len),
                np.arange(old_len),
                audio,
            )

        mu, sigma = audio.mean(), audio.std()
        if sigma > 1e-10:
            audio = (audio - mu) / sigma

        input_values = audio[np.newaxis, :].astype(np.float32)

        logits = self._session.run(None, {self._input_name: input_values})[0]

        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)

        pred_id = int(np.argmax(probs[0]))
        confidence = float(probs[0, pred_id])

        return CLASS_MAP[pred_id], round(confidence * 100, 1)
