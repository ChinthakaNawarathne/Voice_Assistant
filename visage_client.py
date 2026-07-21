import cv2
import requests
import config
import atexit

VISAGE_API_URL = config.VISAGE_API_URL
VISAGE_API_KEY = config.VISAGE_API_KEY

_CAMERA = None


def _get_camera():
    global _CAMERA
    if _CAMERA is None:
        _CAMERA = cv2.VideoCapture(0)
        if not _CAMERA.isOpened():
            _CAMERA = None
    return _CAMERA


def _release_camera():
    global _CAMERA
    if _CAMERA is not None:
        _CAMERA.release()
        _CAMERA = None

atexit.register(_release_camera)


def _capture_frame():
    cap = _get_camera()
    if cap is None:
        return None
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return frame


def _encode_jpeg(frame) -> bytes | None:
    success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        return None
    return buf.tobytes()


def verify_face() -> str | None:
    try:
        frame = _capture_frame()
        if frame is None:
            print("[Visage] No camera frame captured.")
            return None

        image_bytes = _encode_jpeg(frame)
        if image_bytes is None:
            return None

        print(f"[Visage] Captured image: {len(image_bytes)} bytes")

        files = {"image": ("face.jpg", image_bytes, "image/jpeg")}
        headers = {"api": VISAGE_API_KEY, "user": "visage4humonoid"}

        resp = requests.post(VISAGE_API_URL, files=files, headers=headers, timeout=10)

        if resp.status_code != 200:
            print(f"[Visage] API returned status {resp.status_code}")
            try:
                print(f"[Visage] Response body: {resp.text[:500]}")
            except Exception:
                pass
            return None

        data = resp.json()
        print(f"[Visage] API response: {data}")
        user = data.get("user")
        if user and user.get("user_id"):
            user_id = str(user["user_id"])
            print(f"[Visage] Recognized user: {user_id}")
            return user_id

        print(f"[Visage] No user match in response.")
        return None

    except requests.RequestException as e:
        print(f"[Visage] Network error: {e}")
        return None
    except Exception as e:
        print(f"[Visage] Unexpected error: {e}")
        return None
