"""Frame sampling."""

import cv2

import config


def sample_frames(path, fps, max_frames):
    """Sample frames at `fps` (sparser if that would exceed `max_frames`).

    Returns a list of (timestamp_in_seconds, jpeg_bytes).
    """
    capture = cv2.VideoCapture(path)
    video_fps = capture.get(cv2.CAP_PROP_FPS) or 25
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(1, round(video_fps / fps))
    if total / step > max_frames:
        step = -(-total // max_frames)  # ceiling division

    frames = []
    index = 0
    while len(frames) < max_frames and capture.grab():
        if index % step == 0:
            ok, image = capture.retrieve()
            if ok:
                frames.append((index / video_fps, _to_jpeg(image)))
        index += 1
    capture.release()
    return frames


def _to_jpeg(image):
    height, width = image.shape[:2]
    scale = config.FRAME_SIZE / max(height, width)
    if scale < 1:
        image = cv2.resize(image, (round(width * scale), round(height * scale)))
    return cv2.imencode(".jpg", image)[1].tobytes()
