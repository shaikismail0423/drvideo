"""Frame sampling."""

import cv2

import config


def sample_frames(path, fps, max_frames):
    """Sample frames at `fps`, then thin them evenly if there are more than `max_frames`.

    Returns a list of (timestamp_in_seconds, jpeg_bytes).
    """
    capture = cv2.VideoCapture(path)
    video_fps = capture.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(video_fps / fps))  # keep one frame in every `step`

    frames = []
    index = 0
    while capture.grab():  # grab() is cheap; only decode the frames we keep
        if index % step == 0:
            ok, image = capture.retrieve()
            if ok:
                frames.append((index / video_fps, _to_jpeg(image)))
        index += 1
    capture.release()

    if len(frames) > max_frames:  # spread the kept frames over the whole video
        frames = [frames[i * len(frames) // max_frames] for i in range(max_frames)]
    return frames


def _to_jpeg(image):
    height, width = image.shape[:2]
    scale = config.FRAME_SIZE / max(height, width)
    if scale < 1:
        image = cv2.resize(image, (round(width * scale), round(height * scale)))
    return cv2.imencode(".jpg", image)[1].tobytes()
