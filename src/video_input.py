import os
from urllib.parse import urlsplit

import cv2


RTSP_SCHEMES = {"rtsp", "rtsps"}
LIVE_SCHEMES = {
    "http",
    "https",
    "rtmp",
    "rtmps",
    *RTSP_SCHEMES,
}
FFMPEG_CAPTURE_OPTIONS = "OPENCV_FFMPEG_CAPTURE_OPTIONS"


def _open_ffmpeg_capture(source, timeout_params, rtsp_transport=None):
    previous_options = os.environ.get(FFMPEG_CAPTURE_OPTIONS)
    try:
        if rtsp_transport:
            os.environ[FFMPEG_CAPTURE_OPTIONS] = (
                f"rtsp_transport;{rtsp_transport}"
            )
        return cv2.VideoCapture(
            source,
            cv2.CAP_FFMPEG,
            timeout_params,
        )
    finally:
        if previous_options is None:
            os.environ.pop(FFMPEG_CAPTURE_OPTIONS, None)
        else:
            os.environ[FFMPEG_CAPTURE_OPTIONS] = previous_options


def open_capture(source):
    if source == "browser":
        return cv2.VideoCapture()

    if isinstance(source, str):
        try:
            scheme = urlsplit(source).scheme.lower()
        except ValueError:
            scheme = ""

        if scheme in LIVE_SCHEMES:
            timeout_params = [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                5000,
            ]
            if scheme in RTSP_SCHEMES:
                capture = _open_ffmpeg_capture(
                    source,
                    timeout_params,
                    rtsp_transport="tcp",
                )
                if capture.isOpened():
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return capture
                capture.release()

                capture = _open_ffmpeg_capture(
                    source,
                    timeout_params,
                    rtsp_transport="udp",
                )
            else:
                capture = _open_ffmpeg_capture(source, timeout_params)

            if capture.isOpened():
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return capture

    capture = cv2.VideoCapture(source)
    if isinstance(source, int) and capture.isOpened():
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture
