import os
import unittest
from unittest.mock import patch

import cv2

from src.video_input import FFMPEG_CAPTURE_OPTIONS, open_capture


class FakeCapture:
    def __init__(self, opened):
        self.opened = opened
        self.released = False
        self.properties = []

    def isOpened(self):
        return self.opened

    def release(self):
        self.released = True

    def set(self, property_id, value):
        self.properties.append((property_id, value))


class VideoInputTests(unittest.TestCase):
    def setUp(self):
        self.previous_options = os.environ.get(FFMPEG_CAPTURE_OPTIONS)

    def tearDown(self):
        if self.previous_options is None:
            os.environ.pop(FFMPEG_CAPTURE_OPTIONS, None)
        else:
            os.environ[FFMPEG_CAPTURE_OPTIONS] = self.previous_options

    def test_rtsp_opens_using_tcp_transport_first(self):
        capture = FakeCapture(opened=True)

        with patch("src.video_input.cv2.VideoCapture", return_value=capture) as open_video:
            result = open_capture("rtsp://camera.local/stream")

        self.assertIs(result, capture)
        self.assertEqual(open_video.call_count, 1)
        self.assertEqual(
            capture.properties,
            [(cv2.CAP_PROP_BUFFERSIZE, 1)],
        )
        timeout_params = open_video.call_args.args[2]
        self.assertEqual(
            timeout_params,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                5000,
            ],
        )
        self.assertNotIn(FFMPEG_CAPTURE_OPTIONS, os.environ)

    def test_rtsp_falls_back_to_udp_if_tcp_does_not_open(self):
        tcp_capture = FakeCapture(opened=False)
        udp_capture = FakeCapture(opened=True)
        captures = [tcp_capture, udp_capture]

        with patch(
            "src.video_input.cv2.VideoCapture",
            side_effect=captures,
        ) as open_video:
            result = open_capture("rtsps://camera.local/stream")

        self.assertIs(result, udp_capture)
        self.assertTrue(tcp_capture.released)
        self.assertEqual(open_video.call_count, 2)
        self.assertEqual(
            udp_capture.properties,
            [(cv2.CAP_PROP_BUFFERSIZE, 1)],
        )
        self.assertNotIn(FFMPEG_CAPTURE_OPTIONS, os.environ)

    def test_rtsp_transport_setting_is_restored(self):
        capture = FakeCapture(opened=True)
        os.environ[FFMPEG_CAPTURE_OPTIONS] = "custom;option"

        with patch("src.video_input.cv2.VideoCapture", return_value=capture):
            open_capture("rtsp://camera.local/stream")

        self.assertEqual(
            os.environ[FFMPEG_CAPTURE_OPTIONS],
            "custom;option",
        )

    def test_non_rtsp_stream_does_not_set_rtsp_transport(self):
        capture = FakeCapture(opened=True)
        observed_options = []

        def open_video(*args):
            observed_options.append(os.environ.get(FFMPEG_CAPTURE_OPTIONS))
            return capture

        with patch("src.video_input.cv2.VideoCapture", side_effect=open_video):
            open_capture("rtmp://camera.local/live")

        self.assertEqual(observed_options, [None])

    def test_browser_source_uses_empty_capture(self):
        capture = FakeCapture(opened=False)

        with patch(
            "src.video_input.cv2.VideoCapture",
            return_value=capture,
        ) as open_video:
            result = open_capture("browser")

        self.assertIs(result, capture)
        open_video.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
