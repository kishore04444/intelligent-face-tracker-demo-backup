import unittest

import numpy as np

from src.person_detection import PersonDetector


class FakeHogDetector:
    def __init__(self, detections):
        self.detections = iter(detections)

    def detectMultiScale(self, *_args, **_kwargs):
        return next(self.detections)


def detection(boxes, scores):
    return np.asarray(boxes, dtype=np.int32).reshape((-1, 4)), np.asarray(
        scores,
        dtype=np.float32,
    ).reshape((-1, 1))


class PersonDetectionTests(unittest.TestCase):
    def test_ignores_low_score_false_positive_and_requires_repeated_detection(self):
        detector = FakeHogDetector(
            [
                detection([(10, 10, 40, 80), (100, 10, 40, 80)], [0.7, 1.4]),
                detection([(102, 11, 40, 80)], [1.5]),
            ]
        )
        person_detector = PersonDetector(
            detector=detector,
            score_threshold=1.0,
            min_hits=2,
        )
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        self.assertEqual(person_detector.detect(frame), [])
        self.assertEqual(
            person_detector.detect(frame),
            [(1, (102, 11, 40, 80))],
        )

    def test_single_frame_false_positive_never_becomes_visible(self):
        detector = FakeHogDetector(
            [
                detection([(10, 10, 40, 80)], [1.3]),
                detection([], []),
            ]
        )
        person_detector = PersonDetector(detector=detector, min_hits=2)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        self.assertEqual(person_detector.detect(frame), [])
        self.assertEqual(person_detector.detect(frame), [])

    def test_reset_clears_temporal_person_tracks(self):
        detector = FakeHogDetector(
            [
                detection([(10, 10, 40, 80)], [1.3]),
                detection([(10, 10, 40, 80)], [1.3]),
                detection([(10, 10, 40, 80)], [1.3]),
            ]
        )
        person_detector = PersonDetector(detector=detector, min_hits=2)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        person_detector.detect(frame)
        person_detector.reset()

        self.assertEqual(person_detector.detect(frame), [])


if __name__ == "__main__":
    unittest.main()
