import cv2


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


class PersonDetector:
    def __init__(
        self,
        max_dimension=640,
        score_threshold=1.1,
        min_hits=2,
        max_missing=1,
        detector=None,
    ):
        self.max_dimension = max(320, int(max_dimension))
        self.score_threshold = float(score_threshold)
        self.min_hits = max(1, int(min_hits))
        self.max_missing = max(0, int(max_missing))
        self.detector = detector or cv2.HOGDescriptor()
        if detector is None:
            self.detector.setSVMDetector(
                cv2.HOGDescriptor_getDefaultPeopleDetector()
            )
        self.tracks = {}
        self.next_track_id = 1

    def reset(self):
        self.tracks.clear()
        self.next_track_id = 1

    def detect(self, frame):
        height, width = frame.shape[:2]
        scale = min(1.0, self.max_dimension / max(width, height))
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )

        boxes, weights = self.detector.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        candidates = [
            tuple(round(float(value) / scale) for value in box)
            for box, weight in zip(boxes, weights)
            if float(weight.item()) >= self.score_threshold
        ]

        for track in self.tracks.values():
            track["missing"] += 1

        matched_track_ids = set()
        for box in candidates:
            x, y, box_width, box_height = box
            box_xyxy = (x, y, x + box_width, y + box_height)
            best_id = None
            best_iou = 0.30
            for track_id, track in self.tracks.items():
                if track_id in matched_track_ids:
                    continue
                tx, ty, tw, th = track["box"]
                overlap = calculate_iou(
                    box_xyxy,
                    (tx, ty, tx + tw, ty + th),
                )
                if overlap >= best_iou:
                    best_id = track_id
                    best_iou = overlap

            if best_id is None:
                best_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[best_id] = {
                    "box": box,
                    "hits": 1,
                    "missing": 0,
                }
            else:
                track = self.tracks[best_id]
                track["box"] = box
                track["hits"] += 1
                track["missing"] = 0
            matched_track_ids.add(best_id)

        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if track["missing"] <= self.max_missing
        }
        return [
            (track_id, track["box"])
            for track_id, track in self.tracks.items()
            if track["hits"] >= self.min_hits
            and track["missing"] == 0
        ]
