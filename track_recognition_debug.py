import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis

VIDEO_PATH = "videos/sample.mp4"
YOLO_MODEL = "models/yolov8n-face.pt"

print("Loading YOLO...")
yolo = YOLO(YOLO_MODEL)

print("Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

# Only inspect the important section of the video.
start_frame = int(15 * fps)
end_frame = int(35 * fps)

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    start_frame
)

print()
print("=" * 70)
print("TRACK + RECOGNITION DEBUG")
print("Testing video: 15s - 35s")
print("=" * 70)

frame_number = start_frame

seen_tracks = {}

while frame_number <= end_frame:

    success, frame = cap.read()

    if not success:
        break

    # Keep the same faster processing resolution.
    original_height, original_width = frame.shape[:2]

    small = cv2.resize(
        frame,
        (960, 544)
    )

    results = yolo.track(
        small,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.30,
        verbose=False
    )

    result = results[0]

    if result.boxes is None or result.boxes.id is None:
        frame_number += 1
        continue

    boxes = result.boxes.xyxy.cpu().numpy()
    track_ids = result.boxes.id.int().cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()

    scale_x = original_width / 960
    scale_y = original_height / 544

    for box, track_id, confidence in zip(
        boxes,
        track_ids,
        confidences
    ):

        sx1, sy1, sx2, sy2 = box

        x1 = int(sx1 * scale_x)
        y1 = int(sy1 * scale_y)
        x2 = int(sx2 * scale_x)
        y2 = int(sy2 * scale_y)

        width = x2 - x1
        height = y2 - y1

        if track_id not in seen_tracks:

            seen_tracks[track_id] = {
                "first_frame": frame_number,
                "last_frame": frame_number,
                "recognition_attempts": 0,
                "embedding_success": 0
            }

            print()
            print(
                f"[NEW TRACK] "
                f"frame={frame_number} "
                f"time={frame_number / fps:.1f}s "
                f"track_id={track_id}"
            )

        else:

            seen_tracks[track_id]["last_frame"] = frame_number

        # Try recognition on EVERY frame for this diagnostic.
        crop = frame[
            max(0, y1):min(original_height, y2),
            max(0, x1):min(original_width, x2)
        ]

        if crop.size == 0:
            continue

        # Add padding around face.
        padding_x = int(width * 0.25)
        padding_y = int(height * 0.25)

        cx1 = max(0, x1 - padding_x)
        cy1 = max(0, y1 - padding_y)
        cx2 = min(original_width, x2 + padding_x)
        cy2 = min(original_height, y2 + padding_y)

        crop = frame[cy1:cy2, cx1:cx2]

        seen_tracks[track_id]["recognition_attempts"] += 1

        try:

            faces = app.get(
                crop,
                max_num=1
            )

            if faces:

                seen_tracks[track_id]["embedding_success"] += 1

                print(
                    f"[EMBEDDING] "
                    f"time={frame_number / fps:.1f}s "
                    f"track_id={track_id} "
                    f"size={width}x{height} "
                    f"confidence={confidence:.2f}"
                )

        except Exception:
            pass

    frame_number += 1

cap.release()

print()
print("=" * 70)
print("TRACK SUMMARY")
print("=" * 70)

for track_id, data in sorted(seen_tracks.items()):

    duration = (
        data["last_frame"] -
        data["first_frame"]
    ) / fps

    print(
        f"track_id={track_id} | "
        f"duration={duration:.1f}s | "
        f"frames={data['last_frame'] - data['first_frame'] + 1} | "
        f"recognition_attempts={data['recognition_attempts']} | "
        f"embedding_success={data['embedding_success']}"
    )

print()
print("DEBUG COMPLETE")