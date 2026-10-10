import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import os

VIDEO_PATH = "videos/sample.mp4"
YOLO_MODEL = "models/yolov8n-face.pt"
OUTPUT_DIR = "recognition_debug"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

print("FPS:", fps)
print("Testing InsightFace on YOLO crops...")
print()

for seconds in [10, 20, 30, 40]:

    frame_number = int(seconds * fps)

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )

    success, frame = cap.read()

    if not success:
        print(f"FAILED TO READ {seconds}s")
        continue

    print("=" * 60)
    print(f"TIME: {seconds}s")

    results = yolo(
        frame,
        conf=0.30,
        verbose=False
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        print("No YOLO faces found")
        continue

    face_number = 0

    for box in result.boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        confidence = float(box.conf[0])

        width = x2 - x1
        height = y2 - y1

        face_number += 1

        print(
            f"\nFace {face_number}"
        )

        print(
            f"YOLO confidence : {confidence:.2f}"
        )

        print(
            f"YOLO size       : {width}x{height}"
        )

        # Add padding around the YOLO face
        padding_x = int(width * 0.25)
        padding_y = int(height * 0.25)

        cx1 = max(0, x1 - padding_x)
        cy1 = max(0, y1 - padding_y)
        cx2 = min(frame.shape[1], x2 + padding_x)
        cy2 = min(frame.shape[0], y2 + padding_y)

        crop = frame[
            cy1:cy2,
            cx1:cx2
        ]

        if crop.size == 0:
            print("❌ Empty crop")
            continue

        crop_height, crop_width = crop.shape[:2]

        print(
            f"Crop size       : "
            f"{crop_width}x{crop_height}"
        )

        # Save crop for inspection
        crop_path = os.path.join(
            OUTPUT_DIR,
            f"{seconds}s_face{face_number}.jpg"
        )

        cv2.imwrite(
            crop_path,
            crop
        )

        # Run InsightFace
        try:

            faces = app.get(
                crop,
                max_num=1
            )

            if not faces:

                print(
                    "❌ InsightFace: "
                    "NO FACE / NO EMBEDDING"
                )

                continue

            best_face = max(
                faces,
                key=lambda face:
                max(
                    0,
                    face.bbox[2] - face.bbox[0]
                )
                *
                max(
                    0,
                    face.bbox[3] - face.bbox[1]
                )
            )

            embedding = np.asarray(
                best_face.embedding,
                dtype=np.float32
            )

            norm = np.linalg.norm(
                embedding
            )

            if norm == 0:

                print(
                    "❌ InsightFace: "
                    "ZERO EMBEDDING"
                )

                continue

            embedding = embedding / norm

            print(
                "✅ InsightFace: "
                "EMBEDDING GENERATED"
            )

            print(
                f"Embedding length: "
                f"{len(embedding)}"
            )

        except Exception as e:

            print(
                "❌ InsightFace ERROR:"
            )

            print(e)

cap.release()

print()
print("=" * 60)
print("RECOGNITION TEST COMPLETE")
print("=" * 60)