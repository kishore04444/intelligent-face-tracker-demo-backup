import cv2
from ultralytics import YOLO
import os

VIDEO_PATH = "videos/sample.mp4"
MODEL_PATH = "models/yolov8n-face.pt"
OUTPUT_DIR = "detection_debug"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading YOLO...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

print("FPS:", fps)
print("Testing frames: 10s, 20s, 30s, 40s")

for seconds in [10, 20, 30, 40]:

    frame_number = int(seconds * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()

    if not success:
        print(f"FAILED: {seconds}s")
        continue

    print(f"\n--- {seconds}s ---")

    results = model(
        frame,
        conf=0.30,
        verbose=False
    )

    result = results[0]

    count = 0

    if result.boxes is not None:
        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            confidence = float(
                box.conf[0]
            )

            width = x2 - x1
            height = y2 - y1

            count += 1

            print(
                f"Face {count}: "
                f"confidence={confidence:.2f}, "
                f"size={width}x{height}"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                f"{confidence:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    print("TOTAL FACES:", count)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"detection_{seconds}s.jpg"
    )

    cv2.imwrite(output_path, frame)

    print("Saved:", output_path)

cap.release()

print("\nDONE")