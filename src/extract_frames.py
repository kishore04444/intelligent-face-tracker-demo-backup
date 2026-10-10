import cv2
import os

VIDEO_PATH = "videos/sample.mp4"
OUTPUT_DIR = "video_debug_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Opening video...")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    print("Video path:", os.path.abspath(VIDEO_PATH))
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", fps)
print("Total frames:", total_frames)

times = [0, 10, 20, 30, 40, 50]

created = 0

for seconds in times:

    frame_number = int(seconds * fps)

    print(
        f"Extracting {seconds}s "
        f"(frame {frame_number})..."
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )

    success, frame = cap.read()

    if not success:
        print(
            f"FAILED to read frame at {seconds}s"
        )
        continue

    output_path = os.path.join(
        OUTPUT_DIR,
        f"frame_{seconds}s.jpg"
    )

    success = cv2.imwrite(
        output_path,
        frame
    )

    if success:
        print(
            f"Created: {output_path}"
        )
        created += 1
    else:
        print(
            f"FAILED to save: {output_path}"
        )

cap.release()

print()
print("=" * 50)
print(f"{created} frames created")
print("=" * 50)