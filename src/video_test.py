import os
import sys
import cv2
from collections import defaultdict

# Add project src directory to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ultralytics import YOLO

from face_recognizer import FaceRecognizer
from visitor_manager import VisitorManager
from logger import log_event
from image_logger import save_face_image


VIDEO_PATH = os.path.join(BASE_DIR, "videos", "sample.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n-face.pt")

# Fast YOLO processing resolution
PROCESS_WIDTH = 960
PROCESS_HEIGHT = 544

# Run YOLO + ByteTrack every Nth frame
DETECTION_INTERVAL = 2

# Run InsightFace periodically for each track
RECOGNITION_INTERVAL = 8

# Minimum detected face size in the resized frame
MIN_FACE_WIDTH = 30
MIN_FACE_HEIGHT = 30

# Frames a track can disappear before being considered exited
MAX_MISSING_FRAMES = 20

# YOLO confidence
CONFIDENCE = 0.50


def main():

    print("=" * 60)
    print("INTELLIGENT FACE TRACKER - SAMPLE VIDEO TEST")
    print("=" * 60)

    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video not found: {VIDEO_PATH}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: YOLO model not found: {MODEL_PATH}")
        return

    print(f"Video : {VIDEO_PATH}")
    print(f"Model : {MODEL_PATH}")
    print(
        f"Processing resolution: "
        f"{PROCESS_WIDTH}x{PROCESS_HEIGHT}"
    )
    print(f"Detection interval: {DETECTION_INTERVAL}")
    print(f"Recognition interval: {RECOGNITION_INTERVAL}")
    print()

    # ---------------------------------------------------------
    # LOAD MODELS
    # ---------------------------------------------------------

    print("Loading YOLO...")
    model = YOLO(MODEL_PATH)

    print("Loading InsightFace...")
    recognizer = FaceRecognizer()

    print("Loading visitor manager...")
    manager = VisitorManager(similarity_threshold=0.5)

    # ---------------------------------------------------------
    # OPEN VIDEO
    # ---------------------------------------------------------

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Could not open sample video.")
        manager.close()
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    original_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    original_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print()
    print("VIDEO INFORMATION")
    print(
        f"Original resolution : "
        f"{original_width}x{original_height}"
    )
    print(f"FPS                 : {original_fps:.2f}")
    print(f"Frames              : {total_frames}")

    if original_fps > 0:
        print(
            f"Duration            : "
            f"{total_frames / original_fps:.2f} seconds"
        )

    print()

    # ---------------------------------------------------------
    # RUNTIME STATE
    # ---------------------------------------------------------

    frame_number = 0
    processed_frames = 0

    track_last_seen = {}
    track_to_bbox = {}

    recognition_last_frame = defaultdict(
        lambda: -RECOGNITION_INTERVAL
    )

    recognition_attempts = defaultdict(int)

    entry_logged_tracks = set()

    unique_visitors = set()

    print("Starting video processing...")
    print("Press Q to stop.")
    print()

    # ---------------------------------------------------------
    # MAIN VIDEO LOOP
    # ---------------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # -----------------------------------------------------
        # FRAME SKIPPING
        # -----------------------------------------------------

        if frame_number % DETECTION_INTERVAL != 0:
            continue

        processed_frames += 1

        # -----------------------------------------------------
        # RESIZE ONLY FOR YOLO
        # -----------------------------------------------------

        frame_small = cv2.resize(
            frame,
            (PROCESS_WIDTH, PROCESS_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        # -----------------------------------------------------
        # YOLO + BYTE TRACK
        # -----------------------------------------------------

        results = model.track(
            frame_small,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE,
            verbose=False
        )

        if not results:
            continue

        result = results[0]

        current_tracks = set()

        if result.boxes is not None:

            boxes = result.boxes

            if (
                boxes.xyxy is not None
                and len(boxes.xyxy) > 0
            ):

                xyxy = boxes.xyxy.cpu().numpy()

                if boxes.id is not None:
                    track_ids = (
                        boxes.id.int()
                        .cpu()
                        .tolist()
                    )
                else:
                    track_ids = []

                # -------------------------------------------------
                # PROCESS EACH TRACK
                # -------------------------------------------------

                for bbox, track_id in zip(
                    xyxy,
                    track_ids
                ):

                    x1, y1, x2, y2 = map(
                        int,
                        bbox
                    )

                    # Keep coordinates inside resized frame
                    x1 = max(
                        0,
                        min(PROCESS_WIDTH, x1)
                    )

                    y1 = max(
                        0,
                        min(PROCESS_HEIGHT, y1)
                    )

                    x2 = max(
                        0,
                        min(PROCESS_WIDTH, x2)
                    )

                    y2 = max(
                        0,
                        min(PROCESS_HEIGHT, y2)
                    )

                    width = x2 - x1
                    height = y2 - y1

                    # Ignore extremely tiny detections
                    if (
                        width < MIN_FACE_WIDTH
                        or height < MIN_FACE_HEIGHT
                    ):
                        continue

                    current_tracks.add(track_id)

                    track_last_seen[track_id] = (
                        processed_frames
                    )

                    track_to_bbox[track_id] = (
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    # -------------------------------------------------
                    # DEBUG TRACK OUTPUT
                    # -------------------------------------------------

                    print(
                        f"[TRACK] "
                        f"frame={frame_number} "
                        f"track_id={track_id} "
                        f"face_size={width}x{height}"
                    )

                    # -------------------------------------------------
                    # DRAW TRACK BOX
                    # -------------------------------------------------

                    cv2.rectangle(
                        frame_small,
                        (x1, y1),
                        (x2, y2),
                        (255, 180, 0),
                        2
                    )

                    label = f"Track {track_id}"

                    # -------------------------------------------------
                    # RECOGNITION INTERVAL
                    # -------------------------------------------------

                    should_recognize = (
                        processed_frames
                        - recognition_last_frame[track_id]
                    ) >= RECOGNITION_INTERVAL

                    if should_recognize:

                        recognition_last_frame[
                            track_id
                        ] = processed_frames

                        recognition_attempts[
                            track_id
                        ] += 1

                        # -------------------------------------------------
                        # CONVERT SMALL-FRAME BOX TO ORIGINAL-FRAME BOX
                        # -------------------------------------------------

                        scale_x = (
                            original_width
                            / PROCESS_WIDTH
                        )

                        scale_y = (
                            original_height
                            / PROCESS_HEIGHT
                        )

                        orig_x1 = int(
                            x1 * scale_x
                        )

                        orig_y1 = int(
                            y1 * scale_y
                        )

                        orig_x2 = int(
                            x2 * scale_x
                        )

                        orig_y2 = int(
                            y2 * scale_y
                        )

                        orig_x1 = max(
                            0,
                            orig_x1
                        )

                        orig_y1 = max(
                            0,
                            orig_y1
                        )

                        orig_x2 = min(
                            original_width,
                            orig_x2
                        )

                        orig_y2 = min(
                            original_height,
                            orig_y2
                        )

                        # -------------------------------------------------
                        # INSIGHTFACE ON ORIGINAL HIGH-RESOLUTION FRAME
                        # -------------------------------------------------

                        embedding = (
                            recognizer.get_embedding(
                                frame.copy(),
                                (
                                    orig_x1,
                                    orig_y1,
                                    orig_x2,
                                    orig_y2
                                )
                            )
                        )

                        if embedding is not None:

                            visitor_id, is_new = (
                                manager.identify(
                                    track_id,
                                    embedding
                                )
                            )

                            if visitor_id is not None:

                                unique_visitors.add(
                                    visitor_id
                                )

                                # -------------------------------------------------
                                # ONE ENTRY PER PHYSICAL TRACK
                                # -------------------------------------------------

                                if (
                                    track_id
                                    not in entry_logged_tracks
                                ):

                                    face_crop = frame[
                                        orig_y1:orig_y2,
                                        orig_x1:orig_x2
                                    ]

                                    image_path = None

                                    if face_crop.size > 0:

                                        image_path = (
                                            save_face_image(
                                                face_crop,
                                                "ENTRY"
                                            )
                                        )

                                    manager.db.add_event(
                                        visitor_id,
                                        "ENTRY",
                                        image_path
                                    )

                                    log_event(
                                        f"ENTRY | "
                                        f"visitor_id={visitor_id} "
                                        f"| track_id={track_id}"
                                    )

                                    entry_logged_tracks.add(
                                        track_id
                                    )

                                label = (
                                    f"Visitor {visitor_id}"
                                )

                                if is_new:

                                    print(
                                        f"[NEW VISITOR] "
                                        f"Visitor {visitor_id} "
                                        f"(track {track_id})"
                                    )

                                else:

                                    print(
                                        f"[RECOGNIZED] "
                                        f"Visitor {visitor_id} "
                                        f"(track {track_id})"
                                    )

                            else:

                                label = (
                                    f"Track {track_id} "
                                    f"| Recognizing"
                                )

                        else:

                            label = (
                                f"Track {track_id} "
                                f"| No face"
                            )

                    # -------------------------------------------------
                    # DRAW LABEL
                    # -------------------------------------------------

                    cv2.putText(
                        frame_small,
                        label,
                        (
                            x1,
                            max(25, y1 - 8)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )

        # ---------------------------------------------------------
        # DETECT TRACKS THAT DISAPPEARED
        # ---------------------------------------------------------

        tracks_to_remove = []

        for track_id, last_seen in (
            track_last_seen.items()
        ):

            if (
                processed_frames - last_seen
                > MAX_MISSING_FRAMES
            ):

                tracks_to_remove.append(
                    track_id
                )

        # ---------------------------------------------------------
        # HANDLE EXITS
        # ---------------------------------------------------------

        for track_id in tracks_to_remove:

            visitor_id = manager.remove_track(
                track_id
            )

            if visitor_id is not None:

                bbox = track_to_bbox.get(
                    track_id
                )

                image_path = None

                if bbox is not None:

                    (
                        small_x1,
                        small_y1,
                        small_x2,
                        small_y2
                    ) = bbox

                    # Convert last known box back
                    # to original resolution
                    scale_x = (
                        original_width
                        / PROCESS_WIDTH
                    )

                    scale_y = (
                        original_height
                        / PROCESS_HEIGHT
                    )

                    exit_x1 = max(
                        0,
                        int(small_x1 * scale_x)
                    )

                    exit_y1 = max(
                        0,
                        int(small_y1 * scale_y)
                    )

                    exit_x2 = min(
                        original_width,
                        int(small_x2 * scale_x)
                    )

                    exit_y2 = min(
                        original_height,
                        int(small_y2 * scale_y)
                    )

                    exit_crop = frame[
                        exit_y1:exit_y2,
                        exit_x1:exit_x2
                    ]

                    if exit_crop.size > 0:

                        image_path = (
                            save_face_image(
                                exit_crop,
                                "EXIT"
                            )
                        )

                manager.db.add_event(
                    visitor_id,
                    "EXIT",
                    image_path
                )

                log_event(
                    f"EXIT | "
                    f"visitor_id={visitor_id} "
                    f"| track_id={track_id}"
                )

                print(
                    f"[EXIT] "
                    f"Visitor {visitor_id} "
                    f"(track {track_id})"
                )

            track_last_seen.pop(
                track_id,
                None
            )

            track_to_bbox.pop(
                track_id,
                None
            )

            recognition_last_frame.pop(
                track_id,
                None
            )

            recognition_attempts.pop(
                track_id,
                None
            )

            entry_logged_tracks.discard(
                track_id
            )

        # ---------------------------------------------------------
        # DASHBOARD INFORMATION
        # ---------------------------------------------------------

        cv2.rectangle(
            frame_small,
            (0, 0),
            (PROCESS_WIDTH, 40),
            (30, 30, 30),
            -1
        )

        if total_frames > 0:

            progress = (
                frame_number
                / total_frames
            ) * 100

        else:

            progress = 0

        info = (
            f"Frame: "
            f"{frame_number}/{total_frames} "
            f"({progress:.1f}%) | "
            f"Unique Visitors: "
            f"{len(unique_visitors)}"
        )

        cv2.putText(
            frame_small,
            info,
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Sample Video Test",
            frame_small
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print("Stopped by user.")
            break

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------

    cap.release()
    cv2.destroyAllWindows()

    # ---------------------------------------------------------
    # VIDEO-END EXITS
    # ---------------------------------------------------------

    for track_id in list(
        track_last_seen.keys()
    ):

        visitor_id = manager.remove_track(
            track_id
        )

        if visitor_id is not None:

            bbox = track_to_bbox.get(
                track_id
            )

            image_path = None

            if bbox is not None:

                (
                    small_x1,
                    small_y1,
                    small_x2,
                    small_y2
                ) = bbox

                scale_x = (
                    original_width
                    / PROCESS_WIDTH
                )

                scale_y = (
                    original_height
                    / PROCESS_HEIGHT
                )

                exit_x1 = max(
                    0,
                    int(small_x1 * scale_x)
                )

                exit_y1 = max(
                    0,
                    int(small_y1 * scale_y)
                )

                exit_x2 = min(
                    original_width,
                    int(small_x2 * scale_x)
                )

                exit_y2 = min(
                    original_height,
                    int(small_y2 * scale_y)
                )

                exit_crop = frame[
                    exit_y1:exit_y2,
                    exit_x1:exit_x2
                ]

                if exit_crop.size > 0:

                    image_path = (
                        save_face_image(
                            exit_crop,
                            "EXIT"
                        )
                    )

            manager.db.add_event(
                visitor_id,
                "EXIT",
                image_path
            )

            log_event(
                f"EXIT | "
                f"visitor_id={visitor_id} "
                f"| track_id={track_id} "
                f"| reason=VIDEO_END"
            )

            print(
                f"[VIDEO END] "
                f"Visitor {visitor_id} "
                f"(track {track_id})"
            )

    manager.close()

    # ---------------------------------------------------------
    # FINAL RESULTS
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("VIDEO TEST COMPLETE")
    print("=" * 60)

    print(
        f"Frames in video       : "
        f"{total_frames}"
    )

    print(
        f"Frames processed      : "
        f"{processed_frames}"
    )

    print(
        f"Unique visitors found : "
        f"{len(unique_visitors)}"
    )

    print(
        f"Visitors              : "
        f"{sorted(unique_visitors)}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()