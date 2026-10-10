import os
import sys
from pathlib import Path
from urllib.parse import urlsplit


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

os.chdir(PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

import time

import cv2
import json
import math
import uuid

from collections import defaultdict

from concurrent.futures import (
    ThreadPoolExecutor
)

from ultralytics import YOLO

from face_recognizer import (
    FaceRecognizer
)

from configuration import (
    validate_config,
)

from visitor_manager import (
    VisitorManager
)

from person_detection import PersonDetector
from video_input import open_capture

from logger import (
    log_event
)

from image_logger import (
    delete_event_images,
    save_face_image,
    save_unrecognized_image,
)

from dashboard_server import (
    DashboardServer,
    clear_browser_frame,
    clear_frame,
    describe_source,
    get_source_status,
    set_database_reset_status,
    set_source_status,
    take_database_reset_request,
    take_requested_source,
)


# ============================================================
# CONFIGURATION
# ============================================================

with open(
    PROJECT_ROOT / "config.json",
    "r",
    encoding="utf-8"
) as file:

    config = validate_config(json.load(file))


LIVE_SOURCE_SCHEMES = {
    "http",
    "https",
    "rtmp",
    "rtmps",
    "rtsp",
    "rtsps",
}


def normalize_capture_source(value):
    if isinstance(value, int):
        return value

    if not isinstance(value, str) or not value.strip():
        raise ValueError("Input source must be a camera index, stream URL, or video path.")

    source_text = value.strip()
    if source_text.isdecimal():
        return int(source_text)

    parsed = urlsplit(source_text)
    if parsed.scheme.lower() in LIVE_SOURCE_SCHEMES and parsed.hostname:
        return source_text

    source_path = Path(source_text)
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    return str(source_path.resolve())


def is_live_capture_source(value):
    if value == "browser":
        return True
    if isinstance(value, int):
        return True

    try:
        parsed = urlsplit(str(value))
    except ValueError:
        return False
    return parsed.scheme.lower() in LIVE_SOURCE_SCHEMES


# ============================================================
# MODEL PATHS
# ============================================================

YOLO_MODEL_PATH = (
    str(PROJECT_ROOT / "models" / "yolov8n-face.pt")
)

YUNET_MODEL_PATH = (
    str(PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx")
)


# ============================================================
# DETECTION SETTINGS
# ============================================================

DETECTION_CONFIG = config["detection"]
YOLO_CONFIDENCE = float(
    DETECTION_CONFIG.get("confidence", 0.25)
)
if not 0.0 < YOLO_CONFIDENCE < 1.0:
    raise ValueError("detection.confidence must be between 0 and 1.")

SKIP_FRAMES = max(0, int(DETECTION_CONFIG.get("skip_frames", 1)))
YOLO_IMAGE_SIZE = max(320, int(DETECTION_CONFIG.get("image_size", 800)))
MIN_FACE_SIZE = max(1, int(DETECTION_CONFIG.get("min_face_size", 20)))
VALIDATE_WITH_YUNET = bool(
    DETECTION_CONFIG.get("validate_with_yunet", False)
)
PERSON_DETECTION_CONFIG = config.get("person_detection", {})
PERSON_DETECTION_ENABLED = bool(
    PERSON_DETECTION_CONFIG.get("enabled", True)
)
PERSON_DETECTION_MAX_DIMENSION = max(
    320,
    int(PERSON_DETECTION_CONFIG.get("max_dimension", 640)),
)
PERSON_DETECTION_SCORE_THRESHOLD = float(
    PERSON_DETECTION_CONFIG.get("score_threshold", 1.1)
)


# ============================================================
# RECOGNITION SETTINGS
# ============================================================

SIMILARITY_THRESHOLD = (
    config[
        "recognition"
    ][
        "similarity_threshold"
    ]
)


# ============================================================
# TRACKING SETTINGS
# ============================================================

MAX_MISSING_FRAMES = (
    config[
        "tracking"
    ][
        "max_missing_frames"
    ]
)


# ============================================================
# OUTPUT SETTINGS
# ============================================================

SHOW_WINDOW = (
    config[
        "output"
    ][
        "show_window"
    ]
)


SAVE_IMAGES = (
    config[
        "output"
    ][
        "save_images"
    ]
)


# ============================================================
# FACE FILTER SETTINGS
# ============================================================

# ============================================================
# IMPORTANT RECOGNITION SETTINGS
#
# 1 stable detection is enough to start recognition.
#
# This makes Person 2 start recognition immediately instead
# of waiting for several detection cycles.
# ============================================================

STABLE_TRACK_COUNT = 1


# ============================================================
# Recognition is retried with a per-track backoff if an embedding
# cannot be extracted; a low-quality crop must not permanently block a visitor.
# ============================================================

# ============================================================
# DISPLAY SETTINGS
# ============================================================

MAX_DISPLAY_MISSING_CYCLES = 5


# ============================================================
# YUNET SETTINGS
# ============================================================

YUNET_SCORE_THRESHOLD = float(
    DETECTION_CONFIG.get("yunet_confidence", 0.7)
)

FACE_VALIDATION_IOU = 0.20

DUPLICATE_IOU_THRESHOLD = 0.30


# ============================================================
# IOU
# ============================================================

def calculate_iou(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = (
        box_a
    )

    bx1, by1, bx2, by2 = (
        box_b
    )


    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )


    intersection_width = max(
        0,
        ix2 - ix1
    )

    intersection_height = max(
        0,
        iy2 - iy1
    )


    intersection = (
        intersection_width
        *
        intersection_height
    )


    area_a = (
        max(
            0,
            ax2 - ax1
        )
        *
        max(
            0,
            ay2 - ay1
        )
    )


    area_b = (
        max(
            0,
            bx2 - bx1
        )
        *
        max(
            0,
            by2 - by1
        )
    )


    union = (
        area_a
        +
        area_b
        -
        intersection
    )


    if union <= 0:

        return 0.0


    return (
        intersection
        /
        union
    )


def detect_yunet_boxes(frame, detector):
    detector.setInputSize((frame.shape[1], frame.shape[0]))
    _, faces = detector.detect(frame)
    if faces is None:
        return []

    boxes = []
    frame_height, frame_width = frame.shape[:2]
    for face in faces:
        x, y, width, height = map(int, face[:4])
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_width, x + width)
        y2 = min(frame_height, y + height)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


# ============================================================
# CENTER DISTANCE
# ============================================================

def center_distance(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = (
        box_a
    )

    bx1, by1, bx2, by2 = (
        box_b
    )


    acx = (
        ax1 + ax2
    ) / 2

    acy = (
        ay1 + ay2
    ) / 2


    bcx = (
        bx1 + bx2
    ) / 2

    bcy = (
        by1 + by2
    ) / 2


    return math.sqrt(
        (
            acx - bcx
        ) ** 2
        +
        (
            acy - bcy
        ) ** 2
    )


# ============================================================
# DUPLICATE BOX CHECK
# ============================================================

def boxes_are_duplicate(
    box_a,
    box_b
):

    iou = calculate_iou(
        box_a,
        box_b
    )


    if (
        iou
        >= DUPLICATE_IOU_THRESHOLD
    ):

        return True


    ax1, ay1, ax2, ay2 = (
        box_a
    )

    bx1, by1, bx2, by2 = (
        box_b
    )


    aw = (
        ax2 - ax1
    )

    ah = (
        ay2 - ay1
    )

    bw = (
        bx2 - bx1
    )

    bh = (
        by2 - by1
    )


    minimum_size = min(
        aw,
        ah,
        bw,
        bh
    )


    if minimum_size <= 0:

        return False


    distance = center_distance(
        box_a,
        box_b
    )


    return (
        distance
        <
        minimum_size * 0.45
    )


# ============================================================
# LOAD YOLO
# ============================================================

print(
    "Loading YOLOv8-Face..."
)


yolo = YOLO(
    YOLO_MODEL_PATH
)

if PERSON_DETECTION_ENABLED:
    person_detector = PersonDetector(
        max_dimension=PERSON_DETECTION_MAX_DIMENSION,
        score_threshold=PERSON_DETECTION_SCORE_THRESHOLD,
        min_hits=int(PERSON_DETECTION_CONFIG.get("min_hits", 2)),
        max_missing=int(PERSON_DETECTION_CONFIG.get("max_missing", 1)),
    )
else:
    person_detector = None


# ============================================================
# LOAD YUNET
# ============================================================

if VALIDATE_WITH_YUNET:
    print("Loading YuNet face validator...")
    yunet = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH,
        "",
        (320, 320),
        YUNET_SCORE_THRESHOLD,
        0.30,
        5000,
    )
else:
    yunet = None


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

print(
    "Loading InsightFace..."
)


recognizer = (
    FaceRecognizer()
)


# ============================================================
# LOAD VISITOR MANAGER
# ============================================================

print(
    "Loading Visitor Manager..."
)


visitor_manager = (
    VisitorManager(
        similarity_threshold=(
            SIMILARITY_THRESHOLD
        )
    )
)


print(
    "All AI modules loaded."
)


# ============================================================
# OPEN VIDEO SOURCE
# ============================================================

source = (
    config[
        "input"
    ][
        "source"
    ]
)

try:
    source = normalize_capture_source(source)
    cap = (
        open_capture(source)
        if source != "browser"
        else cv2.VideoCapture()
    )
except (ValueError, cv2.error) as error:
    cap = cv2.VideoCapture()
    message = (
        f"Could not initialize {describe_source(source)}."
        if is_live_capture_source(source)
        else f"Could not initialize video source: {error}"
    )
    print(f"ERROR: {message}")
    set_source_status("error", source, message)
else:
    if source == "browser":
        set_source_status(
            "waiting",
            source,
            "Click Start device camera in the dashboard to begin.",
        )
        print("Waiting for the dashboard device camera.")
    elif cap.isOpened():
        set_source_status("running", source, "Video source opened successfully.")
        print("Video source opened successfully.")
    else:
        message = f"Could not open video source: {describe_source(source)}"
        print(f"ERROR: {message}")
        set_source_status("error", source, message)

source_is_live = is_live_capture_source(source)

print(
    "Press Q to quit. Use the dashboard to upload a video or connect a stream."
)


# ============================================================
# TRACK STATE
# ============================================================

frame_count = 0


last_boxes = {}


last_face_crops = {}


track_seen_count = (
    defaultdict(int)
)


missing_frames = (
    defaultdict(int)
)


recognition_attempts = (
    defaultdict(int)
)

recognition_retry_after = (
    defaultdict(float)
)


recognition_jobs = {}


failed_tracks = set()


active_tracks = set()


expired_tracks = set()


unique_visitors = set()


# ============================================================
# ENTRY TRACK STATE
#
# This prevents:
#
# Person 1
# Person 1
# Person 1
# Person 1
#
# from generating multiple ENTRY events.
#
# But after Person 1 exits, the track is removed.
# When Person 1 comes back with a new ByteTrack ID,
# a new ENTRY will be generated.
# ============================================================

entry_logged_tracks = set()
read_failure_count = 0
browser_frame_version = 0
browser_last_frame_at = 0.0
last_person_boxes = []
sighting_track_keys = {}


def record_unrecognized_sighting(kind, track_id, label, image_crop):
    tracker_key = (kind, track_id)
    if tracker_key in sighting_track_keys:
        return

    persistent_key = uuid.uuid4().hex
    image_path = None
    if image_crop is not None and image_crop.size:
        try:
            image_path = save_unrecognized_image(image_crop)
        except OSError as error:
            log_event(
                f"UNRECOGNIZED_IMAGE_SAVE_ERROR | "
                f"kind={kind} | track_id={track_id} | error={error}"
            )

    visitor_manager.db.add_unrecognized_sighting(
        persistent_key,
        kind,
        label,
        image_path,
    )
    sighting_track_keys[tracker_key] = persistent_key
    log_event(
        f"UNRECOGNIZED_SIGHTING | kind={kind} | "
        f"track_id={track_id} | sighting_id={persistent_key}"
    )


def finish_active_tracks():
    for track_id, future in list(recognition_jobs.items()):
        try:
            embedding = future.result(timeout=10)
        except Exception as error:
            log_event(
                f"RECOGNITION_ERROR | "
                f"track_id={track_id} | "
                f"error={error}"
            )
            embedding = None

        if embedding is None or track_id in expired_tracks:
            continue

        visitor_id, is_new = visitor_manager.identify(track_id, embedding)
        if visitor_id is None:
            continue

        unique_visitors.add(visitor_id)
        if track_id not in entry_logged_tracks:
            image_path = None
            face_crop = last_face_crops.get(track_id)
            if SAVE_IMAGES and face_crop is not None:
                image_path = save_face_image(face_crop, "ENTRY")

            visitor_manager.db.add_event(visitor_id, "ENTRY", image_path)
            log_event(
                f"ENTRY | visitor_id={visitor_id} | track_id={track_id}"
            )
            entry_logged_tracks.add(track_id)

        event_name = "VISITOR_REGISTERED" if is_new else "VISITOR_RECOGNIZED"
        log_event(
            f"{event_name} | visitor_id={visitor_id} | track_id={track_id}"
        )

    recognition_jobs.clear()

    for track_id in list(active_tracks):
        visitor_id = visitor_manager.remove_track(track_id)
        if visitor_id is not None:
            image_path = None
            face_crop = last_face_crops.get(track_id)
            if SAVE_IMAGES and face_crop is not None:
                image_path = save_face_image(face_crop, "EXIT")

            visitor_manager.db.add_event(visitor_id, "EXIT", image_path)
            log_event(
                f"EXIT | visitor_id={visitor_id} | track_id={track_id}"
            )

    active_tracks.clear()
    visitor_manager.pending_tracks.clear()


def reset_tracker_data():
    global last_person_boxes
    image_paths = visitor_manager.db.get_event_image_paths()
    image_paths.extend(visitor_manager.db.get_unrecognized_image_paths())
    delete_event_images(image_paths, PROJECT_ROOT)

    visitor_manager.db.reset()
    for future in recognition_jobs.values():
        future.cancel()
    recognition_jobs.clear()
    last_boxes.clear()
    last_face_crops.clear()
    track_seen_count.clear()
    missing_frames.clear()
    recognition_attempts.clear()
    recognition_retry_after.clear()
    last_person_boxes = []
    sighting_track_keys.clear()
    if person_detector is not None:
        person_detector.reset()
    failed_tracks.clear()
    active_tracks.clear()
    expired_tracks.clear()
    entry_logged_tracks.clear()
    unique_visitors.clear()
    visitor_manager.track_to_visitor.clear()
    visitor_manager.pending_tracks.clear()
    visitor_manager.clear_template_cache()
    clear_frame()


# ============================================================
# BACKGROUND RECOGNITION
#
# Two workers allow multiple faces to enter the recognition
# queue without forcing Person 2 to wait unnecessarily behind
# Person 1.
# ============================================================

executor = (
    ThreadPoolExecutor(
        max_workers=2
    )
)


# ============================================================
# DASHBOARD
# ============================================================

dashboard_server = (
    DashboardServer()
)


dashboard_server.start()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        if take_database_reset_request():
            try:
                reset_tracker_data()
            except Exception as error:
                log_event(f"DATABASE_RESET_ERROR | error={error}")
                set_database_reset_status(
                    "error",
                    "Database reset failed. Check logs/events.log for details.",
                )
            else:
                log_event(
                    "DATABASE_RESET | visitors=0 | events=0 | "
                    "saved_face_evidence=cleared"
                )
                set_database_reset_status(
                    "complete",
                    "Reset complete. Visitor IDs, event counts, and saved face evidence were cleared.",
                )

        requested_source = take_requested_source()
        if requested_source is not None:
            try:
                next_source = normalize_capture_source(requested_source)
                next_cap = (
                    None
                    if next_source == "browser"
                    else open_capture(next_source)
                )
            except (ValueError, cv2.error) as error:
                message = (
                    f"Could not initialize {describe_source(requested_source)}."
                    if is_live_capture_source(requested_source)
                    else f"Could not initialize requested source: {error}"
                )
                print(f"ERROR: {message}")
                set_source_status("error", source, message)
            else:
                if next_cap is not None and not next_cap.isOpened():
                    next_cap.release()
                    if (
                        isinstance(next_source, str)
                        and urlsplit(next_source).scheme.lower()
                        in {"rtsp", "rtsps"}
                    ):
                        message = (
                            f"Could not open RTSP stream "
                            f"({describe_source(next_source)}). Tried TCP and "
                            "UDP transport; check that the camera is reachable, "
                            "the RTSP port/path is correct, and credentials are "
                            "valid. The current source remains selected."
                        )
                    else:
                        message = (
                            f"Could not open {describe_source(next_source)}; "
                            "the current source remains selected."
                        )
                    print(f"ERROR: {message}")
                    set_source_status("error", source, message)
                else:
                    finish_active_tracks()
                    cap.release()
                    cap = next_cap if next_cap is not None else cv2.VideoCapture()
                    source = next_source
                    sighting_track_keys.clear()
                    if person_detector is not None:
                        person_detector.reset()
                    last_person_boxes = []
                    source_is_live = is_live_capture_source(source)
                    read_failure_count = 0
                    browser_frame_version = 0
                    frame_count = 0
                    browser_last_frame_at = (
                        time.monotonic()
                        if source == "browser"
                        else 0.0
                    )
                    if source != "browser":
                        clear_browser_frame()

                    for future in recognition_jobs.values():
                        future.cancel()
                    recognition_jobs.clear()
                    last_boxes.clear()
                    last_face_crops.clear()
                    track_seen_count.clear()
                    missing_frames.clear()
                    recognition_attempts.clear()
                    failed_tracks.clear()
                    active_tracks.clear()
                    expired_tracks.clear()
                    entry_logged_tracks.clear()
                    visitor_manager.track_to_visitor.clear()
                    visitor_manager.pending_tracks.clear()

                    for tracker in getattr(
                        getattr(yolo, "predictor", None),
                        "trackers",
                        (),
                    ):
                        reset_tracker = getattr(tracker, "reset", None)
                        if reset_tracker is not None:
                            reset_tracker()

                    set_source_status(
                        "running",
                        source,
                        (
                            "Waiting for browser camera frames."
                            if source == "browser"
                            else "Video source opened successfully."
                        ),
                    )
                    print(f"Switched video source to {describe_source(source)}.")

        if source == "browser":
            if get_source_status()["state"] == "paused":
                time.sleep(0.02)
                continue
            frame, new_version = take_browser_frame(browser_frame_version)
            if frame is None:
                if (
                    browser_last_frame_at
                    and time.monotonic() - browser_last_frame_at > 1.0
                    and get_source_status()["state"] == "running"
                ):
                    clear_frame()
                    set_source_status(
                        "waiting",
                        source,
                        "Waiting for browser camera frames.",
                    )
                time.sleep(0.01)
                continue
            browser_frame_version = new_version
            browser_last_frame_at = time.monotonic()
            if get_source_status()["state"] != "running":
                set_source_status(
                    "running",
                    source,
                    "Receiving browser camera frames.",
                )
            ret = True
        else:
            frame = None
            ret = False

        if source != "browser" and not cap.isOpened():
            if is_live_capture_source(source):
                set_source_status(
                    "reconnecting",
                    source,
                    "Live source is unavailable; retrying connection.",
                )
                time.sleep(1.0)
                try:
                    cap = open_capture(source)
                except cv2.error:
                    message = f"Could not reconnect to {describe_source(source)}."
                    print(f"ERROR: {message}")
                    set_source_status("error", source, message)
                    cap = cv2.VideoCapture()
                else:
                    if cap.isOpened():
                        read_failure_count = 0
                        set_source_status(
                            "running",
                            source,
                            "Live source reconnected.",
                        )
                    else:
                        message = "Live source is still unavailable; retrying."
                        set_source_status("reconnecting", source, message)
            else:
                time.sleep(0.1)
            continue

        # ====================================================
        # READ FRAME
        # ====================================================

        if source != "browser":
            ret, frame = cap.read()


        if not ret:
            read_failure_count += 1
            if not source_is_live:
                cap.release()
                finish_active_tracks()
                clear_frame()
                message = "Video playback reached its end or the file could not be decoded."
                print(message)
                set_source_status("completed", source, message)
            elif read_failure_count >= 3:
                cap.release()
                clear_frame()
                set_source_status(
                    "reconnecting",
                    source,
                    "Live stream stopped delivering frames; retrying connection.",
                )
            else:
                time.sleep(0.1)
            continue

        read_failure_count = 0


        frame_count += 1


        # ====================================================
        # CURRENT TRACKS
        # ====================================================

        current_tracks = set()


        # ====================================================
        # DETECTION CYCLE
        # ====================================================

        if (
            frame_count
            %
            (
                (0 if source == "browser" else SKIP_FRAMES) + 1
            )
            == 0
        ):

            last_person_boxes = (
                person_detector.detect(frame)
                if person_detector is not None
                else []
            )

            valid_yunet_boxes = (
                detect_yunet_boxes(frame, yunet)
                if yunet is not None
                else []
            )


            # =================================================
            # YOLO + BYTETRACK
            # =================================================

            results = yolo.track(

                frame,

                persist=True,

                tracker=(
                    "bytetrack.yaml"
                ),

                conf=(
                    YOLO_CONFIDENCE
                ),

                iou=0.50,

                imgsz=YOLO_IMAGE_SIZE,

                verbose=False
            )


            result = (
                results[0]
            )


            candidates = []


            # =================================================
            # YOLO BOXES
            # =================================================

            if (
                result.boxes
                is not None
                and
                result.boxes.id
                is not None
            ):

                boxes = (
                    result.boxes
                )


                for i in range(
                    len(boxes)
                ):


                    # =========================================
                    # TRACK ID
                    # =========================================

                    track_id = int(
                        boxes.id[i].item()
                    )


                    # =========================================
                    # CONFIDENCE
                    # =========================================

                    if (
                        boxes.conf
                        is not None
                    ):

                        confidence = float(
                            boxes.conf[
                                i
                            ].item()
                        )

                    else:

                        confidence = 0.0


                    if (
                        confidence
                        <
                        YOLO_CONFIDENCE
                    ):

                        continue


                    # =========================================
                    # BOUNDING BOX
                    # =========================================

                    (
                        x1,
                        y1,
                        x2,
                        y2
                    ) = (
                        boxes.xyxy[
                            i
                        ]
                        .cpu()
                        .numpy()
                    )


                    x1 = max(
                        0,
                        int(x1)
                    )

                    y1 = max(
                        0,
                        int(y1)
                    )

                    x2 = min(
                        frame.shape[1],
                        int(x2)
                    )

                    y2 = min(
                        frame.shape[0],
                        int(y2)
                    )


                    if (
                        x2 <= x1
                        or
                        y2 <= y1
                    ):

                        continue


                    width = (
                        x2 - x1
                    )

                    height = (
                        y2 - y1
                    )


                    # =========================================
                    # FACE SIZE FILTER
                    # =========================================

                    if min(width, height) < MIN_FACE_SIZE:

                        continue


                    yolo_box = (
                        x1,
                        y1,
                        x2,
                        y2
                    )


                    # =========================================
                    # YUNET VALIDATION
                    # =========================================

                    face_confirmed = not VALIDATE_WITH_YUNET


                    for yunet_box in (
                        valid_yunet_boxes
                    ):


                        # -------------------------------------
                        # IOU MATCH
                        # -------------------------------------

                        overlap = (
                            calculate_iou(
                                yolo_box,
                                yunet_box
                            )
                        )


                        if (
                            overlap
                            >=
                            FACE_VALIDATION_IOU
                        ):

                            face_confirmed = (
                                True
                            )

                            break


                        # -------------------------------------
                        # CENTER MATCH
                        # -------------------------------------

                        cx = (
                            x1 + x2
                        ) / 2

                        cy = (
                            y1 + y2
                        ) / 2


                        (
                            ux1,
                            uy1,
                            ux2,
                            uy2
                        ) = yunet_box


                        padding_x = (
                            ux2 - ux1
                        ) * 0.20


                        padding_y = (
                            uy2 - uy1
                        ) * 0.20


                        if (
                            ux1
                            - padding_x
                            <= cx
                            <= ux2
                            + padding_x
                            and
                            uy1
                            - padding_y
                            <= cy
                            <= uy2
                            + padding_y
                        ):

                            face_confirmed = (
                                True
                            )

                            break


                    if not face_confirmed:

                        continue


                    # =========================================
                    # CANDIDATE
                    # =========================================

                    candidates.append(
                        {
                            "track_id":
                                track_id,

                            "bbox":
                                yolo_box,

                            "confidence":
                                confidence,

                            "known":
                                (
                                    track_id
                                    in
                                    visitor_manager
                                    .track_to_visitor
                                )
                        }
                    )


            # =================================================
            # SORT CANDIDATES
            # =================================================

            candidates.sort(

                key=lambda item: (

                    item["known"],

                    item["confidence"]

                ),

                reverse=True
            )


            # =================================================
            # REMOVE DUPLICATES
            # =================================================

            filtered_candidates = []


            for candidate in (
                candidates
            ):

                duplicate = False


                for kept in (
                    filtered_candidates
                ):

                    if boxes_are_duplicate(

                        candidate[
                            "bbox"
                        ],

                        kept[
                            "bbox"
                        ]

                    ):

                        duplicate = True

                        break


                if not duplicate:

                    filtered_candidates.append(
                        candidate
                    )


            # =================================================
            # PROCESS TRACKS
            # =================================================

            for candidate in (
                filtered_candidates
            ):

                track_id = (
                    candidate[
                        "track_id"
                    ]
                )


                (
                    x1,
                    y1,
                    x2,
                    y2
                ) = (
                    candidate[
                        "bbox"
                    ]
                )


                current_tracks.add(
                    track_id
                )


                track_seen_count[
                    track_id
                ] += 1


                missing_frames[
                    track_id
                ] = 0


                last_boxes[
                    track_id
                ] = candidate[
                    "bbox"
                ]


                # =============================================
                # SAVE FACE CROP
                # =============================================

                face_crop = frame[
                    y1:y2,
                    x1:x2
                ]


                if (
                    face_crop.size
                    > 0
                ):

                    last_face_crops[
                        track_id
                    ] = (
                        face_crop.copy()
                    )


                # =============================================
                # RECOGNITION STATE
                # =============================================

                already_identified = (
                    track_id
                    in
                    visitor_manager
                    .track_to_visitor
                )


                recognition_running = (
                    track_id
                    in
                    recognition_jobs
                )


                # =============================================
                # START RECOGNITION
                # =============================================

                if (

                    not already_identified

                    and
                    not recognition_running

                    and

                    track_seen_count[
                        track_id
                    ]
                    >=
                    STABLE_TRACK_COUNT

                    and

                    time.monotonic()
                    >=
                    recognition_retry_after[
                        track_id
                    ]

                ):


                    x1, y1, x2, y2 = candidate["bbox"]
                    padding_x = int((x2 - x1) * 0.35)
                    padding_y = int((y2 - y1) * 0.35)
                    crop_x1 = max(0, x1 - padding_x)
                    crop_y1 = max(0, y1 - padding_y)
                    crop_x2 = min(frame.shape[1], x2 + padding_x)
                    crop_y2 = min(frame.shape[0], y2 + padding_y)
                    recognition_crop = frame[
                        crop_y1:crop_y2,
                        crop_x1:crop_x2,
                    ].copy()
                    recognition_bbox = (
                        x1 - crop_x1,
                        y1 - crop_y1,
                        x2 - crop_x1,
                        y2 - crop_y1,
                    )


                    recognition_jobs[
                        track_id
                    ] = (

                        executor.submit(

                            recognizer.get_embedding,

                            recognition_crop,

                            recognition_bbox

                        )

                    )


                    recognition_attempts[
                        track_id
                    ] += 1


                    log_event(

                        f"RECOGNITION_STARTED | "
                        f"track_id={track_id} | "
                        f"attempt="
                        f"{recognition_attempts[track_id]}"

                    )


            # =================================================
            # CHECK RECOGNITION JOBS
            # =================================================

            for (
                track_id,
                future
            ) in list(
                recognition_jobs.items()
            ):


                if not future.done():

                    continue


                del recognition_jobs[
                    track_id
                ]


                # =============================================
                # EXPIRED TRACK
                # =============================================

                if (
                    track_id
                    in
                    expired_tracks
                ):

                    continue


                if (
                    missing_frames.get(
                        track_id,
                        0
                    )
                    >= MAX_MISSING_FRAMES
                ):

                    continue


                # =============================================
                # GET EMBEDDING
                # =============================================

                try:

                    embedding = (
                        future.result()
                    )


                except Exception as error:

                    log_event(

                        f"RECOGNITION_ERROR | "
                        f"track_id={track_id} | "
                        f"error={error}"

                    )

                    embedding = None


                # =============================================
                # FAILED EMBEDDING
                # =============================================

                if embedding is None:

                    log_event(

                        f"EMBEDDING_FAILED | "
                        f"track_id={track_id} | "
                        f"attempt="
                        f"{recognition_attempts[track_id]}"

                    )


                    failed_tracks.add(track_id)
                    recognition_retry_after[track_id] = (
                        time.monotonic()
                        + min(
                            2.0,
                            0.25 * recognition_attempts[track_id],
                        )
                    )


                    continue


                # =============================================
                # EMBEDDING SUCCESS
                # =============================================

                log_event(

                    f"EMBEDDING_GENERATED | "
                    f"track_id={track_id}"

                )
                failed_tracks.discard(track_id)
                recognition_retry_after[track_id] = time.monotonic() + 0.25


                # =============================================
                # IDENTIFY VISITOR
                # =============================================

                (
                    visitor_id,
                    is_new
                ) = (
                    visitor_manager.identify(

                        track_id,

                        embedding

                    )
                )


                # =============================================
                # NO IDENTIFICATION YET
                #
                # This is normal for a brand-new face because
                # VisitorManager waits for consistent
                # observations before registering it.
                # =============================================

                if (
                    visitor_id
                    is None
                ):

                    continue


                # =============================================
                # UNIQUE VISITOR
                #
                # A returning person does NOT increase this.
                # =============================================

                unique_visitors.add(
                    visitor_id
                )


                # =============================================
                # ENTRY
                #
                # IMPORTANT:
                #
                # We do NOT check is_new here.
                #
                # Therefore:
                #
                # NEW PERSON
                #     -> ENTRY
                #
                # KNOWN PERSON RETURNING
                #     -> ENTRY
                #
                # SAME TRACK
                #     -> no duplicate ENTRY
                # =============================================

                if (
                    track_id
                    not in
                    entry_logged_tracks
                ):


                    image_path = None


                    if SAVE_IMAGES:

                        face_crop = (
                            last_face_crops.get(
                                track_id
                            )
                        )


                        if (
                            face_crop
                            is not None
                        ):

                            image_path = (
                                save_face_image(

                                    face_crop,

                                    "ENTRY"

                                )
                            )


                    visitor_manager.db.add_event(

                        visitor_id,

                        "ENTRY",

                        image_path

                    )


                    log_event(

                        f"ENTRY | "
                        f"visitor_id={visitor_id} | "
                        f"track_id={track_id}"

                    )


                    entry_logged_tracks.add(
                        track_id
                    )


                # =============================================
                # NEW VISITOR LOG
                # =============================================

                if is_new:

                    log_event(

                        f"VISITOR_REGISTERED | "
                        f"visitor_id={visitor_id} | "
                        f"track_id={track_id}"

                    )


                # =============================================
                # EXISTING VISITOR LOG
                # =============================================

                else:

                    log_event(

                        f"VISITOR_RECOGNIZED | "
                        f"visitor_id={visitor_id} | "
                        f"track_id={track_id}"

                    )


            # =================================================
            # HANDLE MISSING TRACKS
            # =================================================

            for old_track in list(
                active_tracks
            ):


                if (
                    old_track
                    in
                    current_tracks
                ):

                    continue


                missing_frames[
                    old_track
                ] += 1


                # =============================================
                # WAIT BEFORE EXIT
                # =============================================

                if (
                    missing_frames[
                        old_track
                    ]
                    <
                    MAX_MISSING_FRAMES
                ):

                    continue


                # =============================================
                # REMOVE TRACK
                # =============================================

                visitor_id = (
                    visitor_manager.remove_track(
                        old_track
                    )
                )


                # =============================================
                # EXIT EVENT
                # =============================================

                if (
                    visitor_id
                    is not None
                ):


                    image_path = None


                    if SAVE_IMAGES:

                        face_crop = (
                            last_face_crops.get(
                                old_track
                            )
                        )


                        if (
                            face_crop
                            is not None
                        ):

                            image_path = (
                                save_face_image(

                                    face_crop,

                                    "EXIT"

                                )
                            )


                    visitor_manager.db.add_event(

                        visitor_id,

                        "EXIT",

                        image_path

                    )


                    log_event(

                        f"EXIT | "
                        f"visitor_id={visitor_id} | "
                        f"track_id={old_track}"

                    )


                # =============================================
                # EXPIRE TRACK
                # =============================================

                expired_tracks.add(
                    old_track
                )


                # =============================================
                # VERY IMPORTANT
                #
                # Remove the ENTRY lock.
                #
                # When this person comes back with another
                # ByteTrack ID, a new ENTRY is allowed.
                # =============================================

                entry_logged_tracks.discard(
                    old_track
                )


                # =============================================
                # CLEAN TRACK STATE
                # =============================================

                missing_frames.pop(
                    old_track,
                    None
                )


                track_seen_count.pop(
                    old_track,
                    None
                )


                recognition_attempts.pop(
                    old_track,
                    None
                )


                last_boxes.pop(
                    old_track,
                    None
                )


                last_face_crops.pop(
                    old_track,
                    None
                )


                recognition_jobs.pop(
                    old_track,
                    None
                )


                failed_tracks.discard(
                    old_track
                )
                sighting_track_keys.pop(
                    ("face", old_track),
                    None,
                )


            # =================================================
            # KEEP TEMPORARILY MISSING TRACKS
            # =================================================

            active_tracks = (

                current_tracks

                |

                {
                    track_id

                    for track_id
                    in active_tracks

                    if (

                        track_id
                        not in
                        expired_tracks

                        and

                        missing_frames.get(
                            track_id,
                            0
                        )
                        <
                        MAX_MISSING_FRAMES

                    )
                }

            )


        # ====================================================
        # DRAW TRACKS
        # ====================================================

        for (
            track_id,
            bbox
        ) in list(
            last_boxes.items()
        ):


            # ================================================
            # EXPIRED
            # ================================================

            if (
                track_id
                in
                expired_tracks
            ):

                continue


            # ================================================
            # STALE
            # ================================================

            if (
                missing_frames.get(
                    track_id,
                    0
                )
                >
                MAX_DISPLAY_MISSING_CYCLES
            ):

                continue


            # ================================================
            # IDENTIFIED
            # ================================================

            if (
                track_id
                in
                visitor_manager
                .track_to_visitor
            ):

                visitor_id = (
                    visitor_manager
                    .track_to_visitor[
                        track_id
                    ]
                )


                label = (
                    f"Person {visitor_id}"
                )
                box_color = (0, 255, 0)


            # ================================================
            # RECOGNIZING
            # ================================================

            elif (
                track_id
                in
                recognition_jobs
            ):

                label = (
                    "Recognizing..."
                )
                box_color = (0, 200, 255)


            elif track_id in visitor_manager.pending_tracks:
                label = "Confirming new face..."
                box_color = (0, 200, 255)

            else:
                if track_id in failed_tracks:
                    label = "Unrecognized face"
                    box_color = (0, 0, 255)
                else:
                    label = "Face"
                    box_color = (0, 200, 255)

            if track_id not in visitor_manager.track_to_visitor:
                face_crop = last_face_crops.get(track_id)
                record_unrecognized_sighting(
                    "face",
                    track_id,
                    "Face detected - not recognized",
                    face_crop,
                )


            x1, y1, x2, y2 = (
                bbox
            )


            # ================================================
            # BOX
            # ================================================

            cv2.rectangle(

                frame,

                (
                    x1,
                    y1
                ),

                (
                    x2,
                    y2
                ),

                box_color,

                2

            )


            # ================================================
            # LABEL
            # ================================================

            cv2.putText(

                frame,

                label,

                (
                    x1,
                    max(
                        y1 - 10,
                        20
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                box_color,

                2

            )

        active_person_track_ids = {
            person_track_id
            for person_track_id, _ in last_person_boxes
        }
        for key in list(sighting_track_keys):
            if key[0] == "person" and key[1] not in active_person_track_ids:
                sighting_track_keys.pop(key, None)

        for person_track_id, (
            person_x,
            person_y,
            person_width,
            person_height,
        ) in last_person_boxes:
            person_x1 = max(0, person_x)
            person_y1 = max(0, person_y)
            person_x2 = min(frame.shape[1], person_x + person_width)
            person_y2 = min(frame.shape[0], person_y + person_height)
            if person_x2 <= person_x1 or person_y2 <= person_y1:
                continue

            face_visible = any(
                person_x1 <= (x1 + x2) / 2 <= person_x2
                and person_y1 <= (y1 + y2) / 2
                <= person_y1 + (person_y2 - person_y1) * 0.45
                for face_track_id, (x1, y1, x2, y2) in last_boxes.items()
                if missing_frames.get(face_track_id, 0)
                <= MAX_DISPLAY_MISSING_CYCLES
            )
            if face_visible:
                continue

            person_color = (255, 180, 0)
            record_unrecognized_sighting(
                "person",
                person_track_id,
                "Person detected - no face visible",
                frame[person_y1:person_y2, person_x1:person_x2],
            )
            cv2.rectangle(
                frame,
                (person_x1, person_y1),
                (person_x2, person_y2),
                person_color,
                2,
            )
            cv2.putText(
                frame,
                "Person detected - no face visible",
                (person_x1, max(person_y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                person_color,
                2,
            )


        # ====================================================
        # UNIQUE VISITOR COUNT
        # ====================================================

        cv2.putText(

            frame,

            (
                "Unique Visitors: "
                f"{len(unique_visitors)}"
            ),

            (
                20,
                35
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (
                0,
                255,
                255
            ),

            2

        )


        # ====================================================
        # FRAME NUMBER
        # ====================================================

        cv2.putText(

            frame,

            (
                f"Frame: "
                f"{frame_count}"
            ),

            (
                20,
                65
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (
                255,
                255,
                255
            ),

            1

        )


        # ====================================================
        # SEND FRAME TO DASHBOARD
        # ====================================================

        dashboard_server.update_frame(
            frame
        )


        # ====================================================
        # OPENCV WINDOW
        # ====================================================

        if SHOW_WINDOW:

            cv2.imshow(

                "Intelligent Face Tracker",

                frame

            )


            if (
                cv2.waitKey(1)
                &
                0xFF
            ) == ord("q"):

                break


# ============================================================
# CLEANUP
# ============================================================

finally:

    cap.release()


    cv2.destroyAllWindows()


    executor.shutdown(

        wait=False,

        cancel_futures=True

    )


    dashboard_server.stop()


    visitor_manager.close()


    print()

    print(
        "==================================="
    )

    print(
        "Intelligent Face Tracker stopped."
    )

    print(
        f"Unique Visitors: "
        f"{len(unique_visitors)}"
    )

    print(
        "==================================="
    )