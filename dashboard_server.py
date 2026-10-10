import os
import queue
import sqlite3
import threading
import time
from pathlib import Path
from datetime import datetime
from urllib.parse import urlsplit
from uuid import uuid4

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    jsonify,
    request,
    send_from_directory,
    send_file,
)
from werkzeug.exceptions import RequestEntityTooLarge


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DASHBOARD_DIR = os.path.join(
    BASE_DIR,
    "dashboard"
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "database",
    "visitors.db"
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "videos",
    "uploads"
)

MAX_VIDEO_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_STREAM_WIDTH = 1280
JPEG_QUALITY = 75
DASHBOARD_HOST = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "5000"))
DASHBOARD_TLS_CERT = os.environ.get("DASHBOARD_TLS_CERT")
DASHBOARD_TLS_KEY = os.environ.get("DASHBOARD_TLS_KEY")
ALLOWED_VIDEO_EXTENSIONS = {
    ".avi",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}
STREAM_SCHEMES = {
    "http",
    "https",
    "rtmp",
    "rtmps",
    "rtsp",
    "rtsps",
}
MANAGED_FACE_IMAGE_DIRS = (
    (Path(BASE_DIR) / "logs" / "entries").resolve(),
    (Path(BASE_DIR) / "logs" / "exits").resolve(),
    (Path(BASE_DIR) / "logs" / "unrecognized").resolve(),
)


def resolve_face_image(image_path):
    if not image_path:
        return None

    path = Path(image_path)
    if not path.is_absolute():
        path = Path(BASE_DIR) / path
    resolved = path.resolve()
    if not any(
        resolved.is_relative_to(directory)
        for directory in MANAGED_FACE_IMAGE_DIRS
    ):
        return None
    return resolved if resolved.is_file() else None


def get_dashboard_ssl_context():
    if bool(DASHBOARD_TLS_CERT) != bool(DASHBOARD_TLS_KEY):
        raise ValueError(
            "Set both DASHBOARD_TLS_CERT and DASHBOARD_TLS_KEY to enable HTTPS."
        )
    if not DASHBOARD_TLS_CERT:
        return None

    cert_path = Path(DASHBOARD_TLS_CERT).resolve()
    key_path = Path(DASHBOARD_TLS_KEY).resolve()
    if not cert_path.is_file() or not key_path.is_file():
        raise FileNotFoundError(
            "Dashboard TLS certificate and key files must both exist."
        )
    return str(cert_path), str(key_path)


# ============================================================
# CURRENT SESSION
# ============================================================

# Every time main.py starts, this module is imported again.
# Therefore this timestamp represents the beginning of the
# current tracker session.

SESSION_START = datetime.now().isoformat(
    timespec="microseconds"
)


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    static_folder=DASHBOARD_DIR,
    static_url_path=""
)
app.config["MAX_CONTENT_LENGTH"] = MAX_VIDEO_UPLOAD_BYTES


# ============================================================
# INPUT SOURCE CONTROL
# ============================================================

source_requests = queue.Queue(maxsize=1)
source_status_lock = threading.Lock()
reset_status_lock = threading.Lock()
source_status = {
    "state": "starting",
    "source": None,
    "message": "Waiting for the processing pipeline.",
}
browser_frame_lock = threading.Lock()
browser_frame = None
browser_frame_version = 0
browser_frames_received = 0
browser_last_frame_received_at = None
browser_frame_dimensions = None
reset_status = {
    "state": "idle",
    "message": "",
}


def describe_source(source):
    if isinstance(source, int):
        return f"Camera {source}"

    source_text = str(source)
    if source_text == "browser":
        return "Browser camera"
    try:
        parsed = urlsplit(source_text)
    except ValueError:
        return "Video source"
    if parsed.scheme.lower() in STREAM_SCHEMES:
        host = parsed.hostname or "stream"
        try:
            port_value = parsed.port
        except ValueError:
            return f"{parsed.scheme.upper()} stream"
        port = f":{port_value}" if port_value is not None else ""
        return f"{parsed.scheme.upper()} stream ({host}{port})"

    return Path(source_text).name or "Video source"


def normalize_requested_source(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Source must be a camera index or stream URL.")

    source = value.strip()
    if source.lower() == "browser":
        return "browser"
    if source.isdecimal():
        return int(source)

    try:
        parsed = urlsplit(source)
        hostname = parsed.hostname
        parsed.port
    except ValueError as error:
        raise ValueError("Stream URL is malformed.") from error

    if parsed.scheme.lower() in STREAM_SCHEMES and hostname:
        return source

    raise ValueError(
        "Use a camera index (for example 0) or an RTSP, RTMP, HTTP, or HTTPS stream URL."
    )


def request_source(source):
    clear_frame()
    if source == "browser":
        clear_browser_frame()
    with source_status_lock:
        while True:
            try:
                source_requests.put_nowait(source)
                break
            except queue.Full:
                try:
                    source_requests.get_nowait()
                except queue.Empty:
                    continue

        label = describe_source(source)
        source_status.update({
            "state": "queued",
            "source": label,
            "message": f"Connecting to {label}.",
        })


def take_requested_source():
    try:
        return source_requests.get_nowait()
    except queue.Empty:
        return None


def set_source_status(state, source=None, message=""):
    with source_status_lock:
        source_status.update({
            "state": state,
            "source": describe_source(source) if source is not None else None,
            "message": message,
        })


def get_source_status():
    with source_status_lock:
        return dict(source_status)


def request_database_reset():
    with reset_status_lock:
        if reset_status["state"] in {"queued", "processing"}:
            raise RuntimeError("A database reset is already in progress.")
        reset_status.update({
            "state": "queued",
            "message": "Reset is queued; face identities and event history will be cleared.",
        })


def take_database_reset_request():
    with reset_status_lock:
        if reset_status["state"] != "queued":
            return False
        reset_status.update({
            "state": "processing",
            "message": "Clearing visitor profiles, events, and saved face images.",
        })
        return True


def set_database_reset_status(state, message):
    with reset_status_lock:
        reset_status.update({
            "state": state,
            "message": message,
        })


def get_database_reset_status():
    with reset_status_lock:
        return dict(reset_status)


@app.errorhandler(RequestEntityTooLarge)
def handle_oversized_upload(error):
    app.logger.warning("Rejected oversized video upload: %s", error)
    return jsonify({
        "error": (
            "Video upload exceeds the "
            f"{MAX_VIDEO_UPLOAD_BYTES // (1024 * 1024)} MB limit."
        )
    }), 413


# ============================================================
# LIVE FRAME
# ============================================================

latest_frame = None

latest_frame_version = 0
frame_condition = threading.Condition()


# ============================================================
# DASHBOARD PAGE
# ============================================================

@app.route("/")
def dashboard():

    index_path = os.path.join(
        DASHBOARD_DIR,
        "index.html"
    )

    if not os.path.exists(index_path):

        return (
            "Dashboard index.html not found.<br><br>"
            f"Expected location:<br>{index_path}",
            404
        )

    return send_from_directory(
        DASHBOARD_DIR,
        "index.html"
    )


# ============================================================
# STATIC FILES
# ============================================================

@app.route("/<path:filename>")
def static_files(filename):

    file_path = os.path.join(
        DASHBOARD_DIR,
        filename
    )

    if os.path.isfile(file_path):

        return send_from_directory(
            DASHBOARD_DIR,
            filename
        )

    return "Not Found", 404


# ============================================================
# DATABASE
# ============================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=5
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# GET CURRENT SESSION EVENTS
# ============================================================

def get_session_events():

    if not os.path.exists(DATABASE_PATH):

        return []

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                visitor_id,
                event_type,
                timestamp,
                image_path
            FROM events
            WHERE timestamp >= ?
            ORDER BY id DESC
            LIMIT 200
            """,
            (SESSION_START,)
        )

        rows = cursor.fetchall()

        return [
            {
                "id": row["id"],
                "visitor_id": row["visitor_id"],
                "event_type": row["event_type"],
                "timestamp": row["timestamp"],
                "image_path": row["image_path"]
            }
            for row in rows
        ]

    finally:

        connection.close()


def get_session_stats():
    if not os.path.exists(DATABASE_PATH):
        return {
            "unique_visitors": 0,
            "active_visitors": 0,
            "entries": 0,
            "exits": 0,
        }

    connection = get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT
                COUNT(DISTINCT CASE
                    WHEN event_type = 'ENTRY' THEN visitor_id
                END) AS unique_visitors,
                COUNT(CASE WHEN event_type = 'ENTRY' THEN 1 END) AS entries,
                COUNT(CASE WHEN event_type = 'EXIT' THEN 1 END) AS exits
            FROM events
            WHERE timestamp >= ?
            """,
            (SESSION_START,),
        )
        totals = cursor.fetchone()

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT events.event_type
                FROM events
                JOIN (
                    SELECT visitor_id, MAX(id) AS latest_id
                    FROM events
                    WHERE timestamp >= ?
                      AND event_type IN ('ENTRY', 'EXIT')
                      AND visitor_id IS NOT NULL
                    GROUP BY visitor_id
                ) AS latest
                  ON events.id = latest.latest_id
                WHERE events.event_type = 'ENTRY'
            )
            """,
            (SESSION_START,),
        )
        active_visitors = cursor.fetchone()[0]

        return {
            "unique_visitors": totals["unique_visitors"],
            "active_visitors": active_visitors,
            "entries": totals["entries"],
            "exits": totals["exits"],
        }
    finally:
        connection.close()


# ============================================================
# API - STATS
# ============================================================

@app.route("/api/stats")
def stats():

    try:

        session_stats = get_session_stats()
        unique_visitors = session_stats["unique_visitors"]
        active_visitors = session_stats["active_visitors"]
        entries = session_stats["entries"]
        exits = session_stats["exits"]

        return jsonify({

            # Current session only
            "unique_visitors": unique_visitors,

            "active_visitors": active_visitors,

            "entries": entries,

            "exits": exits,

            # Compatibility aliases
            "unique": unique_visitors,

            "active": active_visitors,

            "session_start": SESSION_START
        })

    except Exception as error:

        return jsonify({

            "unique_visitors": 0,

            "active_visitors": 0,

            "entries": 0,

            "exits": 0,

            "unique": 0,

            "active": 0,

            "error": str(error)

        }), 500


# ============================================================
# API - EVENTS
# ============================================================

@app.route("/api/events")
def events():

    try:

        return jsonify(
            get_session_events()
        )

    except Exception as error:

        return jsonify({
            "error": str(error),
            "events": []
        }), 500


# ============================================================
# API - ALL REGISTERED VISITORS
#
# This intentionally shows the complete visitor database.
# The statistics above are session-specific.
# ============================================================

@app.route("/api/visitors")
def visitors():

    try:

        connection = get_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                first_seen,
                last_seen,
                (
                    SELECT image_path
                    FROM events
                    WHERE events.visitor_id = visitors.id
                      AND image_path IS NOT NULL
                    ORDER BY id DESC
                    LIMIT 1
                ) AS image_path
            FROM visitors
            ORDER BY id DESC
            """
        )

        rows = cursor.fetchall()

        connection.close()

        result = []

        for row in rows:

            result.append({

                "id": row["id"],

                "visitor_id": row["id"],

                "first_seen": row["first_seen"],

                "last_seen": row["last_seen"],

                "face_image_url": (
                    f"/api/visitors/{row['id']}/image"
                    if resolve_face_image(row["image_path"])
                    else None
                )

            })

        return jsonify(result)

    except Exception as error:

        return jsonify({
            "error": str(error),
            "visitors": []
        }), 500


def send_managed_image(image_path):
    resolved = resolve_face_image(image_path)
    if resolved is None:
        return jsonify({"error": "No saved image is available."}), 404
    return send_file(resolved, mimetype="image/jpeg", conditional=True)


@app.route("/api/visitors/<int:visitor_id>/image")
def visitor_image(visitor_id):
    if not os.path.exists(DATABASE_PATH):
        return jsonify({"error": "Visitor database is not available."}), 404

    connection = get_connection()
    try:
        row = connection.execute(
            """
            SELECT image_path
            FROM events
            WHERE visitor_id = ?
              AND image_path IS NOT NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (visitor_id,),
        ).fetchone()
    finally:
        connection.close()

    return send_managed_image(row["image_path"] if row else None)


@app.route("/api/unrecognized-people")
def unrecognized_people():
    if not os.path.exists(DATABASE_PATH):
        return jsonify([])

    connection = get_connection()
    try:
        rows = connection.execute(
            """
            SELECT id, kind, label, first_seen, last_seen, image_path
            FROM unrecognized_sightings
            ORDER BY id DESC
            """
        ).fetchall()
    except sqlite3.OperationalError as error:
        if "no such table" not in str(error).lower():
            raise
        rows = []
    finally:
        connection.close()

    return jsonify([
        {
            "id": row["id"],
            "kind": row["kind"],
            "label": row["label"],
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
            "image_url": (
                f"/api/unrecognized-people/{row['id']}/image"
                if resolve_face_image(row["image_path"])
                else None
            ),
        }
        for row in rows
    ])


@app.route("/api/unrecognized-people/<int:sighting_id>/image")
def unrecognized_person_image(sighting_id):
    if not os.path.exists(DATABASE_PATH):
        return jsonify({"error": "Unrecognized sightings are not available."}), 404

    connection = get_connection()
    try:
        row = connection.execute(
            """
            SELECT image_path FROM unrecognized_sightings WHERE id = ?
            """,
            (sighting_id,),
        ).fetchone()
    except sqlite3.OperationalError as error:
        if "no such table" not in str(error).lower():
            raise
        row = None
    finally:
        connection.close()

    return send_managed_image(row["image_path"] if row else None)


# ============================================================
# LIVE CAMERA FRAME
# ============================================================

def update_frame(frame):

    global latest_frame
    global latest_frame_version

    if frame is None:
        return

    try:
        frame_height, frame_width = frame.shape[:2]
        if frame_width > MAX_STREAM_WIDTH:
            output_height = round(
                frame_height * MAX_STREAM_WIDTH / frame_width
            )
            frame = cv2.resize(
                frame,
                (MAX_STREAM_WIDTH, output_height),
                interpolation=cv2.INTER_AREA,
            )

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY
            ]
        )

        if not success:
            app.logger.warning("OpenCV could not encode the current dashboard frame")
            return

        with frame_condition:

            latest_frame = encoded.tobytes()
            latest_frame_version += 1
            frame_condition.notify_all()

    except cv2.error:
        app.logger.exception("Failed to encode the dashboard video frame")


def clear_frame():
    global latest_frame
    global latest_frame_version

    with frame_condition:
        placeholder = cv2.imencode(
            ".jpg",
            np.zeros((360, 640, 3), dtype=np.uint8),
        )
        latest_frame = placeholder[1].tobytes() if placeholder[0] else None
        latest_frame_version += 1
        frame_condition.notify_all()


def update_browser_frame(frame):
    global browser_frame
    global browser_frame_version
    global browser_frames_received
    global browser_last_frame_received_at
    global browser_frame_dimensions

    with browser_frame_lock:
        browser_frame = frame
        browser_frame_version += 1
        browser_frames_received += 1
        browser_last_frame_received_at = time.time()
        browser_frame_dimensions = [
            int(frame.shape[1]),
            int(frame.shape[0]),
        ]


def clear_browser_frame():
    global browser_frame
    global browser_frame_version
    with browser_frame_lock:
        browser_frame = None
        browser_frame_version += 1


def take_browser_frame(last_version):
    with browser_frame_lock:
        if browser_frame is None or browser_frame_version == last_version:
            return None, last_version
        return browser_frame, browser_frame_version


# ============================================================
# MJPEG STREAM
# ============================================================

def generate_stream():

    last_version = -1
    while True:
        with frame_condition:
            frame_condition.wait_for(
                lambda: (
                    latest_frame is not None
                    and latest_frame_version != last_version
                ),
                timeout=15,
            )
            frame = latest_frame
            version = latest_frame_version

        if frame is not None and version != last_version:
            last_version = version

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )

        time.sleep(0.03)


@app.route("/api/stream")
def stream():

    return Response(
        generate_stream(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


@app.route("/api/status")
def input_status():
    return jsonify({
        **get_source_status(),
        "browser_camera": get_browser_camera_status(),
        "reset": get_database_reset_status(),
    })


def get_browser_camera_status():
    with browser_frame_lock:
        return {
            "frames_received": browser_frames_received,
            "last_frame_received_at": browser_last_frame_received_at,
            "dimensions": browser_frame_dimensions,
        }


@app.route("/api/source", methods=["POST"])
def select_source():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        source = normalize_requested_source(payload.get("source"))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    request_source(source)
    label = describe_source(source)
    return jsonify({
        "state": "queued",
        "source": label,
        "message": f"Connecting to {label}.",
    }), 202


@app.route("/api/browser-frame", methods=["POST"])
def browser_camera_frame():
    content_length = request.content_length or 0
    if content_length > 8 * 1024 * 1024:
        return jsonify({"error": "Camera frame exceeds the 8 MB limit."}), 413

    if not (request.content_type or "").startswith("image/jpeg"):
        return jsonify({"error": "Camera frames must be sent as image/jpeg."}), 415

    image_bytes = request.get_data(cache=False)
    if not image_bytes:
        return jsonify({"error": "Camera frame is empty."}), 400

    frame = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if frame is None:
        return jsonify({"error": "Camera frame is not a valid JPEG image."}), 400

    status = get_source_status()
    if status["source"] != "Browser camera" or status["state"] not in {
        "queued",
        "running",
    }:
        return jsonify({"error": "Browser camera is not the selected source."}), 409

    update_browser_frame(frame)
    return "", 204


@app.route("/api/browser-stop", methods=["POST"])
def stop_browser_camera():
    status = get_source_status()
    if status["source"] != "Browser camera":
        return jsonify({"error": "Browser camera is not the selected source."}), 409

    clear_browser_frame()
    clear_frame()
    set_source_status(
        "paused",
        "browser",
        "Browser camera is stopped.",
    )
    return jsonify(get_source_status()), 200


@app.route("/api/reset", methods=["POST"])
def reset_database():
    try:
        request_database_reset()
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 409
    return jsonify(get_database_reset_status()), 202


@app.route("/api/upload", methods=["POST"])
def upload_video():
    uploaded_file = request.files.get("video")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Choose a video file to upload."}), 400

    extension = Path(uploaded_file.filename).suffix.lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({
            "error": (
                "Unsupported video format. Use AVI, MKV, MOV, MP4, MPEG, MPG, or WEBM."
            )
        }), 400

    destination = os.path.join(
        UPLOAD_DIR,
        f"{uuid4().hex}{extension}",
    )
    temporary_path = f"{destination}.part"

    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        uploaded_file.save(temporary_path)
        os.replace(temporary_path, destination)
    except OSError:
        app.logger.exception("Failed to save uploaded video")
        for path in (temporary_path, destination):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        return jsonify({"error": "The server could not save the uploaded video."}), 500

    request_source(destination)
    label = describe_source(destination)
    return jsonify({
        "state": "queued",
        "source": label,
        "message": f"Uploaded {label}; switching the processing source.",
    }), 202


# ============================================================
# DASHBOARD SERVER CLASS
# ============================================================

class DashboardServer:

    def __init__(
        self,
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT
    ):

        self.host = host
        self.port = port
        self.thread = None

    def start(self):

        if self.thread is not None:
            return
        ssl_context = get_dashboard_ssl_context()
        scheme = "https" if ssl_context else "http"

        def run_server():

            app.run(
                host=self.host,
                port=self.port,
                debug=False,
                threaded=True,
                use_reloader=False,
                ssl_context=ssl_context,
            )

        self.thread = threading.Thread(
            target=run_server,
            daemon=True
        )

        self.thread.start()

        print(
            f"Dashboard: {scheme}://{self.host}:{self.port}"
        )

        print(
            f"Dashboard session started: {SESSION_START}"
        )

    def update_frame(self, frame):

        update_frame(frame)

    def stop(self):

        # Flask runs in a daemon thread.
        # It will terminate automatically when main.py exits.

        self.thread = None


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    print(
        "=============================================="
    )

    print(
        "Intelligent Visitor Analytics Dashboard"
    )

    print(
        "=============================================="
    )

    print(
        f"Dashboard folder: {DASHBOARD_DIR}"
    )

    print(
        f"Database: {DATABASE_PATH}"
    )

    print(
        f"Session started: {SESSION_START}"
    )

    ssl_context = get_dashboard_ssl_context()
    scheme = "https" if ssl_context else "http"
    print(f"Dashboard: {scheme}://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    app.run(
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        debug=False,
        threaded=True,
        ssl_context=ssl_context,
    )