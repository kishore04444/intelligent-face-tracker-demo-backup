import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import dashboard_server


class DashboardSourceTests(unittest.TestCase):
    def setUp(self):
        self.client = dashboard_server.app.test_client()
        while dashboard_server.take_requested_source() is not None:
            pass
        dashboard_server.clear_browser_frame()
        dashboard_server.set_source_status(
            "idle",
            None,
            "Waiting for a video, stream, or browser camera.",
        )
        dashboard_server.set_database_reset_status("idle", "")

    def test_camera_source_is_queued(self):
        response = self.client.post(
            "/api/source",
            json={"source": "0"},
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(dashboard_server.take_requested_source(), 0)
        self.assertEqual(
            dashboard_server.get_source_status()["state"],
            "queued",
        )

    def test_browser_camera_frame_is_accepted_for_selected_source(self):
        response = self.client.post(
            "/api/source",
            json={"source": "browser"},
        )
        self.assertEqual(response.status_code, 202)

        success, encoded = cv2.imencode(
            ".jpg",
            np.full((24, 32, 3), 127, dtype=np.uint8),
        )
        self.assertTrue(success)
        version = dashboard_server.browser_frame_version
        response = self.client.post(
            "/api/browser-frame",
            data=encoded.tobytes(),
            content_type="image/jpeg",
        )

        self.assertEqual(response.status_code, 204)
        frame, updated_version = dashboard_server.take_browser_frame(version)
        self.assertEqual(updated_version, version + 1)
        self.assertEqual(frame.shape, (24, 32, 3))
        camera_status = self.client.get("/api/status").get_json()["browser_camera"]
        self.assertGreater(camera_status["frames_received"], 0)
        self.assertEqual(camera_status["dimensions"], [32, 24])

    def test_changing_source_clears_the_previous_video_frame(self):
        dashboard_server.update_frame(
            np.full((24, 32, 3), 255, dtype=np.uint8),
        )
        response = self.client.post(
            "/api/source",
            json={"source": "browser"},
        )

        self.assertEqual(response.status_code, 202)
        with dashboard_server.frame_condition:
            displayed = cv2.imdecode(
                np.frombuffer(dashboard_server.latest_frame, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
        self.assertEqual(int(displayed.max()), 0)

    def test_browser_camera_rejects_invalid_frames_and_non_browser_source(self):
        response = self.client.post(
            "/api/browser-frame",
            data=b"not a jpeg",
            content_type="image/jpeg",
        )
        self.assertEqual(response.status_code, 400)

        response = self.client.post(
            "/api/source",
            json={"source": "0"},
        )
        self.assertEqual(response.status_code, 202)
        success, encoded = cv2.imencode(
            ".jpg",
            np.zeros((24, 32, 3), dtype=np.uint8),
        )
        self.assertTrue(success)
        response = self.client.post(
            "/api/browser-frame",
            data=encoded.tobytes(),
            content_type="image/jpeg",
        )
        self.assertEqual(response.status_code, 409)

    def test_stopping_browser_camera_clears_the_displayed_frame(self):
        self.client.post("/api/source", json={"source": "browser"})
        response = self.client.post("/api/browser-stop")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["state"], "paused")
        with dashboard_server.frame_condition:
            displayed = cv2.imdecode(
                np.frombuffer(dashboard_server.latest_frame, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
        self.assertEqual(int(displayed.max()), 0)

    def test_dashboard_exposes_reset_control_and_status(self):
        page = self.client.get("/")
        status = self.client.get("/api/status").get_json()

        self.assertEqual(page.status_code, 200)
        self.assertIn(b'id="resetDatabaseButton"', page.data)
        self.assertIn("reset", status)
        self.assertEqual(status["reset"]["state"], "idle")

    def test_unrecognized_sightings_are_persistent_and_separate_from_visitors(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            image_directory = (
                project_root / "logs" / "unrecognized" / "2026-10-07"
            )
            image_directory.mkdir(parents=True)
            image_path = image_directory / "person.jpg"
            image_ok, encoded_image = cv2.imencode(
                ".jpg",
                np.full((16, 16, 3), 180, dtype=np.uint8),
            )
            self.assertTrue(image_ok)
            image_path.write_bytes(encoded_image.tobytes())

            database_path = Path(directory) / "visitors.db"
            connection = sqlite3.connect(database_path)
            connection.executescript(
                """
                CREATE TABLE visitors (
                    id INTEGER PRIMARY KEY,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    embedding BLOB NOT NULL
                );
                CREATE TABLE events (
                    id INTEGER PRIMARY KEY,
                    visitor_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    image_path TEXT
                );
                CREATE TABLE unrecognized_sightings (
                    id INTEGER PRIMARY KEY,
                    tracker_key TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,
                    label TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    image_path TEXT
                );
                INSERT INTO unrecognized_sightings
                    (tracker_key, kind, label, first_seen, last_seen, image_path)
                VALUES
                    ('unique-track-key', 'person',
                     'Person detected - no face visible',
                     'first timestamp', 'first timestamp',
                     'logs/unrecognized/2026-10-07/person.jpg');
                """
            )
            connection.commit()
            connection.close()

            with patch.object(
                dashboard_server,
                "DATABASE_PATH",
                str(database_path),
            ), patch.object(
                dashboard_server,
                "BASE_DIR",
                str(project_root),
            ), patch.object(
                dashboard_server,
                "MANAGED_FACE_IMAGE_DIRS",
                ((project_root / "logs" / "unrecognized").resolve(),),
            ):
                first_response = self.client.get("/api/unrecognized-people")
                first_sighting = first_response.get_json()[0]
                image_response = self.client.get(first_sighting["image_url"])
                dashboard_server.request_source("browser")
                self.assertEqual(
                    dashboard_server.take_requested_source(),
                    "browser",
                )
                second_response = self.client.get("/api/unrecognized-people")
                image_response_data = image_response.data
                image_response.close()
                connection = sqlite3.connect(database_path)
                visitor_count = connection.execute(
                    "SELECT COUNT(*) FROM visitors",
                ).fetchone()[0]
                event_count = connection.execute(
                    "SELECT COUNT(*) FROM events",
                ).fetchone()[0]
                connection.close()

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(len(first_response.get_json()), 1)
        self.assertEqual(
            first_response.get_json(),
            second_response.get_json(),
        )
        self.assertEqual(
            first_sighting["image_url"],
            "/api/unrecognized-people/1/image",
        )
        self.assertEqual(image_response.status_code, 200)
        self.assertEqual(image_response.mimetype, "image/jpeg")
        self.assertGreater(len(image_response_data), 0)
        self.assertEqual(visitor_count, 0)
        self.assertEqual(event_count, 0)

    def test_visitors_include_and_serve_latest_saved_face_image(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            image_dir = project_root / "logs" / "entries" / "2026-10-07"
            image_dir.mkdir(parents=True)
            image_path = image_dir / "entry.jpg"
            success, encoded = cv2.imencode(
                ".jpg",
                np.full((20, 20, 3), 160, dtype=np.uint8),
            )
            self.assertTrue(success)
            image_path.write_bytes(encoded.tobytes())

            database_path = project_root / "visitors.db"
            connection = sqlite3.connect(database_path)
            connection.executescript(
                """
                CREATE TABLE visitors (
                    id INTEGER PRIMARY KEY,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL
                );
                CREATE TABLE events (
                    id INTEGER PRIMARY KEY,
                    visitor_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    image_path TEXT
                );
                INSERT INTO visitors VALUES (7, 'first', 'last');
                INSERT INTO visitors VALUES (8, 'first', 'last');
                """
            )
            connection.execute(
                """
                INSERT INTO events
                    (visitor_id, event_type, timestamp, image_path)
                VALUES (?, ?, ?, ?)
                """,
                (
                    7,
                    "ENTRY",
                    "2026-10-07T12:00:00",
                    str(image_path.relative_to(project_root)),
                ),
            )
            outside_image = project_root / "private.jpg"
            outside_image.write_bytes(b"private")
            connection.execute(
                """
                INSERT INTO events
                    (visitor_id, event_type, timestamp, image_path)
                VALUES (?, ?, ?, ?)
                """,
                (
                    8,
                    "ENTRY",
                    "2026-10-07T12:00:01",
                    str(outside_image),
                ),
            )
            connection.commit()
            connection.close()

            with (
                patch.object(dashboard_server, "BASE_DIR", str(project_root)),
                patch.object(
                    dashboard_server,
                    "MANAGED_FACE_IMAGE_DIRS",
                    (
                        (project_root / "logs" / "entries").resolve(),
                        (project_root / "logs" / "exits").resolve(),
                    ),
                ),
                patch.object(
                    dashboard_server,
                    "DATABASE_PATH",
                    str(database_path),
                ),
            ):
                visitors = self.client.get("/api/visitors").get_json()
                image_response = self.client.get(
                    "/api/visitors/7/image",
                )
                rejected_image_response = self.client.get(
                    "/api/visitors/8/image",
                )
                image_response_data = image_response.data
                image_response.close()
                rejected_image_response.close()

        visitor_images = {visitor["id"]: visitor["face_image_url"] for visitor in visitors}
        self.assertEqual(
            visitor_images[7],
            "/api/visitors/7/image",
        )
        self.assertIsNone(visitor_images[8])
        self.assertEqual(image_response.status_code, 200)
        self.assertEqual(image_response.mimetype, "image/jpeg")
        self.assertGreater(len(image_response_data), 0)
        self.assertEqual(rejected_image_response.status_code, 404)

    def test_live_stream_url_is_queued(self):
        response = self.client.post(
            "/api/source",
            json={"source": "rtsp://operator:secret@camera.example/live"},
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(
            dashboard_server.take_requested_source(),
            "rtsp://operator:secret@camera.example/live",
        )
        status = self.client.get("/api/status").get_json()
        self.assertNotIn("secret", str(status))
        self.assertEqual(status["source"], "RTSP stream (camera.example)")

    def test_live_stream_rejects_invalid_port(self):
        response = self.client.post(
            "/api/source",
            json={"source": "rtsp://camera.example:99999/live"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIsNone(dashboard_server.take_requested_source())

    def test_source_rejects_unsupported_values(self):
        response = self.client.post(
            "/api/source",
            json={"source": "../private/video.mp4"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_database_reset_is_queued_once_for_the_processing_thread(self):
        response = self.client.post("/api/reset")
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json()["state"], "queued")

        duplicate = self.client.post("/api/reset")
        self.assertEqual(duplicate.status_code, 409)

        self.assertTrue(dashboard_server.take_database_reset_request())
        self.assertEqual(
            dashboard_server.get_database_reset_status()["state"],
            "processing",
        )
        self.assertFalse(dashboard_server.take_database_reset_request())
        dashboard_server.set_database_reset_status("complete", "Reset complete.")

    def test_upload_is_saved_with_generated_name_and_queued(self):
        with tempfile.TemporaryDirectory() as upload_dir:
            with patch.object(dashboard_server, "UPLOAD_DIR", upload_dir):
                response = self.client.post(
                    "/api/upload",
                    data={
                        "video": (
                            io.BytesIO(b"sample video bytes"),
                            "../../unsafe-name.MP4",
                        )
                    },
                    content_type="multipart/form-data",
                )

            self.assertEqual(response.status_code, 202)
            source = dashboard_server.take_requested_source()
            self.assertIsNotNone(source)
            self.assertEqual(Path(source).parent, Path(upload_dir))
            self.assertRegex(Path(source).name, r"^[0-9a-f]{32}\.mp4$")
            self.assertEqual(Path(source).read_bytes(), b"sample video bytes")

    def test_upload_rejects_unsupported_file_extension(self):
        response = self.client.post(
            "/api/upload",
            data={
                "video": (
                    io.BytesIO(b"not a video"),
                    "payload.exe",
                )
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported video format", response.get_json()["error"])
        self.assertIsNone(dashboard_server.take_requested_source())

    def test_stats_aggregate_database_events_without_loading_event_list(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "visitors.db"
            connection = sqlite3.connect(database_path)
            connection.execute(
                """
                CREATE TABLE events (
                    id INTEGER PRIMARY KEY,
                    visitor_id INTEGER,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    image_path TEXT
                )
                """
            )
            connection.executemany(
                """
                INSERT INTO events (visitor_id, event_type, timestamp)
                VALUES (?, ?, ?)
                """,
                [
                    (1, "ENTRY", dashboard_server.SESSION_START),
                    (1, "EXIT", dashboard_server.SESSION_START),
                    (2, "ENTRY", dashboard_server.SESSION_START),
                    (3, "ENTRY", dashboard_server.SESSION_START),
                    (3, "EXIT", dashboard_server.SESSION_START),
                    (3, "ENTRY", dashboard_server.SESSION_START),
                ],
            )
            connection.commit()
            connection.close()

            with patch.object(dashboard_server, "DATABASE_PATH", str(database_path)):
                response = self.client.get("/api/stats")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {
                "unique_visitors": 3,
                "active_visitors": 2,
                "entries": 4,
                "exits": 2,
                "unique": 3,
                "active": 2,
                "session_start": dashboard_server.SESSION_START,
            },
        )

    def test_events_endpoint_returns_only_latest_two_hundred(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "visitors.db"
            connection = sqlite3.connect(database_path)
            connection.execute(
                """
                CREATE TABLE events (
                    id INTEGER PRIMARY KEY,
                    visitor_id INTEGER,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    image_path TEXT
                )
                """
            )
            connection.executemany(
                """
                INSERT INTO events (visitor_id, event_type, timestamp)
                VALUES (?, ?, ?)
                """,
                [
                    (index, "ENTRY", dashboard_server.SESSION_START)
                    for index in range(205)
                ],
            )
            connection.commit()
            connection.close()

            with patch.object(dashboard_server, "DATABASE_PATH", str(database_path)):
                response = self.client.get("/api/events")

        events = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(events), 200)
        self.assertEqual(events[0]["visitor_id"], 204)
        self.assertEqual(events[-1]["visitor_id"], 5)

    def test_dashboard_frame_is_downscaled_before_jpeg_encoding(self):
        with dashboard_server.frame_condition:
            previous_frame = dashboard_server.latest_frame
            previous_version = dashboard_server.latest_frame_version

        try:
            frame = np.zeros((900, 1600, 3), dtype=np.uint8)
            dashboard_server.update_frame(frame)
            with dashboard_server.frame_condition:
                encoded_frame = dashboard_server.latest_frame

            decoded = cv2.imdecode(
                np.frombuffer(encoded_frame, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            self.assertEqual(decoded.shape[:2], (720, 1280))
        finally:
            with dashboard_server.frame_condition:
                dashboard_server.latest_frame = previous_frame
                dashboard_server.latest_frame_version = previous_version


if __name__ == "__main__":
    unittest.main()
