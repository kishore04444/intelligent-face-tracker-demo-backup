import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.database import Database
from src.image_logger import delete_event_images


class DatabaseResetTests(unittest.TestCase):
    def test_reset_clears_profiles_events_and_resets_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "data" / "visitors.db"
            with patch("src.database.DB_PATH", str(database_path)):
                database = Database()
                self.assertTrue(database_path.is_file())
                embedding = np.ones(512, dtype=np.float32)
                visitor_id = database.add_visitor(embedding)
                database.add_event(
                    visitor_id,
                    "ENTRY",
                    "logs/entries/test.jpg",
                )
                database.add_event(visitor_id, "EXIT")

                image_paths = database.reset()
                self.assertEqual(image_paths, ["logs/entries/test.jpg"])
                self.assertEqual(database.get_visitors(), [])
                self.assertEqual(database.get_all_embeddings(), [])
                self.assertEqual(
                    database.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
                    0,
                )

                self.assertEqual(database.add_visitor(embedding), 1)
                database.close()

    def test_reset_image_cleanup_only_deletes_managed_event_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            image_path = project_root / "logs" / "entries" / "entry.jpg"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"face evidence")
            unrecognized_image = (
                project_root / "logs" / "unrecognized" / "person.jpg"
            )
            unrecognized_image.parent.mkdir(parents=True)
            unrecognized_image.write_bytes(b"unrecognized evidence")
            outside_path = project_root / "private.jpg"
            outside_path.write_bytes(b"do not delete")

            with self.assertRaises(OSError):
                delete_event_images(
                    [
                        str(image_path.relative_to(project_root)),
                        str(unrecognized_image.relative_to(project_root)),
                        str(outside_path),
                    ],
                    project_root,
                )

            self.assertFalse(image_path.exists())
            self.assertFalse(unrecognized_image.exists())
            self.assertTrue(outside_path.exists())

    def test_unrecognized_sightings_persist_once_and_are_removed_by_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "visitors.db"
            with patch("src.database.DB_PATH", str(database_path)):
                database = Database()
                database.add_unrecognized_sighting(
                    "track-one",
                    "person",
                    "Person detected - no face visible",
                    "logs/unrecognized/person.jpg",
                )
                database.add_unrecognized_sighting(
                    "track-one",
                    "person",
                    "Person detected - no face visible",
                    "logs/unrecognized/replacement.jpg",
                )

                sightings = database.get_unrecognized_sightings()
                self.assertEqual(len(sightings), 1)
                self.assertEqual(
                    sightings[0][5],
                    "logs/unrecognized/person.jpg",
                )

                image_paths = database.reset()
                self.assertIn(
                    "logs/unrecognized/person.jpg",
                    image_paths,
                )
                self.assertEqual(database.get_unrecognized_sightings(), [])
                database.close()


if __name__ == "__main__":
    unittest.main()
