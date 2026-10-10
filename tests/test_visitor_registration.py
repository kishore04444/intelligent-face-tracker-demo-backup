import unittest
from unittest.mock import MagicMock

import numpy as np

from src.visitor_manager import VisitorManager


class VisitorRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.manager = VisitorManager.__new__(VisitorManager)
        self.manager.db = MagicMock()
        self.manager.db.add_visitor.return_value = 14
        self.manager.similarity_threshold = 0.62
        self.manager.pending_similarity_threshold = 0.60
        self.manager.template_add_threshold = 0.70
        self.manager.required_confirmation = 2
        self.manager.track_to_visitor = {}
        self.manager.pending_tracks = {}
        self.manager._template_visitor_ids = []
        self.manager._template_matrix = np.empty((0, 512), dtype=np.float32)

    def test_new_visitor_registers_after_two_consistent_embeddings(self):
        embedding = np.zeros(512, dtype=np.float32)
        embedding[0] = 1

        self.assertEqual(
            self.manager.identify(27, embedding),
            (None, False),
        )
        self.assertIn(27, self.manager.pending_tracks)

        self.assertEqual(
            self.manager.identify(27, embedding),
            (14, True),
        )
        self.manager.db.add_visitor.assert_called_once()
        self.assertEqual(self.manager.track_to_visitor[27], 14)


if __name__ == "__main__":
    unittest.main()
