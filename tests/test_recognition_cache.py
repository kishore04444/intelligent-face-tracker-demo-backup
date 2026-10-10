import unittest

import numpy as np

from src.visitor_manager import VisitorManager


class RecognitionCacheTests(unittest.TestCase):
    def setUp(self):
        self.manager = VisitorManager.__new__(VisitorManager)
        self.manager.similarity_threshold = 0.62
        self.manager._template_visitor_ids = [5, 9]
        first = np.zeros(512, dtype=np.float32)
        first[0] = 1
        second = np.zeros(512, dtype=np.float32)
        second[1] = 1
        self.manager._template_matrix = np.stack((first, second))

    def test_matching_uses_cached_normalized_templates(self):
        embedding = np.zeros(512, dtype=np.float32)
        embedding[1] = 1

        self.assertEqual(
            self.manager.find_matching_visitor(embedding),
            (9, 1.0),
        )

    def test_unmatched_embedding_is_not_assigned(self):
        embedding = np.zeros(512, dtype=np.float32)
        embedding[2] = 1

        self.assertEqual(
            self.manager.find_matching_visitor(embedding),
            (None, 0.0),
        )

    def test_reset_clears_cached_identity_templates(self):
        self.manager.clear_template_cache()

        self.assertEqual(self.manager._template_visitor_ids, [])
        self.assertEqual(self.manager._template_matrix.shape, (0, 512))


if __name__ == "__main__":
    unittest.main()
