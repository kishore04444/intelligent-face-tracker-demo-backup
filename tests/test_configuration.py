import copy
import json
import unittest
from pathlib import Path

from src.configuration import validate_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        with (PROJECT_ROOT / "config.json").open(encoding="utf-8") as config_file:
            self.config = json.load(config_file)

    def test_current_project_configuration_is_valid(self):
        self.assertEqual(validate_config(self.config), self.config)

    def test_rejects_confidence_outside_supported_range(self):
        config = copy.deepcopy(self.config)
        config["detection"]["confidence"] = 1.0

        with self.assertRaisesRegex(ValueError, "detection.confidence"):
            validate_config(config)

    def test_rejects_wrong_boolean_type(self):
        config = copy.deepcopy(self.config)
        config["output"]["save_images"] = "false"

        with self.assertRaisesRegex(ValueError, "output.save_images"):
            validate_config(config)

    def test_rejects_fractional_frame_skip(self):
        config = copy.deepcopy(self.config)
        config["detection"]["skip_frames"] = 1.5

        with self.assertRaisesRegex(ValueError, "detection.skip_frames"):
            validate_config(config)

    def test_rejects_negative_camera_index(self):
        config = copy.deepcopy(self.config)
        config["input"]["source"] = -1

        with self.assertRaisesRegex(ValueError, "camera index"):
            validate_config(config)

    def test_rejects_non_object_configuration(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            validate_config([])


if __name__ == "__main__":
    unittest.main()
