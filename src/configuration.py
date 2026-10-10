import math
from numbers import Real


def _section(config, name):
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"config.json: '{name}' must be an object.")
    return value


def _number(
    section,
    section_name,
    key,
    minimum=None,
    maximum=None,
    minimum_inclusive=True,
    maximum_inclusive=True,
):
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"config.json: '{section_name}.{key}' must be a number.")
    if not math.isfinite(value):
        raise ValueError(f"config.json: '{section_name}.{key}' must be finite.")
    if minimum is not None and (
        value < minimum or (not minimum_inclusive and value == minimum)
    ):
        operator = ">=" if minimum_inclusive else ">"
        raise ValueError(
            f"config.json: '{section_name}.{key}' must be {operator} {minimum}."
        )
    if maximum is not None and (
        value > maximum or (not maximum_inclusive and value == maximum)
    ):
        operator = "<=" if maximum_inclusive else "<"
        raise ValueError(
            f"config.json: '{section_name}.{key}' must be {operator} {maximum}."
        )
    return value


def _integer(section, section_name, key, minimum):
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"config.json: '{section_name}.{key}' must be an integer.")
    if value < minimum:
        raise ValueError(f"config.json: '{section_name}.{key}' must be >= {minimum}.")


def _boolean(section, section_name, key):
    if key in section and not isinstance(section[key], bool):
        raise ValueError(f"config.json: '{section_name}.{key}' must be true or false.")


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object.")

    input_config = _section(config, "input")
    source = input_config.get("source")
    if isinstance(source, bool) or not isinstance(source, (int, str)):
        raise ValueError(
            "config.json: 'input.source' must be a camera index, path, URL, or 'browser'."
        )
    if isinstance(source, int) and source < 0:
        raise ValueError("config.json: 'input.source' camera index cannot be negative.")
    if isinstance(source, str) and not source.strip():
        raise ValueError("config.json: 'input.source' cannot be empty.")

    detection = _section(config, "detection")
    _number(
        detection,
        "detection",
        "confidence",
        0,
        1,
        minimum_inclusive=False,
        maximum_inclusive=False,
    )
    _integer(detection, "detection", "skip_frames", 0)
    _integer(detection, "detection", "image_size", 320)
    _integer(detection, "detection", "min_face_size", 1)
    _boolean(detection, "detection", "validate_with_yunet")
    if "yunet_confidence" in detection:
        _number(detection, "detection", "yunet_confidence", 0, 1)

    person_detection = config.get("person_detection", {})
    if not isinstance(person_detection, dict):
        raise ValueError("config.json: 'person_detection' must be an object.")
    _boolean(person_detection, "person_detection", "enabled")
    if "max_dimension" in person_detection:
        _integer(person_detection, "person_detection", "max_dimension", 320)
    if "score_threshold" in person_detection:
        _number(person_detection, "person_detection", "score_threshold")
    if "min_hits" in person_detection:
        _integer(person_detection, "person_detection", "min_hits", 1)
    if "max_missing" in person_detection:
        _integer(person_detection, "person_detection", "max_missing", 0)

    recognition = _section(config, "recognition")
    _number(recognition, "recognition", "similarity_threshold", 0, 1)

    tracking = _section(config, "tracking")
    _integer(tracking, "tracking", "max_missing_frames", 1)

    output = _section(config, "output")
    for key in ("show_window", "save_images"):
        if key not in output:
            raise ValueError(f"config.json: 'output.{key}' is required.")
    _boolean(output, "output", "show_window")
    _boolean(output, "output", "save_images")

    return config
