import os
import cv2
from datetime import datetime
from pathlib import Path


def save_face_image(face_crop, event_type):

    today = datetime.now().strftime("%Y-%m-%d")

    event_type = event_type.upper()

    if event_type == "ENTRY":
        folder_name = "entries"
    elif event_type == "EXIT":
        folder_name = "exits"
    else:
        folder_name = event_type.lower()

    folder = os.path.join(
        "logs",
        folder_name,
        today
    )

    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = f"{event_type.lower()}_{timestamp}.jpg"

    path = os.path.join(
        folder,
        filename
    )

    success = cv2.imwrite(
        path,
        face_crop
    )

    if not success:
        raise IOError(
            f"Failed to save image: {path}"
        )

    return path


def save_unrecognized_image(image_crop):
    today = datetime.now().strftime("%Y-%m-%d")
    folder = Path("logs") / "unrecognized" / today
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = folder / f"sighting_{timestamp}.jpg"
    if not cv2.imwrite(str(path), image_crop):
        raise IOError(f"Failed to save unrecognized-person image: {path}")
    return str(path)


def delete_event_images(image_paths, project_root):
    project_root = Path(project_root).resolve()
    allowed_roots = (
        (project_root / "logs" / "entries").resolve(),
        (project_root / "logs" / "exits").resolve(),
        (project_root / "logs" / "unrecognized").resolve(),
    )
    failures = []

    for image_path in image_paths:
        if not image_path:
            continue

        path = Path(image_path)
        if not path.is_absolute():
            path = project_root / path
        resolved_path = path.resolve()
        if not any(
            resolved_path.is_relative_to(root)
            for root in allowed_roots
        ):
            failures.append(
                f"evidence path is outside the managed image folders: {image_path}"
            )
            continue

        try:
            resolved_path.unlink(missing_ok=True)
        except OSError as error:
            failures.append(
                f"could not delete saved face evidence {resolved_path}: {error}"
            )

    if failures:
        raise OSError("; ".join(failures))