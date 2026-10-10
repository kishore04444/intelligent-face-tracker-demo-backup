import argparse
from pathlib import Path
from urllib.parse import urlsplit

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
LIVE_SCHEMES = {"http", "https", "rtmp", "rtmps", "rtsp", "rtsps"}


def parse_source(value):
    if value.isdecimal():
        return int(value)

    parsed = urlsplit(value)
    if parsed.scheme.lower() in LIVE_SCHEMES and parsed.hostname:
        return value

    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def main():
    parser = argparse.ArgumentParser(
        description="Run the YuNet face detector on a camera, video, or stream."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="0",
        help="Camera index, video path, or RTSP/RTMP/HTTP(S) stream URL.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to the YuNet ONNX model.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Face detection confidence threshold (0 to 1).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Process frames without opening a display window.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1.")

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.is_file():
        raise SystemExit(f"YuNet model not found: {model_path}")

    try:
        detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            args.confidence,
            0.3,
            5000,
        )
        capture = cv2.VideoCapture(parse_source(args.source))
    except cv2.error as error:
        raise SystemExit(f"Could not initialize the detector or video source: {error}")

    if not capture.isOpened():
        capture.release()
        raise SystemExit(f"Could not open video source: {args.source}")

    print("YuNet face detection started. Press Q to quit.")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Video source ended or stopped delivering frames.")
                break

            height, width = frame.shape[:2]
            detector.setInputSize((width, height))
            _, faces = detector.detect(frame)

            if faces is not None:
                for face in faces:
                    x, y, face_width, face_height = face[:4].astype(int)
                    confidence = float(face[-1])
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + face_width, y + face_height),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Face {confidence:.2f}",
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            if not args.headless:
                cv2.imshow("Intelligent Face Tracker - YuNet", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()

    print("YuNet face detection stopped.")


if __name__ == "__main__":
    main()
