import logging
import os
from logging.handlers import RotatingFileHandler

LOG_PATH = "logs/events.log"

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("face_tracker")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def log_event(message):
    logger.info(message)