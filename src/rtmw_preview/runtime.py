"""Project paths and daily UTC+08:00 logging."""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHINA_TIME = timezone(timedelta(hours=8))


class DailyLog(logging.Handler):
    """Route each record to its UTC+08:00 calendar date."""

    def emit(self, record: logging.LogRecord) -> None:
        date = datetime.fromtimestamp(record.created, CHINA_TIME).strftime("%Y-%m-%d")
        try:
            with (ROOT / "log" / f"log_{date}.log").open("a", encoding="utf-8") as stream:
                stream.write(self.format(record) + "\n")
        except Exception:
            self.handleError(record)


class ChinaFormatter(logging.Formatter):
    """Format timestamps independently of the operating system timezone."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return datetime.fromtimestamp(record.created, CHINA_TIME).strftime(
            "%Y-%m-%d %H:%M:%S +08:00"
        )


def configure_logging() -> None:
    """Enable console and daily file logging."""
    (ROOT / "log").mkdir(parents=True, exist_ok=True)
    formatter = ChinaFormatter("[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s")
    console = logging.StreamHandler()
    daily = DailyLog()
    for destination in (console, daily):
        destination.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[console, daily], force=True)
