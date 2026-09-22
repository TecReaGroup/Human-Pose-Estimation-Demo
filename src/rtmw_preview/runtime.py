"""Project paths and daily UTC+08:00 logging."""

import logging
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHINA_TIME = timezone(timedelta(hours=8))
PERF = 15
LOG_LEVEL = {
    "DEBUG": logging.DEBUG,
    "PERF": PERF,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


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
    """Configure console and daily file logging with the configured minimum level."""
    with (ROOT / "config" / "config.toml").open("rb") as stream:
        level_name = tomllib.load(stream).get("logging", {}).get("level", "INFO")
    if not isinstance(level_name, str) or level_name.upper() not in LOG_LEVEL:
        raise ValueError(f"Invalid logging.level: {level_name!r}; expected {', '.join(LOG_LEVEL)}")
    logging.addLevelName(PERF, "PERF")
    (ROOT / "log").mkdir(parents=True, exist_ok=True)
    formatter = ChinaFormatter("[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s")
    console = logging.StreamHandler()
    daily = DailyLog()
    for destination in (console, daily):
        destination.setFormatter(formatter)
    logging.basicConfig(level=LOG_LEVEL[level_name.upper()], handlers=[console, daily], force=True)
