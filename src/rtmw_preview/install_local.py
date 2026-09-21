"""Install available local wheels before synchronizing remaining locked dependencies."""

import argparse
import logging
import re
import subprocess

from .runtime import ROOT, configure_logging

LOGGER = logging.getLogger("install")


def main() -> int:
    """Try each locally available dependency offline, then let uv fill the gaps."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uv", default="uv")
    arguments = parser.parse_args()
    configure_logging()
    uv = arguments.uv
    try:
        if not (ROOT / ".venv" / "pyvenv.cfg").is_file():
            subprocess.run([uv, "venv", "--python", "3.12", ".venv"], cwd=ROOT, check=True)
        exported = subprocess.run(
            [uv, "export", "--frozen", "--no-dev", "--no-emit-project",
             "--no-hashes", "--no-annotate", "--no-header"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        local_names = {
            re.sub(r"[-_.]+", "-", wheel.name.split("-")[0]).lower()
            for wheel in (ROOT / "package").glob("*.whl")
        }
        for requirement in exported.stdout.splitlines():
            match = re.match(r"([A-Za-z0-9_.-]+)==", requirement)
            if match is None:
                continue
            name = re.sub(r"[-_.]+", "-", match[1]).lower()
            if name not in local_names:
                continue
            LOGGER.info("Trying local wheel for %s", requirement)
            installation = subprocess.run(
                [uv, "pip", "install", "--python", ".venv", "--no-deps",
                 "--no-index", "--offline", "--find-links", "package", requirement],
                cwd=ROOT, capture_output=True, text=True,
            )
            if installation.returncode:
                LOGGER.warning(
                    "Local installation unavailable for %s; deferring to online sync: %s",
                    name, installation.stderr.strip(),
                )
            else:
                LOGGER.info("%s", installation.stderr.strip() or f"Local dependency ready: {name}")
        LOGGER.info("Synchronizing locked dependencies; downloading only remaining requirements")
        subprocess.run([uv, "sync", "--no-dev", "--locked"], cwd=ROOT, check=True)
    except (OSError, subprocess.CalledProcessError):
        LOGGER.exception("Dependency installation failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
