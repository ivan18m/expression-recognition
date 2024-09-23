"""Constant variables and configurations for the project."""

import logging
import os
from pathlib import Path
from typing import Final

from src.common.utils import read_yaml

_log = logging.getLogger(__name__)

ROOT_DIR: Final[str] = Path(os.path.realpath(__file__)).parent.parent.parent
CONFIG = type("Config", (object,), read_yaml(ROOT_DIR, "config.yaml"))()
_log.info("Loaded config.yaml for %s", CONFIG.app_name)
