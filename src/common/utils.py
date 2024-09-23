import logging
import os
import re
from pathlib import Path

import yaml


def setup_logging(file_path: str, log_level: int = logging.INFO) -> None:
    """Setup logging configuration."""
    # Setup logging
    if not file_path.endswith(".log"):
        file_path = f"{file_path}.log"

    f_handler = logging.FileHandler(file_path, mode="w", encoding="utf-8")
    f_handler.setLevel(log_level)
    s_handler = logging.StreamHandler()
    s_handler.setLevel(log_level)
    logging.basicConfig(
        handlers=[f_handler, s_handler],
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_path(*parts: str | Path) -> Path:
    """Create a file path from parts."""
    return Path(*parts).resolve()


def read_text_file(*parts: str, mode: str = "r") -> str:
    """Read the content of a file as string."""
    fname = get_path(*parts)
    with Path.open(fname, mode, encoding="utf-8") as file:
        return file.read()


def file_exists(*parts: str) -> bool:
    return Path.is_file(get_path(*parts))


def dir_exists(*parts: str) -> bool:
    return Path.is_dir(get_path(*parts))


def make_folder(*parts: str) -> Path:
    """Create a folder recursively if it doesn't exist yet and return the complete path."""
    folder = get_path(*parts)
    if not dir_exists(folder):
        # Create the dependency build folder:
        Path.mkdir(folder, parents=True, exist_ok=True)
    return folder


def read_yaml(*parts: str) -> dict:
    """Read a YAML file as dict."""
    fname = get_path(*parts)
    with Path.open(fname, encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_all_files(path: Path, exp: str = ".*", recursive: bool = False) -> list[str]:
    """Get all the files matching a given pattern in a folder."""
    # prepare a pattern:
    path_str = path.resolve().as_posix() if isinstance(path, Path) else path
    pat = re.compile(exp)
    num = len(path_str) + 1
    if recursive:
        # Retrieve all files in a given folder recursively
        res = []
        for root, _directories, filenames in os.walk(path):
            for filename in filenames:
                fname = get_path(root, filename)
                if Path.is_file(fname) and pat.search(fname) is not None:
                    # Remove the folder prefix
                    res.append(fname[num:])
        return res
    return [f for f in os.listdir(path) if (Path.is_file(get_path(path, f)) and pat.search(f) is not None)]
