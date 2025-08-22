"""Base report class for GenTEval."""

import json
import pathlib
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any


class BaseReport(ABC):
    """Base class for all report generators."""

    def __init__(self, compressors: list[str], root_dir: pathlib.Path):
        """
        Initialize the report generator.

        Args:
            compressors: List of compressor names to evaluate
            root_dir: Root directory containing the output data
        """
        self.compressors = compressors
        self.root_dir = root_dir
        # [report_group][group] = defaultdict(list)
        self.report = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    @abstractmethod
    def generate(self, run_dirs) -> dict[str, Any]:
        """
        Generate the report.

        Args:
            run_dirs: Function that yields (app_name, service, fault, run) tuples

        Returns:
            Dictionary containing the report data
        """

    def load_json_file(self, file_path: pathlib.Path) -> dict[str, Any]:
        """
        Load and parse a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Parsed JSON data
        """
        return json.loads(file_path.read_text())

    def file_exists(self, file_path: pathlib.Path) -> bool:
        """
        Check if a file exists.

        Args:
            file_path: Path to check

        Returns:
            True if file exists, False otherwise
        """
        return file_path.exists()

    def print_skip_message(self, message: str) -> None:
        """
        Print a skip message.

        Args:
            message: Message to print
        """
        print(message)
