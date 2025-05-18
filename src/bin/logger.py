import logging
import sys

from colorama import Fore, Style, init

init()


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
            record.levelname = colored_levelname

        if len(record.name) > 25:
            parts = record.name.split(".")
            if len(parts) > 2:
                shortened = ".".join([p[0] for p in parts[:-1]] + [parts[-1]])
                record.name = shortened

        return super().format(record)


def setup_logging(log_level=logging.INFO, log_file=None, simplified=True):
    """Set up logging configuration with improved readability"""

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)

    if simplified:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    else:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    console_formatter = ColoredFormatter(log_format, datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
