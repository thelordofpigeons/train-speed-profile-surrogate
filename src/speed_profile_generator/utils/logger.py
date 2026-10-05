import logging
import os
import sys

def setup_logger(name: str, save_dir: str, level=logging.INFO):
    """
    Sets up a logger that outputs to both console and a file.

    Args:
        name (str): The name of the logger.
        save_dir (str): The directory where the log file will be saved.
        level (int): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if function is called multiple times
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, "experiment.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
