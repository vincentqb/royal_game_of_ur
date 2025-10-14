import warnings

import torch
import tqdm
from loguru import logger

dtype = torch.float32


def configure_logger(exp_dir):
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(exp_dir / "trace.log", level="TRACE", enqueue=True, serialize=True)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")

    def showwarning_to_loguru(message, category, filename, lineno, file=None, line=None):
        formatted_message = warnings.formatwarning(message, category, filename, lineno, line)
        logger.warning(formatted_message)

    warnings.showwarning = showwarning_to_loguru
