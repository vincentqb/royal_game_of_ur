import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import torch
from loguru import logger
from rich import print as rprint
from rich.markup import escape
from rich.progress import track

dtype = torch.float32


def configure_logger(exp_dir):
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(exp_dir / "trace.log", level="TRACE", enqueue=True, serialize=True)
    logger.add(lambda x: print(escape(x)), colorize=True, level="INFO")

    def showwarning_to_loguru(message, category, filename, lineno, file=None, line=None):
        formatted_message = warnings.formatwarning(message, category, filename, lineno, line)
        logger.warning(formatted_message)

    warnings.showwarning = showwarning_to_loguru


def parallel_map(func, args, *, description="Working...", max_workers=16, use_threads=True):
    """
    Applies func to items in args (list of tuples), preserving order.

    Args:
        func: Function to apply
        args: List of tuples of arguments
        max_workers: Number of workers (None for auto)
        use_threads: If True, use ThreadPoolExecutor; else ProcessPoolExecutor
    """
    if max_workers is None or max_workers >= 0:
        ExecutorClass = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        with ExecutorClass(max_workers=max_workers) as executor:
            futures = [executor.submit(func, *arg) for arg in args]
            for future in track(as_completed(futures), description=description, total=len(futures), transient=True):
                yield future.result()
    else:
        for arg in track(args, description=description, total=len(futures), transient=True):
            yield func(*arg)
