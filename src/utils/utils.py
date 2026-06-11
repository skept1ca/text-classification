"""utils.py — логирование, set_seed, skip_if_exists, timer."""

import logging
import random
import time
from pathlib import Path
import numpy as np


def get_logger(name: str, log_dir: Path = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def skip_if_exists(path: Path, label: str, skip: bool, logger) -> bool:
    if skip and Path(path).exists():
        logger.info(f"[SKIP] {label} — файл уже есть: {path}")
        return True
    return False


def timer(logger, label: str):
    class _T:
        def __enter__(self):
            self.t = time.time()
            logger.info(f"▶ {label}")
            return self
        def __exit__(self, *_):
            m, s = divmod(int(time.time() - self.t), 60)
            logger.info(f"✓ {label} завершён ({m}м {s}с)")
    return _T()
