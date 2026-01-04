import logging
import torch
import numpy as np
import random
from pathlib import Path
import time


def set_seed(seed, cudnn_enabled=True):
    """for reproducibility

    :param seed:
    :return:
    """

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = cudnn_enabled
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_logger(args):
    logger = logging.getLogger(args.log_name)
    logger.setLevel(args.log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f'{args.log_name}_{time.asctime()}.log')
    file_handler.setLevel(args.log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_device(cuda=True, gpus='0'):
    return torch.device("cuda:" + gpus if torch.cuda.is_available() and cuda else "cpu")