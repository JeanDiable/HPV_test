import logging
import time
import os


def get_logger(log_dir: str = None):
    logger = logging.getLogger(__name__)

    logger.setLevel('INFO')
    
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    filename = os.path.join(log_dir, f'exp_{timestamp}.log')

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename)

    formatter = logging.Formatter("%(asctime)s, %(levelname)s: %(message)s", datefmt="%Y.%m.%d, %H:%M:%S")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)

    logger.addHandler(handler1)
    logger.addHandler(handler2)

    return logger
