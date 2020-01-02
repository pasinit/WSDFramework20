import logging
from logging import StreamHandler


def get_info_logger(name):
    logger = logging.getLogger(name)
    sh = StreamHandler()
    sh.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    logger.addHandler(sh)
    return logger