import logging

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.setLevel(logging.DEBUG)
    return logger
