import logging
import os

from asyncssh import logger 


def get_logger(file_name):
    log_dir="logs"
    os.makedirs(log_dir,exist_ok=True)

    logger=logging.getLogger(file_name)
    logger.setLevel("DEBUG")

    consoleHandler=logging.StreamHandler()
    consoleHandler.setLevel("DEBUG")

    log_file_path=os.path.join(log_dir,file_name +".log")
    fileHandler=logging.FileHandler(log_file_path)
    fileHandler.setLevel("DEBUG")

    formatter=logging.Formatter("%(asctime)s -%(name)s - %(levelname)s - %(message)s")
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    return logger