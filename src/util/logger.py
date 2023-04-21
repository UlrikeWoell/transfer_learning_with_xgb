import logging

from src.util.config_reader import Configuration

level_mapping = {"debug": logging.DEBUG,
                 "info": logging.INFO,
                 "warn": logging.WARN}

c = Configuration().get()
log_config = c["LOGGING"]
log_file = log_config["general_logfile"]
log_level = level_mapping[log_config["log_level"]]

logging.basicConfig(filename=log_file,
                    level= log_level,
                    format='%(asctime)s %(levelname)s: %(message)s')

def log_debug(msg:str)->None:
    logging.debug(msg)

def log_info(msg:str)->None:
    logging.info(msg)

def log_warning(msg:str)->None:
    logging.warn(msg)




