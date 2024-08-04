import logging
import os

def configure_logging(log_name, log_dir, log_filename):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    log_file_path = os.path.join(log_dir, log_filename)
    
    # Create file handler which logs even debug messages
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    
    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_file_path