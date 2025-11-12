"""
Module xử lý logging cho hệ thống
"""
import logging
import sys
from config import LOG_LEVEL, LOG_FORMAT


def setup_logger(name):
    """
    Thiết lập logger cho module
    
    Args:
        name: Tên của logger
        
    Returns:
        Logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Tránh thêm handler trùng lặp
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, LOG_LEVEL))
        
        # Formatter
        formatter = logging.Formatter(LOG_FORMAT)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    return logger
