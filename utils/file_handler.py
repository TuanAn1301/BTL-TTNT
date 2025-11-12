"""
Module xử lý file và thư mục
"""
import os
import csv
from datetime import datetime
from config import (
    DATASET_DIR, MODELS_DIR, ATTENDANCE_DIR, HAARCASCADES_DIR,
    ATTENDANCE_TIME_FORMAT, ATTENDANCE_DATE_FORMAT
)
from .logger import setup_logger

logger = setup_logger(__name__)


def ensure_directories():
    """
    Tạo các thư mục cần thiết nếu chưa tồn tại
    """
    directories = [DATASET_DIR, MODELS_DIR, ATTENDANCE_DIR, HAARCASCADES_DIR]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Đã tạo thư mục: {directory}")
        else:
            logger.debug(f"Thư mục đã tồn tại: {directory}")


def save_attendance(name, confidence):
    """
    Lưu thông tin điểm danh vào file CSV
    
    Args:
        name: Tên người điểm danh
        confidence: Độ tin cậy của nhận diện
    """
    now = datetime.now()
    date_str = now.strftime(ATTENDANCE_DATE_FORMAT)
    time_str = now.strftime(ATTENDANCE_TIME_FORMAT)
    
    # Tên file theo ngày
    filename = os.path.join(ATTENDANCE_DIR, f"attendance_{date_str}.csv")
    
    # Kiểm tra file có tồn tại không
    file_exists = os.path.exists(filename)
    
    # Ghi vào file CSV
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Ghi header nếu file mới
        if not file_exists:
            writer.writerow(['Tên', 'Thời gian', 'Độ tin cậy'])
        
        # Ghi dữ liệu điểm danh
        writer.writerow([name, time_str, f"{confidence:.2f}"])
    
    logger.info(f"Đã lưu điểm danh: {name} - {time_str} - Confidence: {confidence:.2f}")


def get_person_image_dir(person_name):
    """
    Lấy đường dẫn thư mục ảnh của một người
    
    Args:
        person_name: Tên người
        
    Returns:
        Đường dẫn thư mục
    """
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    return person_dir


def get_all_persons():
    """
    Lấy danh sách tất cả người trong dataset
    
    Returns:
        List tên người
    """
    if not os.path.exists(DATASET_DIR):
        return []
    
    persons = [d for d in os.listdir(DATASET_DIR) 
               if os.path.isdir(os.path.join(DATASET_DIR, d))]
    return persons
