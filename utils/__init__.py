"""
Utils package cho hệ thống điểm danh khuôn mặt
"""
from .logger import setup_logger
from .file_handler import ensure_directories, save_attendance
from .face_detection import FaceDetector
from .helper import (
    ensure_directory,
    init_attendance_database,
    save_attendance_record,
    load_attendance_records,
    get_attendance_summary,
    get_person_list,
    count_person_images,
    migrate_attendance_csv_to_db,
    upsert_student_info,
    get_student_info,
    student_id_exists,
    find_student_by_student_id,
    save_student_encoding,
    load_student_encodings,
    list_all_students,
    delete_student
)

# Import FaceEncoder sau để tránh lỗi khi face_recognition chưa cài
try:
    from .face_encoding import FaceEncoder
except ImportError:
    FaceEncoder = None

__all__ = [
    'setup_logger',
    'ensure_directories',
    'save_attendance',
    'init_attendance_database',
    'FaceDetector',
    'FaceEncoder',
    'ensure_directory',
    'save_attendance_record',
    'load_attendance_records',
    'get_attendance_summary',
    'get_person_list',
    'count_person_images',
    'migrate_attendance_csv_to_db',
    'upsert_student_info',
    'get_student_info',
    'student_id_exists',
    'find_student_by_student_id',
    'save_student_encoding',
    'load_student_encodings',
    'list_all_students',
    'delete_student'
]
