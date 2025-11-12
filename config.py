"""
Cấu hình chung cho hệ thống điểm danh khuôn mặt
"""
import os

# Đường dẫn thư mục gốc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn các thư mục
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ATTENDANCE_DIR = os.path.join(BASE_DIR, 'attendance')
HAARCASCADES_DIR = os.path.join(BASE_DIR, 'haarcascades')

# Đường dẫn file
HAARCASCADE_PATH = os.path.join(HAARCASCADES_DIR, 'haarcascade_frontalface_default.xml')
MODEL_PATH = os.path.join(MODELS_DIR, 'face_recognition_model.pkl')
ATTENDANCE_DB_PATH = os.path.join(ATTENDANCE_DIR, 'attendance.db')

# Cấu hình nhận diện khuôn mặt
FACE_DETECTION_SCALE_FACTOR = 1.1  # Tỷ lệ scale cho Haar Cascade
FACE_DETECTION_MIN_NEIGHBORS = 5    # Số lượng láng giềng tối thiểu
FACE_DETECTION_MIN_SIZE = (30, 30)  # Kích thước khuôn mặt tối thiểu

# Ngưỡng nhận diện
RECOGNITION_THRESHOLD = 0.6  # Ngưỡng độ tin cậy để nhận diện (0-1)
FACE_ENCODING_MODEL = 'large'  # 'small' hoặc 'large' cho face_recognition
DUPLICATE_SIMILARITY_THRESHOLD = 0.70  # Ngưỡng phát hiện trùng khuôn mặt (cosine sim)

# Cấu hình camera
CAMERA_INDEX = 0  # Index của camera (0 là camera mặc định)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Cấu hình điểm danh
ATTENDANCE_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
ATTENDANCE_DATE_FORMAT = '%Y-%m-%d'
ATTENDANCE_COOLDOWN = 300  # Thời gian chờ giữa các lần điểm danh (giây) - 5 phút

# Cấu hình thu thập ảnh
IMAGES_PER_PERSON = 1  # Số lượng ảnh thu thập cho mỗi người (mặc định 1 ảnh)
IMAGE_SIZE = (160, 160)  # Kích thước ảnh lưu trữ

# Màu sắc hiển thị (BGR format)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)

# Font chữ
FONT = 'Arial'
FONT_SCALE = 0.8
FONT_THICKNESS = 2

# Logging
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
