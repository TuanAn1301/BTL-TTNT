"""
Thread xử lý huấn luyện mô hình
"""
import os
import pickle
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from config import DATASET_DIR, MODEL_PATH, FACE_ENCODING_MODEL


class TrainingThread(QThread):
    """
    Thread huấn luyện mô hình
    """
    progress_update = pyqtSignal(str)
    training_complete = pyqtSignal(bool, str)  # success, message
    
    def __init__(self):
        super().__init__()
    
    def run(self):
        """
        Chạy huấn luyện
        """
        try:
            self.progress_update.emit("Bắt đầu huấn luyện mô hình...")
            
            # Import các thư viện cần thiết
            from utils import FaceEncoder, get_person_list
            from sklearn.svm import SVC
            from sklearn.preprocessing import LabelEncoder
            
            # Khởi tạo encoder
            face_encoder = FaceEncoder()
            
            # Load dataset
            self.progress_update.emit("Đang load dataset...")
            encodings, names = self.load_dataset(face_encoder)
            
            if len(encodings) == 0:
                self.training_complete.emit(False, "Không có dữ liệu để huấn luyện!")
                return
            
            self.progress_update.emit(f"Đã load {len(encodings)} ảnh từ {len(set(names))} người")
            
            # Encode labels
            self.progress_update.emit("Đang chuẩn bị dữ liệu...")
            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(names)
            
            # Huấn luyện SVM
            self.progress_update.emit("Đang huấn luyện mô hình SVM...")
            model = SVC(
                kernel='linear',
                probability=True,
                C=1.0,
                gamma='scale'
            )
            
            model.fit(encodings, labels)
            
            # Lưu mô hình
            self.progress_update.emit("Đang lưu mô hình...")
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            
            data = {
                'model': model,
                'label_encoder': label_encoder,
                'encodings': encodings,
                'names': names
            }
            
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(data, f)
            
            message = f"✓ Huấn luyện thành công!\n"
            message += f"Số người: {len(label_encoder.classes_)}\n"
            message += f"Tổng ảnh: {len(encodings)}\n"
            message += f"Danh sách: {', '.join(label_encoder.classes_)}"
            
            self.progress_update.emit(message)
            self.training_complete.emit(True, message)
        
        except Exception as e:
            error_msg = f"✗ Lỗi huấn luyện: {str(e)}"
            self.progress_update.emit(error_msg)
            self.training_complete.emit(False, error_msg)
    
    def load_dataset(self, face_encoder):
        """
        Load dataset và tạo encodings
        """
        encodings = []
        names = []
        
        # Lấy danh sách người
        from utils import get_person_list
        persons = get_person_list()
        
        if not persons:
            return encodings, names
        
        self.progress_update.emit(f"Tìm thấy {len(persons)} người trong dataset")
        
        for person_name in persons:
            person_dir = os.path.join(DATASET_DIR, person_name)
            image_files = [f for f in os.listdir(person_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            self.progress_update.emit(f"Đang xử lý {person_name}: {len(image_files)} ảnh")
            
            for img_file in image_files:
                img_path = os.path.join(person_dir, img_file)
                
                try:
                    # Đọc ảnh
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    
                    # Chuyển sang RGB
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Tạo encoding (trích xuất đặc trưng mắt, mũi, miệng)
                    encoding = face_encoder.encode_face(rgb_image)
                    
                    if encoding is not None:
                        encodings.append(encoding)
                        names.append(person_name)
                
                except Exception as e:
                    print(f"Lỗi xử lý {img_path}: {e}")
            
            self.progress_update.emit(f"✓ Hoàn thành {person_name}")
        
        return encodings, names
