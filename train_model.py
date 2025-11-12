"""
Module huấn luyện mô hình nhận diện khuôn mặt
"""
import os
import pickle
import cv2
import numpy as np
import face_recognition
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from config import (
    DATASET_DIR, MODEL_PATH,
    FACE_ENCODING_MODEL
)
from utils.logger import setup_logger
from utils.file_handler import get_all_persons

logger = setup_logger(__name__)


class FaceTrainer:
    """
    Class huấn luyện mô hình nhận diện khuôn mặt
    """
    
    def __init__(self):
        """
        Khởi tạo FaceTrainer
        """
        self.encodings = []
        self.names = []
        self.model = None
        self.label_encoder = None
        logger.info("FaceTrainer đã được khởi tạo")
    
    def load_dataset(self):
        """
        Load dataset và tạo encodings cho các khuôn mặt
        """
        persons = get_all_persons()
        
        if not persons:
            logger.error("Không tìm thấy dữ liệu trong dataset")
            return False
        
        logger.info(f"Tìm thấy {len(persons)} người trong dataset")
        
        total_images = 0
        
        for person_name in persons:
            person_dir = os.path.join(DATASET_DIR, person_name)
            image_files = [f for f in os.listdir(person_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            logger.info(f"Đang xử lý {person_name}: {len(image_files)} ảnh")
            
            for img_file in image_files:
                img_path = os.path.join(person_dir, img_file)
                
                try:
                    # Đọc ảnh
                    image = cv2.imread(img_path)
                    if image is None:
                        logger.warning(f"Không thể đọc ảnh: {img_path}")
                        continue
                    
                    # Chuyển sang RGB (face_recognition dùng RGB)
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Tạo encoding
                    encodings = face_recognition.face_encodings(
                        rgb_image,
                        model=FACE_ENCODING_MODEL
                    )
                    
                    if len(encodings) > 0:
                        # Lấy encoding đầu tiên
                        self.encodings.append(encodings[0])
                        self.names.append(person_name)
                        total_images += 1
                    else:
                        logger.warning(f"Không phát hiện khuôn mặt trong: {img_path}")
                
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý {img_path}: {str(e)}")
        
        logger.info(f"Đã tạo encoding cho {total_images} ảnh")
        return total_images > 0
    
    def train(self):
        """
        Huấn luyện mô hình SVM
        """
        if len(self.encodings) == 0:
            logger.error("Không có dữ liệu để huấn luyện")
            return False
        
        logger.info("Bắt đầu huấn luyện mô hình...")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(self.names)
        
        # Huấn luyện SVM
        self.model = SVC(
            kernel='linear',
            probability=True,
            C=1.0,
            gamma='scale'
        )
        
        self.model.fit(self.encodings, labels)
        
        logger.info("Hoàn thành huấn luyện mô hình")
        return True
    
    def save_model(self):
        """
        Lưu mô hình đã huấn luyện
        """
        if self.model is None:
            logger.error("Chưa có mô hình để lưu")
            return False
        
        try:
            # Tạo thư mục models nếu chưa có
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            
            # Lưu mô hình và label encoder
            data = {
                'model': self.model,
                'label_encoder': self.label_encoder,
                'encodings': self.encodings,
                'names': self.names
            }
            
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Đã lưu mô hình vào {MODEL_PATH}")
            return True
        
        except Exception as e:
            logger.error(f"Lỗi khi lưu mô hình: {str(e)}")
            return False
    
    def train_and_save(self):
        """
        Thực hiện toàn bộ quy trình: load data, train, save
        """
        logger.info("=" * 50)
        logger.info("BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN")
        logger.info("=" * 50)
        
        # Load dataset
        if not self.load_dataset():
            logger.error("Không thể load dataset")
            return False
        
        # Train model
        if not self.train():
            logger.error("Không thể huấn luyện mô hình")
            return False
        
        # Save model
        if not self.save_model():
            logger.error("Không thể lưu mô hình")
            return False
        
        logger.info("=" * 50)
        logger.info("HOÀN THÀNH QUY TRÌNH HUẤN LUYỆN")
        logger.info("=" * 50)
        
        return True


def main():
    """
    Hàm main để chạy huấn luyện
    """
    print("=" * 50)
    print("HUẤN LUYỆN MÔ HÌNH NHẬN DIỆN KHUÔN MẶT")
    print("=" * 50)
    
    try:
        trainer = FaceTrainer()
        success = trainer.train_and_save()
        
        if success:
            print("\n✓ Huấn luyện mô hình thành công!")
        else:
            print("\n✗ Huấn luyện mô hình thất bại!")
    
    except Exception as e:
        logger.error(f"Lỗi: {str(e)}")
        print(f"\n✗ Lỗi: {str(e)}")


if __name__ == "__main__":
    main()
