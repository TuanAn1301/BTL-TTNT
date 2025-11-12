"""
Module huấn luyện mô hình đơn giản - KHÔNG CẦN face-recognition
Sử dụng OpenCV và template matching
"""
import os
import pickle
import cv2
import numpy as np
from config import DATASET_DIR, MODEL_PATH
from utils import setup_logger, get_person_list

logger = setup_logger(__name__)


class SimpleTrainer:
    """
    Huấn luyện mô hình đơn giản với OpenCV
    """
    
    def __init__(self):
        self.templates = {}  # Lưu template ảnh của mỗi người
        self.names = []
    
    def load_dataset(self):
        """
        Load dataset
        """
        persons = get_person_list()
        
        if not persons:
            logger.error("Không tìm thấy dữ liệu trong dataset")
            return False
        
        logger.info(f"Tìm thấy {len(persons)} người trong dataset")
        
        for person_name in persons:
            person_dir = os.path.join(DATASET_DIR, person_name)
            image_files = [f for f in os.listdir(person_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                continue
            
            logger.info(f"Đang xử lý {person_name}: {len(image_files)} ảnh")
            
            # Load tất cả ảnh của người này
            person_images = []
            for img_file in image_files:
                img_path = os.path.join(person_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize về kích thước chuẩn
                    img = cv2.resize(img, (100, 100))
                    person_images.append(img)
            
            if person_images:
                self.templates[person_name] = person_images
                self.names.append(person_name)
                logger.info(f"✓ Đã load {len(person_images)} ảnh cho {person_name}")
        
        return len(self.templates) > 0
    
    def save_model(self):
        """
        Lưu mô hình
        """
        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            
            data = {
                'templates': self.templates,
                'names': self.names,
                'type': 'simple'  # Đánh dấu là mô hình đơn giản
            }
            
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"✓ Đã lưu mô hình vào {MODEL_PATH}")
            return True
        
        except Exception as e:
            logger.error(f"Lỗi khi lưu mô hình: {e}")
            return False
    
    def train(self):
        """
        Huấn luyện (thực ra chỉ là load và lưu templates)
        """
        logger.info("=" * 50)
        logger.info("BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH ĐƠN GIẢN")
        logger.info("=" * 50)
        
        # Load dataset
        if not self.load_dataset():
            logger.error("Không thể load dataset")
            return False
        
        # Lưu mô hình
        if not self.save_model():
            logger.error("Không thể lưu mô hình")
            return False
        
        logger.info("=" * 50)
        logger.info(f"✓ HOÀN THÀNH HUẤN LUYỆN")
        logger.info(f"Số người: {len(self.names)}")
        logger.info(f"Danh sách: {', '.join(self.names)}")
        logger.info("=" * 50)
        
        return True


def main():
    """
    Hàm main
    """
    print("=" * 60)
    print("HUẤN LUYỆN MÔ HÌNH ĐƠN GIẢN (KHÔNG CẦN face-recognition)")
    print("=" * 60)
    
    try:
        trainer = SimpleTrainer()
        success = trainer.train()
        
        if success:
            print("\n✓ Huấn luyện thành công!")
            print("\nBạn có thể chạy nhận diện ngay bây giờ.")
        else:
            print("\n✗ Huấn luyện thất bại!")
    
    except Exception as e:
        logger.error(f"Lỗi: {e}")
        print(f"\n✗ Lỗi: {e}")


if __name__ == "__main__":
    main()
