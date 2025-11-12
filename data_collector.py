"""
Module thu thập dữ liệu khuôn mặt
"""
import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
import threading
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    IMAGES_PER_PERSON, IMAGE_SIZE,
    HAARCASCADE_PATH,
    FACE_DETECTION_SCALE_FACTOR,
    FACE_DETECTION_MIN_NEIGHBORS,
    FACE_DETECTION_MIN_SIZE,
    COLOR_GREEN, COLOR_RED, COLOR_WHITE, COLOR_BLUE,
    DUPLICATE_SIMILARITY_THRESHOLD
)
from utils.logger import setup_logger
from utils.file_handler import get_person_image_dir, get_all_persons
from utils.helper import load_student_encodings, init_attendance_database
from utils.face_encoding import FaceEncoder

logger = setup_logger(__name__)


class DataCollector:
    """
    Class thu thập dữ liệu khuôn mặt
    """
    
    def __init__(self):
        """
        Khởi tạo DataCollector
        """
        self.face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
        if self.face_cascade.empty():
            raise ValueError(f"Không thể load Haar Cascade từ {HAARCASCADE_PATH}")
        
        # Khởi tạo bộ mã hóa khuôn mặt
        try:
            self.encoder = FaceEncoder()
            logger.info("Đã khởi tạo FaceEncoder thành công")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo FaceEncoder: {str(e)}")
            self.encoder = None
        
        # Tải dữ liệu từ CSDL hiện có
        self._load_existing_data()
        
        # Khởi tạo cờ kiểm soát luồng
        self.duplicate_checking = False
        self.duplicate_found = False
        self.duplicate_name = None
        self.duplicate_similarity = 0.0
        self.face_roi = None
        self.save_anyway = False
        
        logger.info("DataCollector đã được khởi tạo")
    
    def _load_existing_data(self):
        """Tải dữ liệu khuôn mặt hiện có từ CSDL"""
        self._db_names = []
        self._db_encodings = []
        try:
            items = load_student_encodings()
            if items:
                for n, vec in items:
                    self._db_names.append(n)
                    self._db_encodings.append(np.array(vec, dtype=np.float32))
                logger.info(f"Đã tải {len(self._db_names)} mẫu khuôn mặt từ CSDL")
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu từ CSDL: {str(e)}")
    
    def _check_duplicate_faces(self, face_img):
        """
        Kiểm tra xem khuôn mặt có trùng với ai trong CSDL không
        
        Args:
            face_img: Ảnh khuôn mặt cần kiểm tra (BGR)
            
        Returns:
            (bool, str, float): (có trùng không, tên người trùng, độ tương đồng)
        """
        if self.encoder is None:
            return False, None, 0.0
            
        try:
            # Mã hóa khuôn mặt
            encoding = self.encoder.encode_face(face_img)
            if encoding is None:
                return False, None, 0.0
                
            # Chuẩn hóa vector đặc trưng
            encoding = np.array(encoding, dtype=np.float32)
            encoding = encoding / (np.linalg.norm(encoding) + 1e-8)
            
            # So sánh với các mẫu trong CSDL
            max_similarity = 0.0
            best_match = None
            
            for i, db_encoding in enumerate(self._db_encodings):
                if db_encoding is None:
                    continue
                    
                # Tính độ tương đồng cosine
                similarity = float(np.dot(encoding, db_encoding))
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = self._db_names[i] if i < len(self._db_names) else None
            
            # Nếu độ tương đồng > ngưỡng, coi là trùng
            if max_similarity >= DUPLICATE_SIMILARITY_THRESHOLD:
                return True, best_match, max_similarity
                
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra trùng lặp: {str(e)}")
            
        return False, None, 0.0
    
    def _show_duplicate_warning(self):
        """Hiển thị cảnh báo trùng lặp"""
        root = tk.Tk()
        root.withdraw()  # Ẩn cửa sổ chính
        
        message = f"Phát hiện khuôn mặt tương tự với {self.duplicate_name} (độ tương đồng: {self.duplicate_similarity*100:.1f}%)\\n\\nBạn có muốn lưu không?"
        
        if messagebox.askyesno("Cảnh báo trùng lặp", message):
            self.save_anyway = True
        else:
            self.save_anyway = False
            
        root.destroy()
    
    def _compute_phash(self, bgr_img):
        """Tính pHash 64-bit (trả về mảng 64 phần tử 0/1) cho ảnh BGR."""
        try:
            import numpy as np
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
            small = np.float32(small)
            dct = cv2.dct(small)
            dct_low = dct[:8, :8]
            mean = dct_low.mean()
            bits = (dct_low > mean).astype(np.uint8).flatten()
            return bits
        except Exception:
            return None

    def collect_faces(self, person_name):
        """
        Thu thập ảnh khuôn mặt cho một người
        
        Args:
            person_name: Tên người cần thu thập ảnh
            
        Returns:
            bool: True nếu thu thập thành công ít nhất 1 ảnh, False nếu thất bại
        """
        # Tạo thư mục cho người này
        person_dir = get_person_image_dir(person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Mở camera
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        if not cap.isOpened():
            logger.error("Không thể mở camera")
            return False
        
        logger.info(f"Bắt đầu thu thập {IMAGES_PER_PERSON} ảnh cho {person_name}")
        logger.info("Nhấn SPACE để chụp ảnh, ESC để thoát")
        
        count = 0
        self.duplicate_found = False
        self.save_anyway = False
        self.face_roi = None
        
        while count < IMAGES_PER_PERSON:
            ret, frame = cap.read()
            if not ret:
                logger.error("Không thể đọc frame từ camera")
                break
            
            # Tạo bản sao để hiển thị
            display_frame = frame.copy()
            
            # Chuyển sang grayscale để phát hiện khuôn mặt
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Phát hiện khuôn mặt
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION_SCALE_FACTOR,
                minNeighbors=FACE_DETECTION_MIN_NEIGHBORS,
                minSize=FACE_DETECTION_MIN_SIZE
            )
            
            # Vẽ khung cho mỗi khuôn mặt
            for (x, y, w, h) in faces:
                # Lưu lại vùng khuôn mặt để xử lý
                self.face_roi = frame[y:y+h, x:x+w].copy()
                
                # Vẽ khung xung quanh khuôn mặt
                if self.duplicate_found and not self.save_anyway:
                    # Nếu phát hiện trùng, vẽ khung đỏ
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), COLOR_RED, 2)
                    warning_text = f"TRUNG LAP: {self.duplicate_name} ({self.duplicate_similarity*100:.1f}%)"
                    cv2.putText(display_frame, warning_text, 
                               (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
                else:
                    # Nếu không trùng, vẽ khung xanh
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), COLOR_GREEN, 2)
                
                # Hiển thị hướng dẫn
                if not self.duplicate_found or self.save_anyway:
                    cv2.putText(display_frame, f"Nhấn SPACE để chụp ({count}/{IMAGES_PER_PERSON})",
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 2)
            
            # Hiển thị thông tin
            status_text = "TRUNG LAP! Nhấn 'Y' để lưu, 'N' để bỏ qua" if self.duplicate_found else f"Đã chụp: {count}/{IMAGES_PER_PERSON}"
            cv2.putText(display_frame, status_text,
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
            
            cv2.putText(display_frame, "SPACE: Chụp | ESC: Thoát",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
            
            # Hiển thị frame
            cv2.imshow(f'Thu thập dữ liệu - {person_name}', display_frame)
            
            # Xử lý phím bấm
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                logger.info("Người dùng hủy thu thập dữ liệu")
                break
            elif key == 32:  # SPACE
                if len(faces) > 0 and (not self.duplicate_found or self.save_anyway):
                    # Lưu ảnh
                    face_img = self.face_roi
                    if face_img is not None:
                        # Resize về kích thước chuẩn
                        face_img = cv2.resize(face_img, IMAGE_SIZE)
                        
                        # Lưu ảnh
                        img_path = os.path.join(person_dir, f"{person_name}_{count}.jpg")
                        cv2.imwrite(img_path, face_img)
                        logger.info(f"Đã lưu ảnh {img_path}")
                        
                        count += 1
                        
                        # Nếu đã lưu 1 ảnh, đặt lại trạng thái kiểm tra
                        if self.duplicate_found:
                            self.duplicate_found = False
                            self.save_anyway = False
            elif key == ord('y') and self.duplicate_found:
                # Người dùng chọn lưu dù bị trùng
                self.save_anyway = True
            elif key == ord('n') and self.duplicate_found:
                # Người dùng chọn không lưu
                self.duplicate_found = False
                self.save_anyway = False
            
            # Kiểm tra trùng lặp nếu chưa kiểm tra và có khuôn mặt
            if (not self.duplicate_checking and len(faces) > 0 and 
                not self.duplicate_found and not self.save_anyway and 
                self.face_roi is not None):
                self.duplicate_checking = True
                threading.Thread(target=self._check_duplicate_thread, 
                                args=(self.face_roi.copy(),)).start()
        
        # Giải phóng tài nguyên
        cap.release()
        cv2.destroyAllWindows()
        
        logger.info(f"Hoàn thành thu thập {count} ảnh cho {person_name}")
        return count > 0
    
    def _check_duplicate_thread(self, face_img):
        """Luồng kiểm tra trùng lặp"""
        try:
            is_duplicate, name, similarity = self._check_duplicate_faces(face_img)
            if is_duplicate:
                self.duplicate_found = True
                self.duplicate_name = name
                self.duplicate_similarity = similarity
                # Hiển thị cảnh báo
                self._show_duplicate_warning()
        except Exception as e:
            logger.error(f"Lỗi trong luồng kiểm tra trùng lặp: {str(e)}")
        finally:
            self.duplicate_checking = False


def main():
    """
    Hàm main để chạy thu thập dữ liệu
    """
    # Khởi tạo CSDL nếu chưa có
    init_attendance_database()
    
    # Nhập tên người cần thu thập
    person_name = input("Nhập tên người cần thu thập: ").strip()
    if not person_name:
        print("Tên không hợp lệ!")
        return
    
    # Khởi tạo bộ thu thập
    try:
        collector = DataCollector()
    except Exception as e:
        print(f"Lỗi khởi tạo bộ thu thập: {str(e)}")
        return
    
    # Bắt đầu thu thập
    print(f"\nBắt đầu thu thập dữ liệu cho {person_name}")
    print("Nhấn SPACE để chụp ảnh")
    print("Nhấn ESC để thoát")
    print("Khi phát hiện trùng lặp, nhấn 'Y' để lưu, 'N' để bỏ qua\n")
    
    success = collector.collect_faces(person_name)
    
    if success:
        print(f"\nĐã hoàn thành thu thập dữ liệu cho {person_name}")
    else:
        print("\nThu thập dữ liệu không thành công")


if __name__ == "__main__":
    main()
