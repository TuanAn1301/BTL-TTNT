"""
Module nhận diện khuôn mặt và điểm danh
"""
import cv2
import pickle
import numpy as np
import face_recognition
from datetime import datetime, timedelta
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    MODEL_PATH, HAARCASCADE_PATH,
    FACE_DETECTION_SCALE_FACTOR,
    FACE_DETECTION_MIN_NEIGHBORS,
    FACE_DETECTION_MIN_SIZE,
    RECOGNITION_THRESHOLD,
    FACE_ENCODING_MODEL,
    ATTENDANCE_COOLDOWN,
    COLOR_GREEN, COLOR_RED, COLOR_WHITE
)
from utils.logger import setup_logger
from utils.helper import save_attendance_record

logger = setup_logger(__name__)


class FaceRecognitionSystem:
    """
    Class hệ thống nhận diện khuôn mặt và điểm danh
    """
    
    def __init__(self):
        """
        Khởi tạo hệ thống nhận diện
        """
        # Load Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
        if self.face_cascade.empty():
            raise ValueError(f"Không thể load Haar Cascade từ {HAARCASCADE_PATH}")
        
        # Load mô hình
        self.load_model()
        
        # Dictionary lưu thời gian điểm danh gần nhất
        self.last_attendance = {}
        
        logger.info("FaceRecognitionSystem đã được khởi tạo")
    
    def load_model(self):
        """
        Load mô hình đã huấn luyện
        """
        try:
            with open(MODEL_PATH, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data['model']
            self.label_encoder = data['label_encoder']
            
            logger.info(f"Đã load mô hình từ {MODEL_PATH}")
            logger.info(f"Số lượng người trong mô hình: {len(self.label_encoder.classes_)}")
        
        except FileNotFoundError:
            logger.error(f"Không tìm thấy file mô hình: {MODEL_PATH}")
            raise
        except Exception as e:
            logger.error(f"Lỗi khi load mô hình: {str(e)}")
            raise
    
    def reload_model(self):
        """
        Reload lại mô hình (khi có người dùng mới và đã huấn luyện lại)
        """
        logger.info("Đang reload mô hình...")
        try:
            self.load_model()
            logger.info("Đã reload mô hình thành công!")
            print("\n✓ Đã reload mô hình thành công!")
        except Exception as e:
            logger.error(f"Lỗi reload mô hình: {str(e)}")
            print(f"\n✗ Lỗi reload mô hình: {str(e)}")
    
    def recognize_face(self, face_encoding):
        """
        Nhận diện khuôn mặt từ encoding
        
        Args:
            face_encoding: Face encoding cần nhận diện
            
        Returns:
            Tuple (name, confidence) hoặc (None, 0) nếu không nhận diện được
        """
        # Dự đoán
        probabilities = self.model.predict_proba([face_encoding])[0]
        max_prob_idx = np.argmax(probabilities)
        max_prob = probabilities[max_prob_idx]
        
        # Kiểm tra ngưỡng
        if max_prob >= RECOGNITION_THRESHOLD:
            name = self.label_encoder.inverse_transform([max_prob_idx])[0]
            return name, max_prob
        else:
            return None, max_prob
    
    def can_mark_attendance(self, name):
        """
        Kiểm tra xem có thể điểm danh cho người này không
        (dựa trên cooldown time)
        
        Args:
            name: Tên người
            
        Returns:
            True nếu có thể điểm danh, False nếu không
        """
        now = datetime.now()
        
        if name not in self.last_attendance:
            return True
        
        time_diff = (now - self.last_attendance[name]).total_seconds()
        return time_diff >= ATTENDANCE_COOLDOWN
    
    def mark_attendance(self, name, confidence):
        """
        Đánh dấu điểm danh cho người
        
        Args:
            name: Tên người
            confidence: Độ tin cậy
        """
        if self.can_mark_attendance(name):
            save_attendance_record(name, confidence)
            self.last_attendance[name] = datetime.now()
            logger.info(f"Đã điểm danh cho {name} với confidence {confidence:.2f}")
            return True
        else:
            logger.debug(f"Chưa đủ thởi gian cooldown cho {name}")
            return False
    
    def run(self):
        """
        Chạy hệ thống nhận diện và điểm danh
        """
        # Mở camera
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        if not cap.isOpened():
            logger.error("Không thể mở camera")
            return
        
        logger.info("Bắt đầu hệ thống nhận diện khuôn mặt")
        logger.info("Nhấn 'q' hoặc ESC để thoát")
        
        frame_count = 0
        process_every_n_frames = 2  # Xử lý mỗi 2 frame để tăng tốc độ
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Không thể đọc frame từ camera")
                break
            
            frame_count += 1
            
            # Chỉ xử lý mỗi n frames
            if frame_count % process_every_n_frames != 0:
                cv2.imshow('He thong diem danh khuon mat', frame)
                if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                    break
                continue
            
            # Chuyển sang grayscale để phát hiện khuôn mặt
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Phát hiện khuôn mặt
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION_SCALE_FACTOR,
                minNeighbors=FACE_DETECTION_MIN_NEIGHBORS,
                minSize=FACE_DETECTION_MIN_SIZE
            )
            
            # Chuyển sang RGB cho face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Xử lý từng khuôn mặt
            for (x, y, w, h) in faces:
                # Crop khuôn mặt
                face_img = rgb_frame[y:y+h, x:x+w]
                
                try:
                    # Tạo encoding
                    encodings = face_recognition.face_encodings(
                        face_img,
                        model=FACE_ENCODING_MODEL
                    )
                    
                    if len(encodings) > 0:
                        encoding = encodings[0]
                        
                        # Nhận diện
                        name, confidence = self.recognize_face(encoding)
                        
                        if name:
                            # Nhận diện thành công
                            color = COLOR_GREEN
                            label = f"{name} ({confidence:.2f})"
                            
                            # Thử điểm danh
                            marked = self.mark_attendance(name, confidence)
                            if marked:
                                status = "Da diem danh"
                            else:
                                status = "Cooldown"
                        else:
                            # Không nhận diện được
                            color = COLOR_RED
                            label = f"Unknown ({confidence:.2f})"
                            status = ""
                        
                        # Vẽ khung và text
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, label, (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        if status:
                            cv2.putText(frame, status, (x, y+h+20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        # Không tạo được encoding
                        cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR_RED, 2)
                        cv2.putText(frame, "No encoding", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)
                
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý khuôn mặt: {str(e)}")
                    cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR_RED, 2)
            
            # Hiển thị hướng dẫn
            cv2.putText(frame, "Nhan 'q'/ESC de thoat | 'r' de reload",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
            
            # Hiển thị frame
            cv2.imshow('He thong diem danh khuon mat', frame)
            
            # Kiểm tra phím bấm
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]:  # 'q' hoặc ESC
                logger.info("Người dùng thoát hệ thống")
                break
            elif key == ord('r'):  # 'r' để reload mô hình
                logger.info("Người dùng yêu cầu reload mô hình")
                self.reload_model()
        
        # Giải phóng tài nguyên
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Đã dừng hệ thống nhận diện")


def main():
    """
    Hàm main để chạy hệ thống
    """
    print("=" * 50)
    print("HỆ THỐNG ĐIỂM DANH KHUÔN MẶT")
    print("=" * 50)
    
    try:
        system = FaceRecognitionSystem()
        system.run()
    except Exception as e:
        logger.error(f"Lỗi: {str(e)}")
        print(f"\n✗ Lỗi: {str(e)}")


if __name__ == "__main__":
    main()
