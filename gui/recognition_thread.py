"""
Thread xử lý nhận diện khuôn mặt và điểm danh
"""
import cv2
import pickle
import os
from datetime import datetime, timedelta
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from config import (
    MODEL_PATH, RECOGNITION_THRESHOLD, ATTENDANCE_COOLDOWN,
    COLOR_GREEN, COLOR_RED, COLOR_WHITE
)


class RecognitionThread(QThread):
    """
    Thread nhận diện khuôn mặt và điểm danh
    """
    frame_ready = pyqtSignal(QImage)
    face_recognized = pyqtSignal(str, float)  # name, confidence
    attendance_marked = pyqtSignal(str, str)  # name, time
    error_occurred = pyqtSignal(str)
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.camera = None
        self.model = None
        self.label_encoder = None
        self.face_detector = None
        self.face_encoder = None
        self.last_attendance = {}
        
        # Load mô hình
        self.load_model()
    
    def load_model(self):
        """
        Load mô hình nhận diện
        """
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError("Chưa có mô hình. Vui lòng huấn luyện trước!")
            
            with open(MODEL_PATH, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data['model']
            self.label_encoder = data['label_encoder']
            
            # Import face detector và encoder
            from utils import FaceDetector, FaceEncoder
            self.face_detector = FaceDetector()
            self.face_encoder = FaceEncoder()
            
            print(f"✓ Đã load mô hình với {len(self.label_encoder.classes_)} người")
        
        except Exception as e:
            self.error_occurred.emit(f"Lỗi load mô hình: {str(e)}")
            raise
    
    def reload_model(self):
        """
        Reload lại mô hình (khi có người dùng mới và đã huấn luyện lại)
        """
        print("\n🔄 Đang reload mô hình...")
        try:
            self.load_model()
            print("✓ Đã reload mô hình thành công!")
        except Exception as e:
            print(f"✗ Lỗi reload mô hình: {e}")
    
    def run(self):
        """
        Chạy thread nhận diện
        """
        try:
            # Mở camera
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                self.error_occurred.emit("Không thể mở camera")
                return
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.running = True
            frame_count = 0
            
            while self.running:
                ret, frame = self.camera.read()
                
                if not ret:
                    break
                
                frame_count += 1
                display_frame = frame.copy()
                
                # Xử lý mỗi 2 frame để tăng tốc độ
                if frame_count % 2 == 0:
                    # Phát hiện khuôn mặt
                    faces = self.face_detector.detect_faces(frame)
                    
                    # Chuyển sang RGB cho face_recognition
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    for (x, y, w, h) in faces:
                        try:
                            # Crop khuôn mặt
                            face_img = rgb_frame[y:y+h, x:x+w]
                            
                            # Tạo encoding (đặc trưng mắt, mũi, miệng)
                            encoding = self.face_encoder.encode_face(face_img)
                            
                            if encoding is not None:
                                # Nhận diện
                                name, confidence = self.recognize_face(encoding)
                                
                                if name and confidence >= RECOGNITION_THRESHOLD:
                                    # Nhận diện thành công
                                    color = COLOR_GREEN
                                    label = f"{name}"
                                    conf_text = f"{confidence:.2%}"
                                    
                                    # Emit signal
                                    self.face_recognized.emit(name, confidence)
                                    
                                    # Thử điểm danh
                                    if self.can_mark_attendance(name):
                                        self.mark_attendance(name)
                                        status = "DIEM DANH"
                                    else:
                                        status = "Cooldown"
                                else:
                                    # Không nhận diện được
                                    color = COLOR_RED
                                    label = "Unknown"
                                    conf_text = f"{confidence:.2%}"
                                    status = ""
                                
                                # Vẽ khung và text
                                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                                
                                # Background cho text
                                cv2.rectangle(display_frame, (x, y-60), (x+w, y), color, -1)
                                
                                # Text
                                cv2.putText(display_frame, label, (x+5, y-35),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
                                cv2.putText(display_frame, conf_text, (x+5, y-15),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
                                if status:
                                    cv2.putText(display_frame, status, (x+5, y-5),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)
                            else:
                                # Không tạo được encoding
                                cv2.rectangle(display_frame, (x, y), (x+w, y+h), COLOR_RED, 2)
                        
                        except Exception as e:
                            print(f"Lỗi xử lý khuôn mặt: {e}")
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), COLOR_RED, 2)
                
                # Chuyển sang QImage
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                self.frame_ready.emit(qt_image)
                
                self.msleep(30)
        
        except Exception as e:
            self.error_occurred.emit(f"Lỗi: {str(e)}")
        
        finally:
            self.stop()
    
    def recognize_face(self, face_encoding):
        """
        Nhận diện khuôn mặt từ encoding
        
        Args:
            face_encoding: Face encoding (128-d vector chứa đặc trưng mắt, mũi, miệng)
            
        Returns:
            Tuple (name, confidence)
        """
        try:
            # Dự đoán
            probabilities = self.model.predict_proba([face_encoding])[0]
            max_prob_idx = probabilities.argmax()
            max_prob = probabilities[max_prob_idx]
            
            if max_prob >= RECOGNITION_THRESHOLD:
                name = self.label_encoder.inverse_transform([max_prob_idx])[0]
                return name, max_prob
            else:
                return None, max_prob
        
        except Exception as e:
            print(f"Lỗi nhận diện: {e}")
            return None, 0.0
    
    def can_mark_attendance(self, name):
        """
        Kiểm tra có thể điểm danh không (cooldown)
        """
        now = datetime.now()
        
        if name not in self.last_attendance:
            return True
        
        time_diff = (now - self.last_attendance[name]).total_seconds()
        return time_diff >= ATTENDANCE_COOLDOWN
    
    def mark_attendance(self, name):
        """
        Đánh dấu điểm danh
        """
        from utils import save_attendance_record
        
        now = datetime.now()
        time_str = now.strftime('%H:%M:%S')
        
        # Lưu điểm danh
        save_attendance_record(name, 1.0)  # confidence = 1.0 cho điểm danh thành công
        
        # Cập nhật thời gian
        self.last_attendance[name] = now
        
        # Emit signal
        self.attendance_marked.emit(name, time_str)
        
        print(f"✓ Đã điểm danh cho {name} lúc {time_str}")
    
    def stop(self):
        """
        Dừng thread
        """
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.quit()
        self.wait()
