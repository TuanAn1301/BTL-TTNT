"""
Thread nhận diện đơn giản - KHÔNG CẦN huấn luyện
Sử dụng template matching trực tiếp từ dataset
"""
import cv2
import os
import numpy as np
from datetime import datetime, timedelta
import time
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from config import (
    DATASET_DIR, ATTENDANCE_COOLDOWN,
    COLOR_GREEN, COLOR_RED, COLOR_WHITE
)


class SimpleRecognitionThread(QThread):
    """
    Thread nhận diện đơn giản - template matching
    """
    frame_ready = pyqtSignal(QImage)
    face_recognized = pyqtSignal(str, float)
    attendance_marked = pyqtSignal(str, str)
    recognition_failed = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, camera_index=0, face_detector=None):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.camera = None
        self.face_detector = face_detector
        self.last_attendance = {}
        self.templates = {}  # Lưu template của mỗi người
        self.required_presence_seconds = 3.0
        self.stable_name = None
        self.stable_since = None
        self.unknown_since = None
        
        # Load templates từ dataset
        self.load_templates()
    
    def load_templates(self):
        """
        Load tất cả ảnh từ dataset làm template - Sử dụng PIL để xử lý Unicode path
        """
        try:
            # Xóa templates cũ
            self.templates.clear()
            
            if not os.path.exists(DATASET_DIR):
                print("⚠️ Thư mục dataset không tồn tại")
                return
            
            persons = [d for d in os.listdir(DATASET_DIR) 
                      if os.path.isdir(os.path.join(DATASET_DIR, d))]
            
            for person_name in persons:
                person_dir = os.path.join(DATASET_DIR, person_name)
                image_files = [f for f in os.listdir(person_dir) 
                              if f.endswith(('.jpg', '.jpeg', '.png'))]
                
                person_templates = []
                for img_file in image_files:
                    img_path = os.path.join(person_dir, img_file)
                    
                    try:
                        # Sử dụng PIL để đọc ảnh (hỗ trợ Unicode path)
                        from PIL import Image
                        pil_img = Image.open(img_path).convert('L')  # Convert to grayscale
                        
                        # Chuyển sang numpy array
                        img = np.array(pil_img)
                        
                        # Resize về kích thước chuẩn
                        img = cv2.resize(img, (100, 100))
                        person_templates.append(img)
                    except Exception as e:
                        print(f"Lỗi đọc ảnh {img_path}: {e}")
                        continue
                
                if person_templates:
                    self.templates[person_name] = person_templates
            
            print(f"✓ Đã load template cho {len(self.templates)} người")
            
            # Log chi tiết
            for name, templates in self.templates.items():
                print(f"  - {name}: {len(templates)} ảnh")
        
        except Exception as e:
            print(f"Lỗi load templates: {e}")
    
    def reload_templates(self):
        """
        Reload lại templates từ dataset (khi có người dùng mới)
        """
        print("\n🔄 Đang reload templates...")
        self.load_templates()
        print("✓ Đã reload templates thành công!")
    
    def recognize_face(self, face_img):
        """
        Nhận diện khuôn mặt bằng template matching
        
        Args:
            face_img: Ảnh khuôn mặt (BGR)
            
        Returns:
            Tuple (name, confidence)
        """
        if not self.templates:
            return None, 0.0
        
        # Chuyển sang grayscale và resize
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (100, 100))
        
        best_match_name = None
        best_match_score = 0.0
        
        # So sánh với từng template
        for person_name, templates in self.templates.items():
            for template in templates:
                # Tính similarity bằng template matching
                result = cv2.matchTemplate(gray_face, template, cv2.TM_CCOEFF_NORMED)
                score = result[0][0]
                
                if score > best_match_score:
                    best_match_score = score
                    best_match_name = person_name
        
        # Ngưỡng nhận diện
        threshold = 0.5
        if best_match_score >= threshold:
            return best_match_name, best_match_score
        else:
            return None, best_match_score
    
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
                
                # Xử lý mỗi 2 frame
                if frame_count % 2 == 0 and self.face_detector:
                    # Phát hiện khuôn mặt
                    faces = self.face_detector.detect_faces(frame)
                    
                    for (x, y, w, h) in faces:
                        try:
                            # Crop khuôn mặt
                            face_img = frame[y:y+h, x:x+w]
                            
                            # Nhận diện
                            name, confidence = self.recognize_face(face_img)
                            
                            if name and confidence >= 0.5:
                                # Nhận diện thành công
                                color = COLOR_GREEN
                                label = f"{name}"
                                conf_text = f"{confidence:.2%}"
                                status = ""
                                
                                # Emit signal liên tục cho UI
                                self.face_recognized.emit(name, confidence)
                                
                                # Yêu cầu đứng ổn định self.required_presence_seconds giây
                                now_ts = time.time()
                                if self.stable_name != name:
                                    self.stable_name = name
                                    self.stable_since = now_ts
                                held = (now_ts - self.stable_since) if self.stable_since else 0.0
                                # Vẽ thời gian giữ yên
                                cv2.putText(display_frame, f"Giu yen: {held:.1f}/{self.required_presence_seconds:.0f}s", (x, max(0, y-70)),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
                                
                                if held >= self.required_presence_seconds:
                                    if self.can_mark_attendance(name):
                                        self.mark_attendance(name)
                                    status = "DIEM DANH"
                            else:
                                # Không nhận diện được
                                color = COLOR_RED
                                label = "Unknown"
                                conf_text = f"{confidence:.2%}"
                                status = ""
                                # Theo dõi ổn định khuôn mặt không nhận diện
                                now_ts = time.time()
                                if self.unknown_since is None:
                                    self.unknown_since = now_ts
                                held_unknown = now_ts - self.unknown_since
                                cv2.putText(display_frame, f"Giu yen: {held_unknown:.1f}/{self.required_presence_seconds:.0f}s", (x, max(0, y-70)),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
                                if held_unknown >= self.required_presence_seconds:
                                    # Thất bại sau 3s không nhận diện được
                                    self.recognition_failed.emit()
                                    # Reset để không spam signal
                                    self.unknown_since = None
                                # Reset ổn định theo tên
                                self.stable_name = None
                                self.stable_since = None
                            
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
            # Dọn dẹp an toàn, không tự wait() trên chính thread
            self.running = False
            if self.camera is not None:
                self.camera.release()
                self.camera = None
    
    def can_mark_attendance(self, name):
        """
        Kiểm tra có thể điểm danh không
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
        save_attendance_record(name, 1.0)
        
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
        # Chỉ quit/wait nếu được gọi từ thread khác
        from PyQt5.QtCore import QThread as _QThread
        if _QThread.currentThread() != self:
            self.quit()
            self.wait()
