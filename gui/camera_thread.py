"""
Thread xử lý camera cho PyQt5
"""
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage


class CameraThread(QThread):
    """
    Thread xử lý camera
    """
    # Signal để gửi frame
    frame_ready = pyqtSignal(QImage)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.camera = None
    
    def run(self):
        """
        Chạy thread camera
        """
        try:
            # Mở camera
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                self.error_occurred.emit("Không thể mở camera. Vui lòng kiểm tra kết nối camera.")
                return
            
            # Cấu hình camera
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.running = True
            
            while self.running:
                ret, frame = self.camera.read()
                
                if not ret:
                    self.error_occurred.emit("Không thể đọc frame từ camera")
                    break
                
                # Chuyển đổi frame sang QImage
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # Gửi signal
                self.frame_ready.emit(qt_image)
                
                # Delay nhỏ để giảm CPU usage
                self.msleep(30)
        
        except Exception as e:
            self.error_occurred.emit(f"Lỗi camera: {str(e)}")
        
        finally:
            # Dọn dẹp an toàn, không tự wait() trên chính thread
            self.running = False
            if self.camera is not None:
                self.camera.release()
                self.camera = None
    
    def stop(self):
        """
        Dừng camera
        """
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        from PyQt5.QtCore import QThread as _QThread
        if _QThread.currentThread() != self:
            self.quit()
            self.wait()
    
    def get_current_frame(self):
        """
        Lấy frame hiện tại (BGR format)
        """
        if self.camera is not None and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                return frame
        return None


class CollectionCameraThread(QThread):
    """
    Thread xử lý camera cho thu thập dữ liệu
    """
    frame_ready = pyqtSignal(QImage, object)  # QImage và frame BGR
    error_occurred = pyqtSignal(str)
    face_detected = pyqtSignal(int)  # Số lượng khuôn mặt phát hiện
    auto_capture = pyqtSignal(object)  # Signal tự động chụp ảnh
    
    def __init__(self, camera_index=0, face_detector=None, auto_capture_mode=True):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.camera = None
        self.face_detector = face_detector
        self.auto_capture_mode = auto_capture_mode  # Chế độ tự động chụp
        self.last_capture_time = 0  # Thời gian chụp ảnh cuối
        self.capture_delay = 1.0  # Delay 1 giây giữa các lần chụp
        self.required_presence_seconds = 3.0
        self.face_present_since = None
        self.paused = False
    
    def run(self):
        """
        Chạy thread camera với phát hiện khuôn mặt
        """
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                self.error_occurred.emit("Không thể mở camera")
                return
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.running = True
            
            import time
            
            while self.running:
                if self.paused:
                    self.msleep(100)
                    continue
                ret, frame = self.camera.read()
                
                if not ret:
                    break
                
                # Phát hiện khuôn mặt nếu có detector
                display_frame = frame.copy()
                num_faces = 0
                
                if self.face_detector:
                    faces = self.face_detector.detect_faces(frame)
                    num_faces = len(faces)
                    
                    # Vẽ khung
                    for (x, y, w, h) in faces:
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(display_frame, "Face Detected", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    self.face_detected.emit(num_faces)
                    
                    # Yêu cầu đứng trước camera liên tục self.required_presence_seconds giây
                    current_time = time.time()
                    if num_faces > 0:
                        if self.face_present_since is None:
                            self.face_present_since = current_time
                        held = current_time - self.face_present_since
                        remaining = max(0.0, self.required_presence_seconds - held)
                        cv2.putText(display_frame, f"Giu yen: {held:.1f}/{self.required_presence_seconds:.0f}s", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        if self.auto_capture_mode and held >= self.required_presence_seconds:
                            if current_time - self.last_capture_time >= self.capture_delay:
                                self.last_capture_time = current_time
                                self.auto_capture.emit(frame)
                                self.face_present_since = None
                    else:
                        self.face_present_since = None
                
                # Chuyển sang QImage
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # Gửi cả QImage và frame gốc
                self.frame_ready.emit(qt_image, frame)
                
                self.msleep(30)
        
        except Exception as e:
            self.error_occurred.emit(f"Lỗi: {str(e)}")
        
        finally:
            # Dọn dẹp an toàn, không tự wait() trên chính thread
            self.running = False
            if self.camera is not None:
                self.camera.release()
                self.camera = None
    
    def stop(self):
        """
        Dừng camera
        """
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        from PyQt5.QtCore import QThread as _QThread
        if _QThread.currentThread() != self:
            self.quit()
            self.wait()

    def pause(self):
        """
        Tạm dừng luồng đọc camera
        """
        self.paused = True

    def resume(self):
        """
        Tiếp tục luồng đọc camera
        """
        self.paused = False
