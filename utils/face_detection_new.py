"""
Module phát hiện khuôn mặt sử dụng OpenCV và Haar Cascade
"""
import cv2
import numpy as np
import os
import tempfile
import shutil
from config import (
    HAARCASCADE_PATH,
    FACE_DETECTION_SCALE_FACTOR,
    FACE_DETECTION_MIN_NEIGHBORS,
    FACE_DETECTION_MIN_SIZE
)


class FaceDetector:
    """
    Class phát hiện khuôn mặt sử dụng Haar Cascade
    """
    
    def __init__(self, cascade_path=None):
        """
        Khởi tạo Face Detector
        
        Args:
            cascade_path: Đường dẫn đến file Haar Cascade (optional)
        """
        if cascade_path is None:
            cascade_path = HAARCASCADE_PATH
        
        # Kiểm tra xem file có tồn tại không
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Không tìm thấy file Haar Cascade tại: {cascade_path}")
        
        # Sử dụng file tạm để tránh lỗi Unicode path
        try:
            # Tạo thư mục tạm nếu chưa tồn tại
            temp_dir = os.path.join(tempfile.gettempdir(), "face_attendance_temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Tạo đường dẫn file tạm
            temp_file = os.path.join(temp_dir, "haarcascade_frontalface_default.xml")
            
            # Sao chép file cascade vào thư mục tạm nếu chưa có hoặc file nguồn mới hơn
            if not os.path.exists(temp_file) or \
               os.path.getmtime(cascade_path) > os.path.getmtime(temp_file):
                shutil.copy2(cascade_path, temp_file)
            
            # Load cascade từ file tạm
            self.face_cascade = cv2.CascadeClassifier(temp_file)
            
            # Kiểm tra xem cascade có được load thành công không
            if self.face_cascade.empty():
                # Thử load lại bằng cách đọc nội dung file
                with open(cascade_path, 'rb') as f:
                    cascade_data = f.read()
                
                # Ghi lại vào file tạm
                with open(temp_file, 'wb') as f:
                    f.write(cascade_data)
                
                # Thử load lại
                self.face_cascade = cv2.CascadeClassifier(temp_file)
                
                if self.face_cascade.empty():
                    raise RuntimeError("Không thể tải bộ phát hiện khuôn mặt")
            
            print("✓ Đã load Haar Cascade thành công qua file tạm")
            self.temp_cascade_path = temp_file
            
        except Exception as e:
            print(f"⚠️ Lỗi khi tải bộ phát hiện khuôn mặt: {str(e)}")
            # Thử load từ đường dẫn gốc như một giải pháp cuối cùng
            try:
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if self.face_cascade.empty():
                    raise RuntimeError("Không thể tải bộ phát hiện khuôn mặt")
                print("✓ Đã load Haar Cascade thành công từ đường dẫn gốc")
                self.temp_cascade_path = None
            except Exception as e2:
                raise RuntimeError(f"Không thể tải bộ phát hiện khuôn mặt: {str(e2)}")
    
    def __del__(self):
        """
        Dọn dẹp file tạm khi object bị xóa
        """
        if hasattr(self, 'temp_cascade_path') and self.temp_cascade_path:
            try:
                if os.path.exists(self.temp_cascade_path):
                    os.unlink(self.temp_cascade_path)
            except:
                pass
    
    def detect_faces(self, image, scale_factor=None, min_neighbors=None, min_size=None):
        """
        Phát hiện khuôn mặt trong ảnh
        
        Args:
            image: Ảnh đầu vào (BGR hoặc grayscale)
            scale_factor: Tỷ lệ scale (optional)
            min_neighbors: Số láng giềng tối thiểu (optional)
            min_size: Kích thước tối thiểu (optional)
            
        Returns:
            List các tuple (x, y, w, h) của khuôn mặt
        """
        # Sử dụng giá trị mặc định nếu không được cung cấp
        if scale_factor is None:
            scale_factor = FACE_DETECTION_SCALE_FACTOR
        if min_neighbors is None:
            min_neighbors = FACE_DETECTION_MIN_NEIGHBORS
        if min_size is None:
            min_size = FACE_DETECTION_MIN_SIZE
        
        # Chuyển sang grayscale nếu cần
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Phát hiện khuôn mặt
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        return faces
    
    def draw_faces(self, image, faces, color=(0, 255, 0), thickness=2):
        """
        Vẽ khung bao quanh khuôn mặt
        
        Args:
            image: Ảnh đầu vào
            faces: List các tuple (x, y, w, h)
            color: Màu khung (BGR)
            thickness: Độ dày khung
            
        Returns:
            Ảnh đã vẽ khung
        """
        result = image.copy()
        
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x+w, y+h), color, thickness)
        
        return result
    
    def extract_face(self, image, face_rect, padding=0):
        """
        Trích xuất vùng khuôn mặt từ ảnh
        
        Args:
            image: Ảnh đầu vào
            face_rect: Tuple (x, y, w, h) của khuôn mặt
            padding: Padding thêm xung quanh khuôn mặt (pixels)
            
        Returns:
            Ảnh khuôn mặt đã trích xuất
        """
        x, y, w, h = face_rect
        
        # Thêm padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Trích xuất vùng khuôn mặt
        face = image[y:y+h, x:x+w]
        
        return face
