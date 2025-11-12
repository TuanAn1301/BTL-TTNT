"""
Module phát hiện khuôn mặt đơn giản sử dụng OpenCV và Haar Cascade
"""
import os
import cv2
import numpy as np
from typing import List, Tuple, Optional

class SimpleFaceDetector:
    def __init__(self):
        """
        Khởi tạo bộ phát hiện khuôn mặt đơn giản
        """
        # Khởi tạo bộ lọc cascade
        self.face_cascade = cv2.CascadeClassifier()
        
        # Danh sách các file cascade mặc định
        self.cascade_files = [
            # Ưu tiên file cascade từ thư viện OpenCV
            os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'),
            # Hoặc thử đường dẫn trực tiếp
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            # Hoặc tên file đơn giản (nếu nằm cùng thư mục)
            'haarcascade_frontalface_default.xml'
        ]
        
        # Thử tải từng file cascade cho đến khi thành công
        for cascade_file in self.cascade_files:
            try:
                if os.path.exists(cascade_file):
                    self.face_cascade = cv2.CascadeClassifier(cascade_file)
                    if not self.face_cascade.empty():
                        print(f"✓ Đã tải bộ lọc từ: {cascade_file}")
                        break
            except Exception as e:
                print(f"⚠️ Lỗi khi tải {cascade_file}: {str(e)}")
        
        if self.face_cascade.empty():
            raise RuntimeError("Không thể tải bất kỳ bộ lọc khuôn mặt nào")
    
    def detect_faces(
        self,
        image: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30)
    ) -> List[Tuple[int, int, int, int]]:
        """
        Phát hiện khuôn mặt trong ảnh
        
        Args:
            image: Ảnh đầu vào (BGR format)
            scale_factor: Tham số scale factor cho detectMultiScale
            min_neighbors: Số lân cận tối thiểu
            min_size: Kích thước tối thiểu của khuôn mặt
            
        Returns:
            Danh sách các hình chữ nhật (x, y, w, h) bao quanh khuôn mặt
        """
        if image is None or image.size == 0:
            return []
            
        # Chuyển ảnh sang ảnh xám
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện khuôn mặt
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        return [tuple(face) for face in faces]
    
    def draw_faces(self, image, faces, color=(0, 255, 0), thickness=2):
        """
        Vẽ khung bao quanh khuôn mặt
        
        Args:
            image: Ảnh gốc
            faces: Danh sách các khuôn mặt dạng (x, y, w, h)
            color: Màu khung (BGR)
            thickness: Độ dày đường viền
            
        Returns:
            Ảnh đã vẽ khung
        """
        result = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x+w, y+h), color, thickness)
        return result
