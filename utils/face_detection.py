"""
Module phát hiện khuôn mặt sử dụng OpenCV và Haar Cascade
"""
import os
import cv2
import numpy as np
import tempfile
import shutil
from typing import List, Tuple, Optional, Union
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
    
    def __init__(self, cascade_path: str = None):
        """
        Khởi tạo bộ phát hiện khuôn mặt với bộ lọc cascade và các bộ tiền xử lý ảnh.
        
        Args:
            cascade_path: Đường dẫn đến file cascade cho phát hiện khuôn mặt.
                         Mặc định sử dụng bộ lọc khuôn mặt chính diện của OpenCV.
        """
        # Khởi tạo bộ lọc cascade
        self.face_cascade = cv2.CascadeClassifier()
        self.profile_cascade = cv2.CascadeClassifier()
        self.temp_dir = None
        
        try:
            # Danh sách các file cascade mặc định
            default_cascades = {
                'front': 'haarcascade_frontalface_default.xml',
                'profile': 'haarcascade_profileface.xml'
            }
            
            # Thử tải cascade từ dữ liệu nhúng trong thư viện OpenCV
            try:
                import pkg_resources
                # Tải bộ lọc khuôn mặt chính diện
                front_data = pkg_resources.resource_string('cv2', f'data/{default_cascades["front"]}')
                # Tạo thư mục tạm để lưu file cascade
                self.temp_dir = tempfile.mkdtemp()
                front_path = os.path.join(self.temp_dir, default_cascades['front'])
                
                with open(front_path, 'wb') as f:
                    f.write(front_data)
                
                self.face_cascade = cv2.CascadeClassifier(front_path)
                if not self.face_cascade.empty():
                    print("✓ Đã tải bộ lọc khuôn mặt chính diện từ dữ liệu nhúng")
                
                # Tải bộ lọc khuôn mặt nghiêng nếu có
                try:
                    profile_data = pkg_resources.resource_string('cv2', f'data/{default_cascades["profile"]}')
                    profile_path = os.path.join(self.temp_dir, default_cascades['profile'])
                    with open(profile_path, 'wb') as f:
                        f.write(profile_data)
                    
                    self.profile_cascade = cv2.CascadeClassifier(profile_path)
                    if not self.profile_cascade.empty():
                        print("✓ Đã tải bộ lọc khuôn mặt nghiêng từ dữ liệu nhúng")
                except Exception as e:
                    print(f"⚠️ Không thể tải bộ lọc khuôn mặt nghiêng: {str(e)}")
                    self.profile_cascade = None
            
            except Exception as e:
                print(f"⚠️ Không thể tải từ dữ liệu nhúng: {str(e)}")
                # Thử tải từ đường dẫn mặc định của OpenCV nếu tải từ dữ liệu nhúng thất bại
                try:
                    front_path = os.path.join(cv2.data.haarcascades, default_cascades['front'])
                    if os.path.exists(front_path):
                        self.face_cascade = cv2.CascadeClassifier(front_path)
                        if not self.face_cascade.empty():
                            print("✓ Đã tải bộ lọc khuôn mặt chính diện từ đường dẫn mặc định")
                    
                    profile_path = os.path.join(cv2.data.haarcascades, default_cascades['profile'])
                    if os.path.exists(profile_path):
                        self.profile_cascade = cv2.CascadeClassifier(profile_path)
                        if not self.profile_cascade.empty():
                            print("✓ Đã tải bộ lọc khuôn mặt nghiêng từ đường dẫn mặc định")
                except Exception as e2:
                    print(f"⚠️ Lỗi khi tải từ đường dẫn mặc định: {str(e2)}")
            
            # Nếu vẫn không tải được, thử tải từ đường dẫn tương đối
            if self.face_cascade.empty():
                try:
                    # Kiểm tra trong thư mục haarcascades
                    haarcascades_dir = os.path.join(os.path.dirname(__file__), '..', 'haarcascades')
                    if os.path.exists(haarcascades_dir):
                        front_path = os.path.join(haarcascades_dir, default_cascades['front'])
                        if os.path.exists(front_path):
                            self.face_cascade = cv2.CascadeClassifier(front_path)
                            if not self.face_cascade.empty():
                                print("✓ Đã tải bộ lọc khuôn mặt chính diện từ thư mục haarcascades")
                        
                        profile_path = os.path.join(haarcascades_dir, default_cascades['profile'])
                        if os.path.exists(profile_path):
                            self.profile_cascade = cv2.CascadeClassifier(profile_path)
                            if not self.profile_cascade.empty():
                                print("✓ Đã tải bộ lọc khuôn mặt nghiêng từ thư mục haarcascades")
                except Exception as e3:
                    print(f"⚠️ Lỗi khi tải từ thư mục haarcascades: {str(e3)}")
            
            # Nếu vẫn không tải được, sử dụng đường dẫn cung cấp
            if self.face_cascade.empty() and cascade_path:
                try:
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                    if not self.face_cascade.empty():
                        print(f"✓ Đã tải bộ lọc từ đường dẫn cung cấp: {cascade_path}")
                except Exception as e4:
                    print(f"⚠️ Lỗi khi tải từ đường dẫn cung cấp: {str(e4)}")
            
            # Kiểm tra cuối cùng
            if self.face_cascade.empty():
                raise RuntimeError("Không thể tải bất kỳ bộ lọc khuôn mặt nào. Vui lòng kiểm tra đường dẫn hoặc cài đặt OpenCV đầy đủ.")
                
        except Exception as e:
            # Dọn dẹp thư mục tạm nếu có lỗi
            if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir)
                except Exception as e_cleanup:
                    print(f"⚠️ Lỗi khi dọn dẹp thư mục tạm: {str(e_cleanup)}")
            raise

        # Kiểm tra xem có ít nhất một bộ lọc đã được tải thành công không
        if self.face_cascade.empty():
            # Thử load từ đường dẫn gốc như một giải pháp cuối cùng
            try:
                if cascade_path:
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                    if self.face_cascade.empty():
                        raise RuntimeError("Không thể tải bộ phát hiện khuôn mặt")
                    print("✓ Đã load Haar Cascade thành công từ đường dẫn gốc")
                    self.temp_cascade_path = None
                else:
                    raise RuntimeError("Không có đường dẫn cascade được cung cấp")
            except Exception as e2:
                raise RuntimeError(f"Không thể tải bộ phát hiện khuôn mặt: {str(e2)}")
            
            # Kiểm tra cuối cùng
            if self.face_cascade.empty():
                raise ValueError(f"Không thể load Haar Cascade từ {cascade_path}")
        
        print("✓ Đã khởi tạo xong bộ phát hiện khuôn mặt")
    
    def __del__(self):
        """
        Dọn dẹp file tạm khi object bị xóa
        """
        # Dọn dẹp thư mục tạm nếu có
        if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"⚠️ Lỗi khi dọn dẹp thư mục tạm: {str(e)}")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ảnh để cải thiện khả năng nhận dạng khuôn mặt.
        
        Args:
            image: Ảnh đầu vào (BGR format).
            
        Returns:
            Ảnh đã được xử lý (grayscale).
        """
        if image is None or image.size == 0:
            return None
            
        # 1. Chuyển sang ảnh xám
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 2. Cân bằng sáng (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        
        # 3. Giảm nhiễu bằng bộ lọc song phương (Bilateral Filter)
        denoised = cv2.bilateralFilter(equalized, 9, 75, 75)
        
        # 4. Làm sắc nét ảnh (Sharpening)
        kernel = np.array([[-1,-1,-1], 
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # 5. Chuẩn hóa ảnh
        normalized = cv2.normalize(sharpened, None, alpha=0, beta=255, 
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return normalized
        
    def _remove_overlapping_faces(self, faces: List[Tuple[int, int, int, int]], 
                                overlap_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Loại bỏ các khuôn mặt bị trùng lặp dựa trên tỷ lệ chồng chéo.
        
        Args:
            faces: Danh sách các khuôn mặt dạng (x, y, w, h)
            overlap_threshold: Ngưỡng chồng chéo tối đa để coi là trùng lặp
            
        Returns:
            Danh sách các khuôn mặt không trùng lặp
        """
        if len(faces) == 0:
            return []
            
        # Chuyển đổi sang định dạng (x1, y1, x2, y2) để dễ tính toán
        boxes = np.array([[x, y, x+w, y+h] for (x, y, w, h) in faces])
        
        # Tính diện tích của các hộp giới hạn
        area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        
        # Sắp xếp các hộp theo thứ tự tăng dần diện tích
        idxs = np.argsort(area)
        
        # Danh sách lưu trữ các chỉ số của các hộp được giữ lại
        keep = []
        
        while len(idxs) > 0:
            # Lấy chỉ số của hộp có diện tích nhỏ nhất
            last = len(idxs) - 1
            i = idxs[last]
            keep.append(i)
            
            # Tìm phần giao nhau giữa hộp hiện tại và tất cả các hộp còn lại
            xx1 = np.maximum(boxes[i, 0], boxes[idxs[:last], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[idxs[:last], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[idxs[:last], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[idxs[:last], 3])
            
            # Tính chiều rộng và chiều cao của các phần giao nhau
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            # Tính tỷ lệ chồng chéo
            overlap = (w * h) / area[idxs[:last]]
            
            # Xóa các chỉ số của các hộp có tỷ lệ chồng chéo lớn hơn ngưỡng
            idxs = np.delete(idxs, np.concatenate(([last],
                                                 np.where(overlap > overlap_threshold)[0])))
        
        # Trả về các khuôn mặt không trùng lặp
        return [faces[i] for i in keep]

    def detect_faces(
        self,
        image: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
        max_size: Tuple[int, int] = None,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Phát hiện khuôn mặt trong ảnh với xử lý ảnh nâng cao.

        Args:
            image: Ảnh đầu vào (BGR format).
            scale_factor: Tham số xác định kích thước ảnh được thu nhỏ mỗi lần quét.
            min_neighbors: Số lân cận tối thiểu để giữ lại một cửa sổ phát hiện.
            min_size: Kích thước tối thiểu của khuôn mặt để phát hiện.
            max_size: Kích thước tối đa của khuôn mặt để phát hiện.

        Returns:
            Danh sách các hình chữ nhật (x, y, w, h) bao quanh khuôn mặt phát hiện được.
        """
        if image is None or image.size == 0:
            return []

        # Tiền xử lý ảnh
        processed = self._preprocess_image(image)
        if processed is None:
            return []
            
        # Phát hiện khuôn mặt chính diện
        faces = self.face_cascade.detectMultiScale(
            processed,
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
    
    def detect_and_extract(self, image, padding=0):
        """
        Phát hiện và trích xuất tất cả khuôn mặt trong ảnh
        
        Args:
            image: Ảnh đầu vào
            padding: Padding thêm xung quanh khuôn mặt
            
        Returns:
            List các ảnh khuôn mặt và tọa độ tương ứng
        """
        faces = self.detect_faces(image)
        
        face_images = []
        face_locations = []
        
        for face_rect in faces:
            face_img = self.extract_face(image, face_rect, padding)
            face_images.append(face_img)
            face_locations.append(face_rect)
        
        return face_images, face_locations
