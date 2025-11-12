"""
Module mã hóa đặc trưng khuôn mặt sử dụng face_recognition
"""
import cv2
import numpy as np

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    # Không hiển thị warning khi import, chỉ khi thực sự dùng
    # print("Warning: face_recognition chưa được cài đặt. Một số tính năng sẽ không khả dụng.")

from config import FACE_ENCODING_MODEL


class FaceEncoder:
    """
    Class mã hóa đặc trưng khuôn mặt
    """
    
    def __init__(self, model=None):
        """
        Khởi tạo Face Encoder
        
        Args:
            model: Model sử dụng ('small' hoặc 'large')
        """
        if not FACE_RECOGNITION_AVAILABLE:
            raise ImportError(
                "⚠️ face_recognition chưa được cài đặt!\n\n"
                "Để sử dụng ML model nhận diện, vui lòng cài đặt:\n"
                "  pip install face-recognition\n\n"
                "Lưu ý: Hệ thống vẫn hoạt động với template matching (không cần face_recognition)"
            )
        self.model = model if model else FACE_ENCODING_MODEL
    
    def encode_face(self, image, face_location=None):
        """
        Mã hóa khuôn mặt thành vector đặc trưng
        
        Args:
            image: Ảnh đầu vào (BGR hoặc RGB)
            face_location: Vị trí khuôn mặt (top, right, bottom, left) - optional
            
        Returns:
            Face encoding (128-d vector) hoặc None nếu không phát hiện được
        """
        # Chuyển sang RGB nếu cần (face_recognition dùng RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Kiểm tra xem có phải BGR không (OpenCV format)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Tạo encoding
        if face_location:
            # Nếu đã biết vị trí khuôn mặt
            encodings = face_recognition.face_encodings(
                rgb_image,
                known_face_locations=[face_location],
                model=self.model
            )
        else:
            # Tự động phát hiện khuôn mặt
            encodings = face_recognition.face_encodings(
                rgb_image,
                model=self.model
            )
        
        if len(encodings) > 0:
            return encodings[0]
        else:
            return None
    
    def encode_faces(self, image):
        """
        Mã hóa tất cả khuôn mặt trong ảnh
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            List các face encodings và vị trí tương ứng
        """
        # Chuyển sang RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Phát hiện vị trí khuôn mặt
        face_locations = face_recognition.face_locations(rgb_image)
        
        # Tạo encodings
        encodings = face_recognition.face_encodings(
            rgb_image,
            known_face_locations=face_locations,
            model=self.model
        )
        
        return encodings, face_locations
    
    def compare_faces(self, known_encodings, face_encoding, tolerance=0.6):
        """
        So sánh khuôn mặt với danh sách khuôn mặt đã biết
        
        Args:
            known_encodings: List các encoding đã biết
            face_encoding: Encoding cần so sánh
            tolerance: Ngưỡng so sánh (càng nhỏ càng strict)
            
        Returns:
            List boolean cho biết có khớp hay không
        """
        return face_recognition.compare_faces(
            known_encodings,
            face_encoding,
            tolerance=tolerance
        )
    
    def face_distance(self, known_encodings, face_encoding):
        """
        Tính khoảng cách giữa khuôn mặt và danh sách khuôn mặt đã biết
        
        Args:
            known_encodings: List các encoding đã biết
            face_encoding: Encoding cần so sánh
            
        Returns:
            Numpy array các khoảng cách
        """
        return face_recognition.face_distance(known_encodings, face_encoding)
    
    def find_best_match(self, known_encodings, known_names, face_encoding, tolerance=0.6):
        """
        Tìm khuôn mặt khớp nhất
        
        Args:
            known_encodings: List các encoding đã biết
            known_names: List tên tương ứng
            face_encoding: Encoding cần so sánh
            tolerance: Ngưỡng so sánh
            
        Returns:
            Tuple (name, distance) hoặc (None, None) nếu không tìm thấy
        """
        if len(known_encodings) == 0:
            return None, None
        
        # Tính khoảng cách
        distances = self.face_distance(known_encodings, face_encoding)
        
        # Tìm khoảng cách nhỏ nhất
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        # Kiểm tra ngưỡng
        if min_distance <= tolerance:
            return known_names[min_distance_idx], min_distance
        else:
            return None, min_distance
    
    def convert_opencv_to_face_recognition_location(self, opencv_rect):
        """
        Chuyển đổi tọa độ từ OpenCV (x, y, w, h) sang face_recognition (top, right, bottom, left)
        
        Args:
            opencv_rect: Tuple (x, y, w, h)
            
        Returns:
            Tuple (top, right, bottom, left)
        """
        x, y, w, h = opencv_rect
        return (y, x + w, y + h, x)
    
    def convert_face_recognition_to_opencv_location(self, fr_location):
        """
        Chuyển đổi tọa độ từ face_recognition (top, right, bottom, left) sang OpenCV (x, y, w, h)
        
        Args:
            fr_location: Tuple (top, right, bottom, left)
            
        Returns:
            Tuple (x, y, w, h)
        """
        top, right, bottom, left = fr_location
        return (left, top, right - left, bottom - top)
