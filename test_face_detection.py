import cv2
from utils.face_detection_simple import SimpleFaceDetector

def main():
    # Khởi tạo detector
    try:
        detector = SimpleFaceDetector()
        print("✓ Đã khởi tạo thành công bộ phát hiện khuôn mặt")
        
        # Mở camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Không thể mở camera")
            
        print("Nhấn 'q' để thoát")
        
        while True:
            # Đọc frame từ camera
            ret, frame = cap.read()
            if not ret:
                print("Không thể đọc frame từ camera")
                break
                
            # Phát hiện khuôn mặt
            faces = detector.detect_faces(frame)
            
            # Vẽ khung bao quanh khuôn mặt
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Hiển thị kết quả
            cv2.imshow('Face Detection', frame)
            
            # Thoát nếu nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Lỗi: {str(e)}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
