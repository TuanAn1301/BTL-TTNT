"""
Script khởi tạo ban đầu cho hệ thống
"""
import os
import urllib.request
from config import HAARCASCADE_PATH, HAARCASCADES_DIR


def download_haarcascade():
    """
    Tải Haar Cascade file nếu chưa có
    """
    if os.path.exists(HAARCASCADE_PATH):
        print(f"✓ Haar Cascade đã tồn tại: {HAARCASCADE_PATH}")
        return True
    
    print("Đang tải Haar Cascade file...")
    
    try:
        # Tạo thư mục nếu chưa có
        os.makedirs(HAARCASCADES_DIR, exist_ok=True)
        
        # URL của file
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        
        # Tải file
        urllib.request.urlretrieve(url, HAARCASCADE_PATH)
        
        print(f"✓ Đã tải Haar Cascade: {HAARCASCADE_PATH}")
        return True
    
    except Exception as e:
        print(f"✗ Lỗi khi tải Haar Cascade: {e}")
        return False


def create_directories():
    """
    Tạo các thư mục cần thiết
    """
    from utils import ensure_directories
    
    print("Đang tạo các thư mục cần thiết...")
    ensure_directories()
    print("✓ Đã tạo các thư mục")


def check_dependencies():
    """
    Kiểm tra các thư viện cần thiết
    """
    print("\nKiểm tra thư viện...")
    
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'PyQt5': 'PyQt5',
        'sklearn': 'scikit-learn',
        'pandas': 'pandas'
    }
    
    missing = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - CHƯA CÀI ĐẶT")
            missing.append(package)
    
    # Kiểm tra face_recognition riêng
    try:
        import face_recognition
        print(f"✓ face-recognition")
    except ImportError:
        print(f"✗ face-recognition - CHƯA CÀI ĐẶT")
        missing.append('face-recognition')
    
    if missing:
        print(f"\n⚠️  Thiếu {len(missing)} thư viện:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nChạy lệnh sau để cài đặt:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("\n✓ Tất cả thư viện đã được cài đặt")
        return True


def main():
    """
    Khởi tạo hệ thống
    """
    print("=" * 60)
    print("KHỞI TẠO HỆ THỐNG ĐIỂM DANH KHUÔN MẶT")
    print("=" * 60)
    
    # 1. Kiểm tra thư viện
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n⚠️  Vui lòng cài đặt thư viện trước khi tiếp tục!")
        return
    
    # 2. Tạo thư mục
    print("\n" + "=" * 60)
    create_directories()
    
    # 3. Tải Haar Cascade
    print("\n" + "=" * 60)
    cascade_ok = download_haarcascade()
    
    # 4. Kiểm tra face detector
    print("\n" + "=" * 60)
    print("Kiểm tra Face Detector...")
    try:
        from utils import FaceDetector
        detector = FaceDetector()
        print("✓ Face Detector hoạt động tốt")
    except Exception as e:
        print(f"✗ Lỗi Face Detector: {e}")
        cascade_ok = False
    
    # Kết quả
    print("\n" + "=" * 60)
    if deps_ok and cascade_ok:
        print("✓ KHỞI TẠO THÀNH CÔNG!")
        print("\nBạn có thể chạy ứng dụng:")
        print("  python app.py")
    else:
        print("✗ KHỞI TẠO THẤT BẠI!")
        print("\nVui lòng khắc phục các lỗi trên trước khi chạy ứng dụng.")
    print("=" * 60)


if __name__ == "__main__":
    main()
