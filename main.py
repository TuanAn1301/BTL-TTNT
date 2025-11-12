"""
File main - Menu chính của hệ thống điểm danh khuôn mặt
"""
import os
import sys
from utils.logger import setup_logger
from utils.file_handler import ensure_directories
from utils import init_attendance_database, migrate_attendance_csv_to_db
from data_collector import DataCollector
from train_model import FaceTrainer
from face_recognition_system import FaceRecognitionSystem

logger = setup_logger(__name__)


def print_menu():
    """
    In menu chính
    """
    print("\n" + "=" * 50)
    print("HỆ THỐNG ĐIỂM DANH KHUÔN MẶT")
    print("=" * 50)
    print("1. Thu thập dữ liệu khuôn mặt")
    print("2. Huấn luyện mô hình")
    print("3. Chạy hệ thống điểm danh")
    print("4. Xem danh sách người trong dataset")
    print("5. Thoát")
    print("=" * 50)


def collect_data():
    """
    Thu thập dữ liệu khuôn mặt
    """
    print("\n--- THU THẬP DỮ LIỆU KHUÔN MẶT ---")
    person_name = input("Nhập tên người cần thu thập dữ liệu: ").strip()
    
    if not person_name:
        print("✗ Tên không được để trống!")
        return
    
    try:
        collector = DataCollector()
        collector.collect_faces(person_name)
        print(f"\n✓ Đã hoàn thành thu thập dữ liệu cho {person_name}")
    except Exception as e:
        logger.error(f"Lỗi khi thu thập dữ liệu: {str(e)}")
        print(f"✗ Lỗi: {str(e)}")


def train_model():
    """
    Huấn luyện mô hình
    """
    print("\n--- HUẤN LUYỆN MÔ HÌNH ---")
    confirm = input("Bạn có chắc chắn muốn huấn luyện lại mô hình? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Đã hủy huấn luyện")
        return
    
    try:
        trainer = FaceTrainer()
        success = trainer.train_and_save()
        
        if success:
            print("\n✓ Huấn luyện mô hình thành công!")
        else:
            print("\n✗ Huấn luyện mô hình thất bại!")
    except Exception as e:
        logger.error(f"Lỗi khi huấn luyện: {str(e)}")
        print(f"✗ Lỗi: {str(e)}")


def run_recognition():
    """
    Chạy hệ thống nhận diện và điểm danh
    """
    print("\n--- HỆ THỐNG ĐIỂM DANH ---")
    
    try:
        system = FaceRecognitionSystem()
        system.run()
    except FileNotFoundError:
        print("✗ Chưa có mô hình! Vui lòng huấn luyện mô hình trước.")
    except Exception as e:
        logger.error(f"Lỗi khi chạy hệ thống: {str(e)}")
        print(f"✗ Lỗi: {str(e)}")


def show_persons():
    """
    Hiển thị danh sách người trong dataset
    """
    from utils.file_handler import get_all_persons
    
    print("\n--- DANH SÁCH NGƯỜI TRONG DATASET ---")
    persons = get_all_persons()
    
    if not persons:
        print("Chưa có dữ liệu trong dataset")
        return
    
    print(f"Tổng số người: {len(persons)}")
    for i, person in enumerate(persons, 1):
        person_dir = os.path.join('dataset', person)
        num_images = len([f for f in os.listdir(person_dir) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"{i}. {person} - {num_images} ảnh")


def main():
    """
    Hàm main
    """
    # Đảm bảo các thư mục cần thiết tồn tại
    ensure_directories()
    # Khởi tạo CSDL điểm danh (SQLite)
    init_attendance_database()
    # Nhập dữ liệu CSV cũ (nếu có) vào SQLite một lần
    migrate_attendance_csv_to_db(force=False)
    
    while True:
        try:
            print_menu()
            choice = input("Chọn chức năng (1-5): ").strip()
            
            if choice == '1':
                collect_data()
            elif choice == '2':
                train_model()
            elif choice == '3':
                run_recognition()
            elif choice == '4':
                show_persons()
            elif choice == '5':
                print("\nCảm ơn bạn đã sử dụng hệ thống!")
                break
            else:
                print("✗ Lựa chọn không hợp lệ! Vui lòng chọn từ 1-5.")
        
        except KeyboardInterrupt:
            print("\n\nĐã dừng chương trình")
            break
        except Exception as e:
            logger.error(f"Lỗi không mong muốn: {str(e)}")
            print(f"✗ Lỗi: {str(e)}")


if __name__ == "__main__":
    main()
