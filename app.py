"""
Chương trình chính - Giao diện PyQt5
"""
import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from gui.main_window import MainWindow
from utils import ensure_directories, setup_logger, init_attendance_database, migrate_attendance_csv_to_db
from utils.helper import ensure_default_admin

logger = setup_logger(__name__)


def main():
    """
    Hàm main
    """
    # Đảm bảo các thư mục tồn tại
    ensure_directories()
    # Khởi tạo CSDL điểm danh (SQLite)
    init_attendance_database()
    # Tạo admin mặc định nếu chưa có
    try:
        ensure_default_admin()
    except Exception:
        pass
    # Nhập dữ liệu CSV cũ (nếu có) vào SQLite một lần
    migrate_attendance_csv_to_db(force=False)
    
    # Khởi tạo ứng dụng
    app = QApplication(sys.argv)
    
    # Thiết lập style
    app.setStyle('Fusion')
    
    # Thiết lập icon ứng dụng và cửa sổ
    base_dir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(base_dir, 'gui', 'logo', 'logo.png')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    # Tạo cửa sổ chính
    window = MainWindow()
    if os.path.exists(icon_path):
        window.setWindowIcon(QIcon(icon_path))
    window.show()
    
    logger.info("Ứng dụng đã khởi động")
    
    # Chạy ứng dụng
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
