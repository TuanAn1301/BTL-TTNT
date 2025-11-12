from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QMessageBox, QFileDialog, QHeaderView, QSizePolicy
from PyQt5.QtCore import Qt
from utils.helper import list_all_students, delete_student
from config import DATASET_DIR
import pandas as pd
import os
import shutil


class StudentManagerWidget(QWidget):
    def __init__(self, parent=None, current_user_role='admin'):
        super().__init__(parent)
        self.current_user_role = current_user_role or 'admin'
        self.all_students = []  # Lưu toàn bộ danh sách để tìm kiếm
        self._build_ui()
        self.refresh_students()

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Thanh tìm kiếm
        search_bar = QHBoxLayout()
        search_bar.addWidget(QLabel("Tìm tên:"))
        self.search_name = QLineEdit()
        self.search_name.setPlaceholderText("VD: Nguyễn Văn A")
        self.search_name.textChanged.connect(self.filter_students)
        search_bar.addWidget(self.search_name)
        
        search_bar.addWidget(QLabel("Mã SV:"))
        self.search_student_id = QLineEdit()
        self.search_student_id.setPlaceholderText("VD: B21DCCN001")
        self.search_student_id.textChanged.connect(self.filter_students)
        search_bar.addWidget(self.search_student_id)
        
        search_bar.addWidget(QLabel("Lớp:"))
        self.search_class = QLineEdit()
        self.search_class.setPlaceholderText("VD: D21CQCN01-N")
        self.search_class.textChanged.connect(self.filter_students)
        search_bar.addWidget(self.search_class)
        
        layout.addLayout(search_bar)

        # Thanh nút phía trên bảng
        button_bar = QHBoxLayout()
        button_bar.addStretch(1)
        self.btn_delete = QPushButton("🗑️ Xóa Sinh Viên")
        self.btn_delete.clicked.connect(self.on_delete_student)
        self.btn_delete.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; padding: 8px 16px; border-radius: 5px; font-weight: bold; }")
        # Chỉ admin và data_manager mới có quyền xóa
        if self.current_user_role not in ('admin', 'data_manager'):
            self.btn_delete.setEnabled(False)
        button_bar.addWidget(self.btn_delete)
        
        self.btn_refresh = QPushButton("🔄 Làm Mới")
        self.btn_refresh.clicked.connect(self.refresh_students)
        self.btn_refresh.setStyleSheet("QPushButton { background-color: #3498db; color: white; padding: 8px 16px; border-radius: 5px; font-weight: bold; }")
        button_bar.addWidget(self.btn_refresh)
        
        self.btn_export = QPushButton("📊 Xuất File Excel")
        self.btn_export.clicked.connect(self.export_to_excel)
        self.btn_export.setStyleSheet("QPushButton { background-color: #27ae60; color: white; padding: 8px 16px; border-radius: 5px; font-weight: bold; }")
        button_bar.addWidget(self.btn_export)
        layout.addLayout(button_bar)

        # Table students
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Tên", "Mã SV", "Lớp"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSortingEnabled(True)
        # Kéo giãn cột để lấp đầy chiều ngang form
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        vheader = self.table.verticalHeader()
        if vheader:
            vheader.setVisible(False)
        # Bảng chiếm hết không gian còn lại
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.table)
        # Đảm bảo bảng chiếm phần lớn không gian
        try:
            layout.setStretch(0, 0)   # search bar
            layout.setStretch(1, 0)   # buttons
            layout.setStretch(2, 1)   # table
        except Exception:
            pass

    def refresh_students(self):
        """Làm mới danh sách sinh viên từ database và áp dụng bộ lọc."""
        try:
            self.all_students = list_all_students()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể tải danh sách sinh viên: {str(e)}")
            self.all_students = []
        
        # Áp dụng bộ lọc
        self.filter_students()

    def filter_students(self):
        """Lọc danh sách sinh viên theo các điều kiện tìm kiếm."""
        # Lấy từ khóa tìm kiếm
        name_kw = self.search_name.text().strip().lower() if hasattr(self, 'search_name') and self.search_name else ''
        sid_kw = self.search_student_id.text().strip().lower() if hasattr(self, 'search_student_id') and self.search_student_id else ''
        class_kw = self.search_class.text().strip().lower() if hasattr(self, 'search_class') and self.search_class else ''
        
        # Lọc danh sách
        filtered = []
        for name, sid, cls in self.all_students:
            name_str = (name or '').lower()
            sid_str = (sid or '').lower()
            cls_str = (cls or '').lower()
            
            # Kiểm tra điều kiện (tất cả điều kiện đều phải khớp nếu có nhập)
            match = True
            if name_kw and name_kw not in name_str:
                match = False
            if sid_kw and sid_kw not in sid_str:
                match = False
            if class_kw and class_kw not in cls_str:
                match = False
            
            if match:
                filtered.append((name, sid, cls))
        
        # Hiển thị kết quả
        self.table.setRowCount(len(filtered))
        for i, (name, sid, cls) in enumerate(filtered):
            self.table.setItem(i, 0, QTableWidgetItem(name or ''))
            self.table.setItem(i, 1, QTableWidgetItem(sid or ''))
            self.table.setItem(i, 2, QTableWidgetItem(cls or ''))

    def _selected_student_info(self):
        """Lấy thông tin sinh viên được chọn (name, student_id)."""
        idxs = self.table.selectedIndexes()
        if not idxs:
            return None, None
        row = idxs[0].row()
        name_item = self.table.item(row, 0)
        sid_item = self.table.item(row, 1)
        name = name_item.text().strip() if name_item else ''
        sid = sid_item.text().strip() if sid_item else ''
        return name, sid

    def on_delete_student(self):
        """Xóa sinh viên khỏi database và xóa thư mục ảnh."""
        if self.current_user_role not in ('admin', 'data_manager'):
            QMessageBox.warning(self, "Lỗi", "Bạn không có quyền xóa sinh viên")
            return
        
        name, sid = self._selected_student_info()
        if not name:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn sinh viên cần xóa")
            return
        
        reply = QMessageBox.question(
            self, "Xác nhận xóa",
            f"Bạn có chắc chắn muốn xóa sinh viên '{name}'?\n\n"
            f"Thao tác này sẽ:\n"
            f"- Xóa thông tin sinh viên khỏi database\n"
            f"- Xóa toàn bộ ảnh của sinh viên trong dataset",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # 1. Xóa thông tin trong database
                delete_student(name)
                
                # 2. Tìm và xóa thư mục ảnh
                # Có thể có nhiều format: name_sid, hoặc chỉ name
                deleted_folders = []
                if os.path.exists(DATASET_DIR):
                    # Thử tìm theo name_sid trước
                    possible_folder_names = []
                    if sid:
                        possible_folder_names.append(f"{name}_{sid}")
                    possible_folder_names.append(name)  # Thử theo tên
                    
                    for folder_name in possible_folder_names:
                        folder_path = os.path.join(DATASET_DIR, folder_name)
                        if os.path.exists(folder_path) and os.path.isdir(folder_path):
                            try:
                                shutil.rmtree(folder_path)
                                deleted_folders.append(folder_name)
                            except Exception as e:
                                QMessageBox.warning(self, "Cảnh báo", f"Không thể xóa thư mục {folder_name}: {str(e)}")
                
                # Thông báo kết quả
                msg = f"Đã xóa sinh viên: {name}\n"
                if deleted_folders:
                    msg += f"Đã xóa {len(deleted_folders)} thư mục ảnh"
                else:
                    msg += "Không tìm thấy thư mục ảnh để xóa"
                QMessageBox.information(self, "Thành công", msg)
                
                # Làm mới danh sách
                self.refresh_students()
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể xóa sinh viên: {str(e)}")

    def export_to_excel(self):
        """Xuất danh sách sinh viên ra file Excel."""
        try:
            # Lấy dữ liệu từ bảng hiện tại (đã lọc)
            rows = []
            for i in range(self.table.rowCount()):
                name = self.table.item(i, 0).text() if self.table.item(i, 0) else ''
                sid = self.table.item(i, 1).text() if self.table.item(i, 1) else ''
                cls = self.table.item(i, 2).text() if self.table.item(i, 2) else ''
                rows.append({'Tên': name, 'Mã SV': sid, 'Lớp': cls})
            
            if not rows:
                QMessageBox.information(self, "Thông báo", "Không có dữ liệu để xuất!")
                return
            
            # Chọn file để lưu
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Xuất File Excel", "danh_sach_sinh_vien.xlsx",
                "Excel Files (*.xlsx);;All Files (*)"
            )
            
            if file_path:
                # Tạo DataFrame và xuất
                df = pd.DataFrame(rows)
                df.to_excel(file_path, index=False, engine='openpyxl')
                QMessageBox.information(self, "Thành công", f"Đã xuất file Excel:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể xuất file Excel: {str(e)}")
