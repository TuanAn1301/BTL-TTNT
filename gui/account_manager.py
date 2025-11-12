from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QMessageBox, QComboBox, QHeaderView, QSizePolicy
from PyQt5.QtCore import Qt
from utils.helper import list_users, create_user, delete_user, update_user_password, update_user_role


class AccountManagerWidget(QWidget):
    def __init__(self, parent=None, current_user_role='user'):
        super().__init__(parent)
        self.current_user_role = current_user_role or 'user'
        self._build_ui()
        self.refresh_users()

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Table users
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Username", "Role"])
        # Hiển thị full form: kéo giãn cột và ẩn header dọc
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        vheader = self.table.verticalHeader()
        if vheader:
            vheader.setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.table)
        # Khi chọn một dòng, đổ dữ liệu xuống form để sửa nhanh role/password
        try:
            self.table.itemSelectionChanged.connect(self.on_table_select)
        except Exception:
            pass

        # Add form
        form = QHBoxLayout()
        self.input_username = QLineEdit()
        self.input_username.setPlaceholderText("username")
        form.addWidget(QLabel("Username:"))
        form.addWidget(self.input_username)
        self.input_password = QLineEdit()
        self.input_password.setEchoMode(QLineEdit.Password)
        self.input_password.setPlaceholderText("password")
        form.addWidget(QLabel("Password:"))
        form.addWidget(self.input_password)
        self.input_role = QComboBox()
        self.input_role.addItems(["admin", "data_manager", "attendance_viewer"])  # quyền hệ thống
        form.addWidget(QLabel("Role:"))
        form.addWidget(self.input_role)
        self.btn_add = QPushButton("Add/Update")
        self.btn_add.clicked.connect(self.on_add_update)
        form.addWidget(self.btn_add)
        # Nút cập nhật role riêng giúp chỉnh quyền nhanh
        self.btn_update_role = QPushButton("Update Role")
        self.btn_update_role.clicked.connect(self.on_update_role)
        form.addWidget(self.btn_update_role)
        layout.addLayout(form)

        # Actions
        actions = QHBoxLayout()
        self.input_new_password = QLineEdit()
        self.input_new_password.setEchoMode(QLineEdit.Password)
        self.input_new_password.setPlaceholderText("new password")
        actions.addWidget(QLabel("New Password:"))
        actions.addWidget(self.input_new_password)
        self.btn_reset_pw = QPushButton("Reset Password")
        self.btn_reset_pw.clicked.connect(self.on_reset_password)
        actions.addWidget(self.btn_reset_pw)
        self.btn_delete = QPushButton("Delete User")
        self.btn_delete.clicked.connect(self.on_delete_user)
        actions.addWidget(self.btn_delete)
        layout.addLayout(actions)
        # Bảng chiếm phần lớn không gian
        try:
            layout.setStretch(0, 1)   # table
            layout.setStretch(1, 0)   # form
            layout.setStretch(2, 0)   # actions
        except Exception:
            pass

        if self.current_user_role != 'admin':
            # Restrict non-admin
            self.btn_add.setEnabled(False)
            self.btn_delete.setEnabled(False)
            self.btn_reset_pw.setEnabled(False)
            self.input_role.setEnabled(False)
            self.btn_update_role.setEnabled(False)

    def refresh_users(self):
        try:
            users = list_users()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể tải danh sách người dùng: {str(e)}")
            users = []
        self.table.setRowCount(len(users))
        for i, (u, r) in enumerate(users):
            self.table.setItem(i, 0, QTableWidgetItem(u))
            self.table.setItem(i, 1, QTableWidgetItem(r or 'user'))

    def _selected_username(self):
        idxs = self.table.selectedIndexes()
        if not idxs:
            return None
        row = idxs[0].row()
        item = self.table.item(row, 0)
        return item.text().strip() if item else None

    def _selected_role(self):
        idxs = self.table.selectedIndexes()
        if not idxs:
            return None
        row = idxs[0].row()
        item = self.table.item(row, 1)
        return item.text().strip() if item else None

    def on_table_select(self):
        """Đổ dữ liệu hàng đang chọn xuống form để chỉnh sửa nhanh."""
        try:
            username = self._selected_username()
            role = self._selected_role()
            if username:
                self.input_username.setText(username)
            else:
                self.input_username.clear()
            # Không tự động đổ password
            self.input_password.clear()
            self.input_new_password.clear()
            if role:
                # Đặt combo role tương ứng nếu tồn tại
                idx = self.input_role.findText(role)
                if idx >= 0:
                    self.input_role.setCurrentIndex(idx)
        except Exception:
            pass

    def on_add_update(self):
        if self.current_user_role != 'admin':
            QMessageBox.warning(self, "Lỗi", "Chỉ admin mới có quyền thêm/sửa người dùng")
            return
        username = self.input_username.text().strip()
        password = self.input_password.text().strip()
        role = (self.input_role.currentText() or 'user').strip()
        if not username or not password:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập username và password")
            return
        try:
            create_user(username, password, role=role)
            QMessageBox.information(self, "Thành công", "Đã lưu tài khoản")
            self.refresh_users()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể lưu tài khoản: {str(e)}")

    def on_reset_password(self):
        if self.current_user_role != 'admin':
            QMessageBox.warning(self, "Lỗi", "Chỉ admin mới có quyền đặt lại mật khẩu")
            return
        username = self._selected_username()
        if not username:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn tài khoản")
            return
        new_pw = self.input_new_password.text().strip()
        if not new_pw:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập mật khẩu mới")
            return
        try:
            update_user_password(username, new_pw)
            QMessageBox.information(self, "Thành công", "Đã đặt lại mật khẩu")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể đặt lại mật khẩu: {str(e)}")

    def on_delete_user(self):
        if self.current_user_role != 'admin':
            QMessageBox.warning(self, "Lỗi", "Chỉ admin mới có quyền xoá tài khoản")
            return
        username = self._selected_username()
        if not username:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn tài khoản")
            return
        if username == 'admin':
            QMessageBox.warning(self, "Lỗi", "Không thể xoá tài khoản admin mặc định")
            return
        try:
            delete_user(username)
            QMessageBox.information(self, "Thành công", "Đã xoá tài khoản")
            self.refresh_users()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể xoá tài khoản: {str(e)}")

    def on_update_role(self):
        """Cập nhật vai trò cho tài khoản đã chọn/đang ở form."""
        if self.current_user_role != 'admin':
            QMessageBox.warning(self, "Lỗi", "Chỉ admin mới có quyền thay đổi role")
            return
        username = (self.input_username.text().strip() or self._selected_username())
        if not username:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn tài khoản hoặc nhập username")
            return
        new_role = (self.input_role.currentText() or '').strip()
        if new_role not in ("admin", "data_manager", "attendance_viewer"):
            QMessageBox.warning(self, "Lỗi", "Role không hợp lệ")
            return
        try:
            update_user_role(username, new_role)
            QMessageBox.information(self, "Thành công", f"Đã cập nhật role cho '{username}' → {new_role}")
            self.refresh_users()
            # Đồng bộ lại table selection
            self.on_table_select()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể cập nhật role: {str(e)}")


