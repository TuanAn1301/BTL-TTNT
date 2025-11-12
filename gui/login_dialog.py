from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from PyQt5.QtCore import Qt
from utils import init_attendance_database
from utils.helper import verify_user, ensure_default_admin


class LoginDialog(QDialog):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setWindowTitle("Đăng nhập")
		self.setModal(True)
		self._role = None
		self._username = None
		self._build_ui()
		self._ensure_default_admin()

	def _build_ui(self):
		layout = QVBoxLayout()
		self.setLayout(layout)

		layout.addWidget(QLabel("Tên đăng nhập"))
		self.input_username = QLineEdit()
		self.input_username.setPlaceholderText("admin")
		layout.addWidget(self.input_username)

		layout.addWidget(QLabel("Mật khẩu"))
		self.input_password = QLineEdit()
		self.input_password.setEchoMode(QLineEdit.Password)
		self.input_password.setPlaceholderText("••••••••")
		layout.addWidget(self.input_password)

		btn_row = QHBoxLayout()
		self.btn_login = QPushButton("Đăng nhập")
		self.btn_login.clicked.connect(self.on_login)
		btn_row.addWidget(self.btn_login)
		self.btn_cancel = QPushButton("Thoát")
		self.btn_cancel.clicked.connect(self.reject)
		btn_row.addWidget(self.btn_cancel)
		layout.addLayout(btn_row)

		self.setFixedWidth(360)

	def _ensure_default_admin(self):
		try:
			init_attendance_database()
			ensure_default_admin()
		except Exception:
			pass

	def on_login(self):
		username = self.input_username.text().strip()
		password = self.input_password.text().strip()
		if not username or not password:
			QMessageBox.warning(self, "Lỗi", "Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu!")
			return
		ok, role = verify_user(username, password)
		if not ok:
			QMessageBox.critical(self, "Đăng nhập thất bại", "Sai tên đăng nhập hoặc mật khẩu!")
			return
		self._username = username
		self._role = role or 'user'
		self.accept()

	def current_user(self):
		return self._username, (self._role or 'user')
