# Hệ Thống Điểm Danh Khuôn Mặt

Hệ thống điểm danh tự động sử dụng công nghệ nhận diện khuôn mặt với OpenCV và face_recognition.
Hỗ trợ cả giao diện PyQt5 (offline) và Google Colab notebooks (online).

## Tính năng

- ✅ Thu thập dữ liệu khuôn mặt
- ✅ Huấn luyện mô hình nhận diện
- ✅ Nhận diện khuôn mặt real-time
- ✅ Điểm danh tự động
- ✅ Lưu lịch sử điểm danh theo ngày

## Cài đặt

### 1. Yêu cầu hệ thống

- Python 3.8 trở lên
- Webcam
- Windows/Linux/MacOS

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

**Lưu ý:** Nếu gặp lỗi khi cài đặt `dlib`, bạn có thể:

**Windows:**
```bash
pip install cmake
pip install dlib
```

Hoặc tải file wheel từ: https://github.com/z-mahmud22/Dlib_Windows_Python3.x

**Linux/MacOS:**
```bash
sudo apt-get install cmake
sudo apt-get install libboost-all-dev
pip install dlib
```

### 3. Tải Haar Cascade

Tải file `haarcascade_frontalface_default.xml` và đặt vào thư mục `haarcascades/`:

```bash
# Tải từ GitHub OpenCV
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml -P haarcascades/
```

Hoặc tải thủ công từ: https://github.com/opencv/opencv/tree/master/data/haarcascades

## Sử dụng

### Chạy ứng dụng với giao diện PyQt5 (Offline)

```bash
python app.py
```

### Sử dụng Google Colab (Online)

1. Upload thư mục dự án lên Google Drive
2. Mở các notebook trong thư mục `notebooks/` bằng Google Colab
3. Chạy từng cell theo thứ tự

### Chạy chương trình console (Cũ)

```bash
python main.py
```

### Menu chức năng

1. **Thu thập dữ liệu khuôn mặt**
   - Nhập tên người cần thu thập
   - Nhấn SPACE để chụp ảnh
   - Thu thập 50 ảnh cho mỗi người

2. **Huấn luyện mô hình**
   - Huấn luyện mô hình từ dữ liệu đã thu thập
   - Mô hình được lưu vào `models/face_recognition_model.pkl`

3. **Chạy hệ thống điểm danh**
   - Nhận diện khuôn mặt real-time
   - Tự động điểm danh khi nhận diện thành công
   - Cooldown 5 phút giữa các lần điểm danh

4. **Xem danh sách người trong dataset**
   - Hiển thị tất cả người đã thu thập dữ liệu

### Chạy từng module riêng lẻ

**Thu thập dữ liệu:**
```bash
python data_collector.py
```

**Huấn luyện mô hình:**
```bash
python train_model.py
```

**Chạy nhận diện:**
```bash
python face_recognition_system.py
```

## Cấu trúc thư mục

```
face_attendance/
│
├── dataset/                         # Lưu hình ảnh khuôn mặt (theo tên sinh viên)
│   ├── NguyenVanA/
│   ├── TranThiB/
│   └── ...
│
├── models/                          # Lưu mô hình đã huấn luyện
│   └── face_recognition_model.pkl
│
├── utils/                           # Các hàm tiện ích
│   ├── __init__.py
│   ├── face_detection.py            # Hàm phát hiện khuôn mặt (OpenCV + Haarcascade)
│   ├── face_encoding.py             # Mã hóa đặc trưng khuôn mặt (face_recognition)
│   ├── helper.py                    # Các hàm hỗ trợ khác
│   ├── logger.py                    # Logging
│   └── file_handler.py              # Xử lý file
│
├── gui/                             # Giao diện PyQt5
│   ├── __init__.py
│   └── main_window.py               # Cửa sổ chính
│
├── notebooks/                       # Notebook trên Google Colab
│   ├── 01_collect_images.ipynb      # Thu thập và lưu ảnh khuôn mặt
│   ├── 02_train_model.ipynb         # Huấn luyện mô hình nhận diện khuôn mặt
│   ├── 03_face_recognition.ipynb    # Nhận diện khuôn mặt theo thời gian thực
│   └── 04_attendance_logging.ipynb  # Ghi danh sách điểm danh vào file CSV
│
├── attendance/                      # Thư mục chứa file điểm danh
│   ├── attendance_2025-10-16.csv
│   └── attendance_history.csv
│
├── haarcascades/                    # Lưu file cascade của OpenCV
│   └── haarcascade_frontalface_default.xml
│
├── app.py                           # Chương trình chính (PyQt5 GUI)
├── main.py                          # Chương trình console (cũ)
├── data_collector.py                # Module thu thập dữ liệu
├── train_model.py                   # Module huấn luyện
├── face_recognition_system.py       # Module nhận diện
├── config.py                        # Cấu hình chung
├── requirements.txt                 # Liệt kê các thư viện cần cài
├── .gitignore                       # Git ignore
└── README.md                        # Mô tả dự án
```

## Cấu hình

Chỉnh sửa file `config.py` để thay đổi các tham số:

- `RECOGNITION_THRESHOLD`: Ngưỡng độ tin cậy (0-1)
- `ATTENDANCE_COOLDOWN`: Thời gian chờ giữa các lần điểm danh (giây)
- `IMAGES_PER_PERSON`: Số ảnh thu thập cho mỗi người
- `CAMERA_INDEX`: Index của camera
- Và nhiều tham số khác...

## Lưu ý

- Đảm bảo có đủ ánh sáng khi thu thập dữ liệu
- Thu thập ảnh từ nhiều góc độ khác nhau
- Mỗi người nên có ít nhất 30-50 ảnh
- Huấn luyện lại mô hình khi thêm người mới

## Xử lý lỗi thường gặp

### Lỗi: "Không thể mở camera"
- Kiểm tra camera có hoạt động không
- Thử thay đổi `CAMERA_INDEX` trong `config.py`

### Lỗi: "Không tìm thấy file mô hình"
- Chạy huấn luyện mô hình trước khi nhận diện

### Lỗi: "Không phát hiện khuôn mặt"
- Đảm bảo có đủ ánh sáng
- Kiểm tra file Haar Cascade đã tải đúng chưa

## Công nghệ sử dụng

- **OpenCV**: Xử lý ảnh và video
- **face_recognition**: Nhận diện khuôn mặt
- **scikit-learn**: Huấn luyện mô hình SVM
- **dlib**: Face detection và encoding

## Tác giả

Hệ thống điểm danh khuôn mặt - 2024 - Nguyễn Trường Quân - Dương Tuấn An (chỉnh sửa và cải tiến)

## License

MIT License
