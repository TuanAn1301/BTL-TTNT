"""
Module các hàm hỗ trợ khác
"""
import os
import csv
import json
import sqlite3
from datetime import datetime
from config import (
    ATTENDANCE_DIR,
    ATTENDANCE_TIME_FORMAT,
    ATTENDANCE_DATE_FORMAT,
    ATTENDANCE_DB_PATH
)
import hashlib
import os as _os


def ensure_directory(directory):
    """
    Tạo thư mục nếu chưa tồn tại
    
    Args:
        directory: Đường dẫn thư mục
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def init_attendance_database():
    """
    Khởi tạo cơ sở dữ liệu SQLite cho điểm danh nếu chưa tồn tại
    """
    ensure_directory(ATTENDANCE_DIR)
    conn = sqlite3.connect(ATTENDANCE_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                confidence REAL NOT NULL,
                student_id TEXT,
                class TEXT
            )
            """
        )
        migrate_attendance_add_student_columns(cur)
        # Bảng lưu thông tin sinh viên (name -> student_id, class)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                name TEXT PRIMARY KEY,
                student_id TEXT,
                class TEXT
            )
            """
        )
        migrate_students_unique_student_id(cur)
        # Bảng lưu encoding khuôn mặt của sinh viên (ưu tiên dùng để rà trùng)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS students_encodings (
                name TEXT PRIMARY KEY,
                encoding TEXT
            )
            """
        )
        # Bảng người dùng hệ thống (đăng nhập)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT DEFAULT 'user'
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


# ========== AUTH HELPERS ==========
def _hash_password(plain_password, salt):
    data = (salt + plain_password).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def create_user(username, plain_password, role='user'):
    ensure_directory(ATTENDANCE_DIR)
    conn = sqlite3.connect(ATTENDANCE_DB_PATH)
    try:
        cur = conn.cursor()
        salt = hashlib.sha256(_os.urandom(32)).hexdigest()[:32]
        pwd_hash = _hash_password(plain_password, salt)
        cur.execute(
            """
            INSERT INTO users(username, password_hash, salt, role)
            VALUES(?,?,?,?)
            ON CONFLICT(username) DO UPDATE SET
                password_hash=excluded.password_hash,
                salt=excluded.salt,
                role=excluded.role
            """,
            (username, pwd_hash, salt, role)
        )
        conn.commit()
    finally:
        conn.close()


def verify_user(username, plain_password):
    ensure_directory(ATTENDANCE_DIR)
    if not os.path.exists(ATTENDANCE_DB_PATH):
        return False, None
    conn = sqlite3.connect(ATTENDANCE_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT password_hash, salt, role FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        if not row:
            return False, None
        stored_hash, salt, role = row
        return (_hash_password(plain_password, salt) == stored_hash), role
    finally:
        conn.close()


def list_users():
    ensure_directory(ATTENDANCE_DIR)
    if not os.path.exists(ATTENDANCE_DB_PATH):
        return []
    conn = sqlite3.connect(ATTENDANCE_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT username, role FROM users ORDER BY username ASC")
        return cur.fetchall()
    finally:
        conn.close()


def delete_user(username):
    ensure_directory(ATTENDANCE_DIR)
    conn = sqlite3.connect(ATTENDANCE_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE username=?", (username,))
        conn.commit()
    finally:
        conn.close()


def update_user_password(username, new_plain_password):
    ensure_directory(ATTENDANCE_DIR)
    conn = sqlite3.connect(ATTENDANCE_DB_PATH)
    try:
        cur = conn.cursor()
        salt = hashlib.sha256(_os.urandom(32)).hexdigest()[:32]
        pwd_hash = _hash_password(new_plain_password, salt)
        cur.execute("UPDATE users SET password_hash=?, salt=? WHERE username=?", (pwd_hash, salt, username))
        conn.commit()
    finally:
        conn.close()


def update_user_role(username, role):
    ensure_directory(ATTENDANCE_DIR)
    conn = sqlite3.connect(ATTENDANCE_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("UPDATE users SET role=? WHERE username=?", (role, username))
        conn.commit()
    finally:
        conn.close()


def ensure_default_admin():
    """Đảm bảo có tài khoản admin mặc định trong hệ thống."""
    ensure_directory(ATTENDANCE_DIR)
    conn = sqlite3.connect(ATTENDANCE_DB_PATH)
    try:
        cur = conn.cursor()
        # Kiểm tra đã có user nào chưa
        cur.execute("SELECT COUNT(1) FROM users")
        cnt = cur.fetchone()[0] if cur.fetchone is not None else 0
        if cnt and cnt > 0:
            # Nếu đã có ít nhất 1 user, đảm bảo có ít nhất 1 admin
            cur.execute("SELECT 1 FROM users WHERE role='admin' LIMIT 1")
            if cur.fetchone():
                return
        # Tạo admin/admin
        salt = hashlib.sha256(_os.urandom(32)).hexdigest()[:32]
        pwd_hash = _hash_password("admin", salt)
        cur.execute(
            """
            INSERT INTO users(username, password_hash, salt, role)
            VALUES(?,?,?,?)
            ON CONFLICT(username) DO UPDATE SET
                password_hash=excluded.password_hash,
                salt=excluded.salt,
                role=excluded.role
            """,
            ("admin", pwd_hash, salt, "admin")
        )
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


def save_student_encoding(name, encoding_vector):
    """
    Lưu/ cập nhật face encoding cho sinh viên vào DB dưới dạng JSON.
    encoding_vector: list/ndarray (128 chiều)
    """
    ensure_directory(ATTENDANCE_DIR)
    conn = sqlite3.connect(ATTENDANCE_DB_PATH)
    try:
        cur = conn.cursor()
        enc_json = json.dumps(list(encoding_vector)) if encoding_vector is not None else None
        cur.execute(
            """
            INSERT INTO students_encodings(name, encoding)
            VALUES(?,?)
            ON CONFLICT(name) DO UPDATE SET
                encoding=excluded.encoding
            """,
            (name, enc_json)
        )
        conn.commit()
    finally:
        conn.close()


def load_student_encodings(exclude_name=None):
    """
    Trả về list các tuple (name, encoding_vector) từ DB. Có thể loại trừ một tên.
    """
    ensure_directory(ATTENDANCE_DIR)
    if not os.path.exists(ATTENDANCE_DB_PATH):
        return []
    conn = sqlite3.connect(ATTENDANCE_DB_PATH)
    try:
        cur = conn.cursor()
        if exclude_name:
            cur.execute("SELECT name, encoding FROM students_encodings WHERE name<>?", (exclude_name,))
        else:
            cur.execute("SELECT name, encoding FROM students_encodings")
        rows = cur.fetchall()
        result = []
        for n, enc_json in rows:
            try:
                if enc_json:
                    vec = json.loads(enc_json)
                    result.append((n, vec))
            except Exception:
                continue
        return result
    finally:
        conn.close()


def migrate_attendance_add_student_columns(cur):
    cur.execute("PRAGMA table_info(attendance)")
    cols = [row[1] for row in cur.fetchall()]
    if 'student_id' not in cols:
        try:
            cur.execute("ALTER TABLE attendance ADD COLUMN student_id TEXT")
        except Exception:
            pass
    if 'class' not in cols:
        try:
            cur.execute("ALTER TABLE attendance ADD COLUMN class TEXT")
        except Exception:
            pass


def migrate_students_unique_student_id(cur):
    """
    Đảm bảo student_id là duy nhất (bỏ qua NULL và chuỗi rỗng).
    """
    try:
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_students_student_id_unique
            ON students(student_id)
            WHERE student_id IS NOT NULL AND TRIM(student_id) <> ''
            """
        )
    except Exception:
        # Nếu đang có dữ liệu trùng, index sẽ không tạo được.
        # Ứng dụng sẽ xử lý logic trùng lặp ở tầng trên.
        pass


def get_current_date():
    """
    Lấy ngày hiện tại theo format
    
    Returns:
        String ngày hiện tại
    """
    return datetime.now().strftime(ATTENDANCE_DATE_FORMAT)


def get_current_time():
    """
    Lấy thời gian hiện tại theo format
    
    Returns:
        String thời gian hiện tại
    """
    return datetime.now().strftime(ATTENDANCE_TIME_FORMAT)


def save_attendance_record(name, confidence, date=None):
    """
    Lưu bản ghi điểm danh vào SQLite
    
    Args:
        name: Tên người điểm danh
        confidence: Độ tin cậy
        date: Ngày điểm danh (optional, mặc định là hôm nay)
    """
    if date is None:
        date = get_current_date()
    time = get_current_time()
    ensure_directory(ATTENDANCE_DIR)
    conn = sqlite3.connect(ATTENDANCE_DB_PATH)
    try:
        cur = conn.cursor()
        # Lấy thông tin SV nếu có
        sid, cls = get_student_info(name)
        cur.execute(
            "INSERT INTO attendance (name, date, time, confidence, student_id, class) VALUES (?,?,?,?,?,?)",
            (name, date, time, float(confidence), sid, cls)
        )
        conn.commit()
    finally:
        conn.close()

    # Tra cứu thông tin sinh viên (nếu có)
    student_id, class_name = get_student_info(name)
    # Ghi thêm vào CSV theo ngày (attendance_YYYY-MM-DD.csv)
    try:
        daily_csv_path = os.path.join(ATTENDANCE_DIR, f"attendance_{date}.csv")
        file_exists = os.path.exists(daily_csv_path)
        with open(daily_csv_path, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["Tên", "Mã SV", "Lớp", "Ngày", "Thời gian", "Độ tin cậy"]
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "Tên": name,
                "Mã SV": student_id or "",
                "Lớp": class_name or "",
                "Ngày": date,
                "Thời gian": time,
                "Độ tin cậy": f"{float(confidence):.4f}"
            })
    except Exception:
        # Không làm gián đoạn luồng chính nếu lỗi ghi CSV
        pass


def upsert_student_info(name, student_id=None, class_name=None):
    """
    Thêm/cập nhật thông tin sinh viên vào bảng students
    """
    ensure_directory(ATTENDANCE_DIR)
    conn = sqlite3.connect(ATTENDANCE_DB_PATH)
    try:
        cur = conn.cursor()
        student_id_clean = student_id.strip() if isinstance(student_id, str) else student_id
        if student_id_clean == "":
            student_id_clean = None
        if student_id_clean:
            cur.execute(
                """
                SELECT name FROM students
                WHERE student_id=? AND (name IS NULL OR name<>?)
                LIMIT 1
                """,
                (student_id_clean, name)
            )
            conflict = cur.fetchone()
            if conflict and conflict[0]:
                raise ValueError(f"Mã sinh viên '{student_id_clean}' đã tồn tại cho sinh viên '{conflict[0]}'")
        class_clean = class_name.strip() if isinstance(class_name, str) else class_name
        if class_clean == "":
            class_clean = None
        cur.execute(
            """
            INSERT INTO students(name, student_id, class)
            VALUES(?,?,?)
            ON CONFLICT(name) DO UPDATE SET
                student_id=excluded.student_id,
                class=excluded.class
            """,
            (name, student_id_clean, class_clean)
        )
        conn.commit()
    finally:
        conn.close()


def find_student_by_student_id(student_id):
    """
    Tìm sinh viên theo mã SV. Trả về tuple (name, class) hoặc None nếu không tìm thấy.
    """
    if not student_id:
        return None
    lookup_id = student_id.strip() if isinstance(student_id, str) else str(student_id).strip()
    if not lookup_id:
        return None
    ensure_directory(ATTENDANCE_DIR)
    if not os.path.exists(ATTENDANCE_DB_PATH):
        return None
    conn = sqlite3.connect(ATTENDANCE_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name, class FROM students WHERE student_id=? LIMIT 1",
            (lookup_id,)
        )
        row = cur.fetchone()
        if row:
            return row[0], row[1]
        return None
    finally:
        conn.close()


def student_id_exists(student_id, exclude_name=None):
    """
    Kiểm tra xem mã SV đã tồn tại cho sinh viên khác hay chưa.
    Trả về (exists, owner_name).
    """
    if not student_id:
        return False, None
    info = find_student_by_student_id(student_id)
    if not info:
        return False, None
    owner_name, _ = info
    if exclude_name and owner_name == exclude_name:
        return False, owner_name
    return True, owner_name


def get_student_info(name):
    """
    Lấy thông tin (student_id, class) theo name. Trả về (None, None) nếu không có.
    """
    ensure_directory(ATTENDANCE_DIR)
    if not os.path.exists(ATTENDANCE_DB_PATH):
        return None, None
    conn = sqlite3.connect(ATTENDANCE_DB_PATH)
    try:
        cur = conn.cursor()
        # 1) Thử khớp chính xác theo name
        cur.execute("SELECT student_id, class FROM students WHERE name=? LIMIT 1", (name,))
        row = cur.fetchone()
        if row:
            return row[0], row[1]
        # 2) Fallback: tên trong attendance có dạng 'ten_masv' (tên thư mục)
        #    Thử tách lấy mã SV ở phần sau dấu '_' cuối cùng và tra lớp theo tên gốc
        try:
            if '_' in name:
                base_name, sid = name.rsplit('_', 1)
                # Lấy lớp theo tên gốc nếu có
                cur.execute("SELECT class FROM students WHERE name=? LIMIT 1", (base_name,))
                cls_row = cur.fetchone()
                cls = cls_row[0] if cls_row else None
                # Nếu sid rỗng hoặc 'unknown' thì bỏ qua
                if sid and sid.lower() != 'unknown':
                    return sid, cls
                # Nếu không có sid hợp lệ, vẫn trả về lớp nếu có
                if cls:
                    return None, cls
        except Exception:
            pass
        return None, None
    finally:
        conn.close()


def list_all_students():
    """
    Lấy danh sách tất cả sinh viên từ bảng students.
    Trả về list các tuple (name, student_id, class)
    """
    ensure_directory(ATTENDANCE_DIR)
    if not os.path.exists(ATTENDANCE_DB_PATH):
        return []
    conn = sqlite3.connect(ATTENDANCE_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name, student_id, class FROM students ORDER BY name ASC")
        return cur.fetchall()
    finally:
        conn.close()


def delete_student(name):
    """
    Xóa sinh viên khỏi bảng students theo name.
    """
    ensure_directory(ATTENDANCE_DIR)
    conn = sqlite3.connect(ATTENDANCE_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM students WHERE name=?", (name,))
        conn.commit()
    finally:
        conn.close()


def load_attendance_records(date=None):
    """
    Load bản ghi điểm danh từ SQLite
    
    Args:
        date: Ngày cần load (optional, mặc định là hôm nay)
        
    Returns:
        List các bản ghi điểm danh với keys: 'Tên','Mã SV','Lớp','Ngày','Thời gian','Độ tin cậy'
    """
    if date is None:
        date = get_current_date()
    ensure_directory(ATTENDANCE_DIR)
    if not os.path.exists(ATTENDANCE_DB_PATH):
        return []
    conn = sqlite3.connect(ATTENDANCE_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name, student_id, class, date, time, confidence FROM attendance WHERE date = ? ORDER BY time ASC",
            (date,)
        )
        rows = cur.fetchall()
        records = []
        for name, sid, cls, d, t, conf in rows:
            # Nếu thiếu Mã SV/Lớp trong DB, thử suy ra từ students hoặc quy ước tên
            disp_sid = sid or None
            disp_cls = cls or None
            disp_name = name
            if not disp_sid or not disp_cls:
                try:
                    gs_sid, gs_cls = get_student_info(name)
                    if not disp_sid and gs_sid:
                        disp_sid = gs_sid
                    if not disp_cls and gs_cls:
                        disp_cls = gs_cls
                    # Nếu suy ra được từ quy ước 'ten_masv' thì hiển thị chỉ tên gốc
                    if '_' in name and (gs_sid or gs_cls):
                        try:
                            base_name, _ = name.rsplit('_', 1)
                            if base_name:
                                disp_name = base_name
                        except Exception:
                            pass
                except Exception:
                    pass
            records.append({
                'Tên': disp_name,
                'Mã SV': disp_sid or '',
                'Lớp': disp_cls or '',
                'Ngày': d,
                'Thời gian': t,
                'Độ tin cậy': f"{float(conf):.4f}"
            })
        return records
    finally:
        conn.close()


def get_attendance_summary(date=None):
    """
    Lấy tóm tắt điểm danh
    
    Args:
        date: Ngày cần tóm tắt (optional)
        
    Returns:
        Dictionary với thống kê
    """

    
def _row_exists(cur, name, date, time):
    """
    Kiểm tra bản ghi đã tồn tại trong DB chưa (theo name+date+time)
    """
    cur.execute(
        "SELECT 1 FROM attendance WHERE name=? AND date=? AND time=? LIMIT 1",
        (name, date, time)
    )
    return cur.fetchone() is not None


def _parse_conf(value):
    try:
        return float(str(value).replace('%', '').strip())
    except Exception:
        return 0.0


def migrate_attendance_csv_to_db(force=False):
    """
    Nhập dữ liệu điểm danh từ CSV trong `ATTENDANCE_DIR` vào SQLite.
    - Hỗ trợ cả file theo ngày `attendance_YYYY-MM-DD.csv` (không có cột Ngày)
      và `attendance_history.csv` (có cột Ngày).
    - Idempotent: bỏ qua nếu đã có marker hoặc nếu bản ghi đã tồn tại.
    
    Args:
        force: True để ép chạy lại migration dù đã có marker.
    """
    ensure_directory(ATTENDANCE_DIR)
    marker_path = os.path.join(ATTENDANCE_DIR, '.sqlite_migrated')
    if os.path.exists(marker_path) and not force:
        return

    conn = sqlite3.connect(ATTENDANCE_DB_PATH)
    try:
        cur = conn.cursor()
        # Đảm bảo bảng tồn tại
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                confidence REAL NOT NULL
            )
            """
        )
        conn.commit()

        # Duyệt tất cả file CSV trong thư mục
        for fname in os.listdir(ATTENDANCE_DIR):
            if not fname.lower().endswith('.csv'):
                continue
            fpath = os.path.join(ATTENDANCE_DIR, fname)
            # Xác định date mặc định từ tên file nếu là attendance_YYYY-MM-DD.csv
            default_date = None
            if fname.startswith('attendance_') and fname.endswith('.csv') and fname != 'attendance_history.csv':
                try:
                    default_date = fname.replace('attendance_', '').replace('.csv', '')
                except Exception:
                    default_date = None

            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Cột tên có thể là 'Tên' hoặc 'name'
                        name = row.get('Tên') or row.get('name') or ''
                        if not name:
                            continue
                        # Ngày có thể thiếu ở file theo ngày
                        date = row.get('Ngày') or row.get('date') or default_date
                        if not date:
                            continue
                        # Thời gian
                        time = row.get('Thời gian') or row.get('time') or ''
                        if not time:
                            continue
                        # Độ tin cậy
                        conf_str = row.get('Độ tin cậy') or row.get('confidence') or '0'
                        confidence = _parse_conf(conf_str)

                        # Bỏ qua nếu đã có
                        if _row_exists(cur, name, date, time):
                            continue
                        cur.execute(
                            "INSERT INTO attendance (name, date, time, confidence) VALUES (?,?,?,?)",
                            (name, date, time, float(confidence))
                        )
                conn.commit()
            except Exception:
                # Bỏ qua file lỗi đọc
                continue

        # Tạo marker
        try:
            with open(marker_path, 'w', encoding='utf-8') as mf:
                mf.write('migrated')
        except Exception:
            pass
    finally:
        conn.close()
    records = load_attendance_records(date)
    
    if not records:
        return {
            'total': 0,
            'unique_persons': 0,
            'persons': []
        }
    
    # Đếm số người unique
    unique_persons = set(record['Tên'] for record in records)
    
    return {
        'total': len(records),
        'unique_persons': len(unique_persons),
        'persons': list(unique_persons)
    }


def save_json(data, file_path):
    """
    Lưu dữ liệu vào file JSON
    
    Args:
        data: Dữ liệu cần lưu
        file_path: Đường dẫn file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(file_path):
    """
    Load dữ liệu từ file JSON
    
    Args:
        file_path: Đường dẫn file
        
    Returns:
        Dữ liệu đã load
    """
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_confidence(confidence):
    """
    Format độ tin cậy thành phần trăm
    
    Args:
        confidence: Độ tin cậy (0-1)
        
    Returns:
        String phần trăm
    """
    return f"{confidence * 100:.2f}%"


def get_person_list():
    """
    Lấy danh sách tất cả người trong dataset
    
    Returns:
        List tên người
    """
    from config import DATASET_DIR
    
    if not os.path.exists(DATASET_DIR):
        return []
    
    persons = [d for d in os.listdir(DATASET_DIR)
               if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    return sorted(persons)


def count_person_images(person_name):
    """
    Đếm số ảnh của một người
    
    Args:
        person_name: Tên người
        
    Returns:
        Số lượng ảnh
    """
    from config import DATASET_DIR
    
    person_dir = os.path.join(DATASET_DIR, person_name)
    
    if not os.path.exists(person_dir):
        return 0
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(person_dir)
              if f.lower().endswith(image_extensions)]
    
    return len(images)
