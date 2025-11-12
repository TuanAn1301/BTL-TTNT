"""
Thread nhận diện và điểm danh nhiều người cùng lúc (template matching, không cần huấn luyện)
"""
import os
import time
from datetime import datetime
from typing import Dict, Tuple, List

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage

from config import DATASET_DIR, ATTENDANCE_COOLDOWN, COLOR_GREEN, COLOR_RED, COLOR_WHITE


class MultiRecognitionThread(QThread):
    """
    Nhận diện nhiều khuôn mặt song song bằng template matching.
    Dựa trên `SimpleRecognitionThread` nhưng theo dõi trạng thái từng khuôn mặt độc lập.
    """
    frame_ready = pyqtSignal(QImage)
    attendance_marked = pyqtSignal(str, str)  # name, time
    error_occurred = pyqtSignal(str)

    def __init__(self, camera_index=0, face_detector=None):
        super().__init__()
        self.camera_index = camera_index
        self.face_detector = face_detector
        self.running = False
        self.camera = None
        # Templates dataset
        self.templates: Dict[str, List[np.ndarray]] = {}
        # Theo dõi điểm danh: dict name -> {"last_marked": datetime, "marked_today": bool}
        self.attendance_status: Dict[str, Dict] = {}
        # Theo dõi ổn định: dict track_id -> thông tin đối tượng
        self.required_presence_seconds = 3.0
        self.track_timeout_seconds = 2.0
        self.tracks: Dict[int, Dict] = {}
        self.next_track_id = 1
        # Load dữ liệu
        self.load_templates()
        
        # Thiết lập timer tự động làm mới sau 45 phút (2700 giây)
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.reset_attendance_status)
        self.refresh_timer.start(45 * 60 * 1000)  # 45 phút

    # -------------------- Data --------------------
    def load_templates(self):
        try:
            self.templates.clear()
            if not os.path.exists(DATASET_DIR):
                print("⚠️ Thư mục dataset không tồn tại")
                return
            persons = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
            from PIL import Image
            for person_name in persons:
                person_dir = os.path.join(DATASET_DIR, person_name)
                image_files = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                person_templates = []
                for img_file in image_files:
                    img_path = os.path.join(person_dir, img_file)
                    try:
                        pil_img = Image.open(img_path).convert('L')
                        img = np.array(pil_img)
                        img = cv2.resize(img, (100, 100))
                        person_templates.append(img)
                    except Exception as e:
                        print(f"Lỗi đọc ảnh {img_path}: {e}")
                if person_templates:
                    self.templates[person_name] = person_templates
            print(f"✓ Đã load template cho {len(self.templates)} người")
        except Exception as e:
            print(f"Lỗi load templates: {e}")

    def reload_templates(self):
        self.load_templates()

    # -------------------- Helpers --------------------
    def recognize_face(self, face_bgr: np.ndarray) -> Tuple[str, float]:
        if not self.templates:
            return None, 0.0
        gray_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (100, 100))
        best_name, best_score = None, 0.0
        for person_name, templates in self.templates.items():
            for template in templates:
                result = cv2.matchTemplate(gray_face, template, cv2.TM_CCOEFF_NORMED)
                score = float(result[0][0])
                if score > best_score:
                    best_score = score
                    best_name = person_name
        return (best_name, best_score) if best_score >= 0.5 else (None, best_score)

    @staticmethod
    def _center(bbox):
        x, y, w, h = bbox
        return (x + w / 2.0, y + h / 2.0)

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        inter_x1, inter_y1 = max(ax, bx), max(ay, by)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a, area_b = aw * ah, bw * bh
        union = area_a + area_b - inter_area + 1e-6
        return inter_area / union

    def _assign_tracks(self, detections: List[Tuple[int, int, int, int]]):
        # Gán bbox -> track id bằng IOU tối đa > 0.3, nếu không thì tạo track mới
        assigned = {}
        used_tracks = set()
        for det in detections:
            best_tid, best_iou = None, 0.0
            for tid, t in self.tracks.items():
                if tid in used_tracks:
                    continue
                iou = self._iou(det, t["bbox"]) if "bbox" in t else 0.0
                if iou > best_iou:
                    best_iou, best_tid = iou, tid
            if best_tid is not None and best_iou >= 0.3:
                assigned[det] = best_tid
                used_tracks.add(best_tid)
            else:
                tid = self.next_track_id
                self.next_track_id += 1
                self.tracks[tid] = {
                    "bbox": tuple(map(int, det)),
                    "since": None,
                    "name": None,
                    "last_seen": time.time()
                }
                assigned[det] = tid
        # Cập nhật bbox theo frame hiện tại
        for det, tid in assigned.items():
            track = self.tracks.get(tid)
            if track is not None:
                track["bbox"] = tuple(map(int, det))
                track["last_seen"] = time.time()
        return assigned

    def can_mark_attendance(self, name: str) -> bool:
        now = datetime.now()
        
        # Kiểm tra nếu là lần đầu điểm danh trong ngày
        today = now.strftime('%Y-%m-%d')
        
        if name not in self.attendance_status:
            self.attendance_status[name] = {
                'last_marked': now,
                'marked_today': False,
                'marked_date': today
            }
            return True
            
        # Nếu đã qua ngày mới, reset trạng thái
        if self.attendance_status[name].get('marked_date') != today:
            self.attendance_status[name] = {
                'last_marked': now,
                'marked_today': False,
                'marked_date': today
            }
            return True
            
        # Kiểm tra nếu đã điểm danh hôm nay
        if self.attendance_status[name]['marked_today']:
            return False
            
        # Kiểm tra thời gian chờ giữa các lần điểm danh
        last_marked = self.attendance_status[name]['last_marked']
        return (now - last_marked).total_seconds() >= ATTENDANCE_COOLDOWN

    def mark_attendance(self, name: str):
        from utils import save_attendance_record
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        
        # Lưu điểm danh
        save_attendance_record(name, 1.0)
        
        # Cập nhật trạng thái
        self.attendance_status[name] = {
            'last_marked': now,
            'marked_today': True,
            'marked_date': today
        }
        
        self.attendance_marked.emit(name, now.strftime('%H:%M:%S'))
        print(f"✓ Đã điểm danh cho {name} lúc {now.strftime('%H:%M:%S')}")
    
    def reset_attendance_status(self):
        """
        Đặt lại trạng thái điểm danh cho tất cả người dùng
        """
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        
        for name in list(self.attendance_status.keys()):
            # Chỉ reset nếu đã điểm danh trước đó trong ngày
            if self.attendance_status[name].get('marked_today', False):
                self.attendance_status[name] = {
                    'last_marked': now,
                    'marked_today': False,
                    'marked_date': today
                }
                print(f"✓ Đã reset trạng thái điểm danh cho {name} sau 45 phút")
        
        # Thông báo trên giao diện
        self.attendance_marked.emit("SYSTEM", "Đã làm mới danh sách điểm danh sau 45 phút")

    # -------------------- Main loop --------------------
    def run(self):
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                self.error_occurred.emit("Không thể mở camera")
                return
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.running = True
            frame_id = 0
            while self.running:
                ok, frame = self.camera.read()
                if not ok:
                    break
                display = frame.copy()
                frame_id += 1
                if self.face_detector and frame_id % 2 == 0:
                    faces_np = self.face_detector.detect_faces(frame)
                    # Chuẩn hóa bbox -> tuple[int,int,int,int]
                    faces = [tuple(map(int, f)) for f in list(faces_np)]
                    # Gán track cho bbox hiện tại bằng tuple (hashable)
                    assigned = self._assign_tracks(faces)
                    now_ts = time.time()
                    stale_cutoff = now_ts - self.track_timeout_seconds
                    for (x, y, w, h) in faces:
                        tid = assigned[(x, y, w, h)]
                        face_img = frame[y:y+h, x:x+w]
                        name, conf = self.recognize_face(face_img)
                        if name is not None and conf >= 0.5:
                            # Đánh dấu ổn định theo track id + name
                            if self.tracks[tid].get("name") != name:
                                self.tracks[tid]["name"] = name
                                self.tracks[tid]["since"] = now_ts
                            held = now_ts - (self.tracks[tid]["since"] or now_ts)
                            # Vẽ HUD
                            color = COLOR_GREEN
                            label = f"{name}"
                            conf_text = f"{conf:.0%}"
                            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                            cv2.rectangle(display, (x, y-60), (x+w, y), color, -1)
                            cv2.putText(display, label, (x+5, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
                            cv2.putText(display, conf_text, (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
                            cv2.putText(display, f"Giu yen: {held:.1f}/{self.required_presence_seconds:.0f}s", (x, max(0, y-70)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
                            if held >= self.required_presence_seconds and self.can_mark_attendance(name):
                                self.mark_attendance(name)
                        else:
                            # Unknown
                            self.tracks[tid]["name"] = None
                            self.tracks[tid]["since"] = None
                            color = COLOR_RED
                            label = "Unknown"
                            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                            cv2.rectangle(display, (x, y-30), (x+w, y), color, -1)
                            cv2.putText(display, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
                    # Loại bỏ track quá hạn xuất hiện
                    for tid in list(self.tracks.keys()):
                        track = self.tracks[tid]
                        if track.get("last_seen", now_ts) < stale_cutoff:
                            self.tracks.pop(tid, None)
                # Emit frame
                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                self.frame_ready.emit(qimg)
                self.msleep(30)
        except Exception as e:
            self.error_occurred.emit(f"Lỗi: {str(e)}")
        finally:
            self.running = False
            if self.camera is not None:
                self.camera.release()
                self.camera = None

    def stop(self):
        self.running = False
        if hasattr(self, 'refresh_timer') and self.refresh_timer.isActive():
            self.refresh_timer.stop()
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        from PyQt5.QtCore import QThread as _QThread
        if _QThread.currentThread() != self:
            self.quit()
            self.wait()
