import cv2
import numpy as np
import time
from board_mapper import get_board_position
from gnugo_text_game import GNUGo

manual_pts = []

def auto_adjust_brightness(gray_image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)

def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(manual_pts) < 4:
        manual_pts.append([x, y])
        print(f"📍 เลือกจุดที่ {len(manual_pts)}: ({x}, {y})")

def board_to_pixel(position):
    if len(position) < 2:
        return (0, 0)
    col = ord(position[0].upper()) - ord('A')
    row = 19 - int(position[1:])
    x = int((col / 18) * 500)
    y = int((row / 18) * 500)
    return (x, y)

class VisionManual:
    def __init__(self, url='http://172.23.32.136:4747/video'):
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            print("❌ ไม่สามารถเชื่อมต่อกล้องได้")
        else:
            print("✅ เชื่อมต่อกล้องสำเร็จ")
        self.board_state = {}
        self.current_turn = 'black'
        self.last_board_count = 0
        self.frame_count = 0
        self.prev_gray = None
        self.last_motion_time = time.time()
        self.motion_cooldown = 2.0  # เพิ่มเวลา wait หลังจากขยับ
        self.stability_threshold = 100000  # ลด sensitivity ลง

        self.gnugo = GNUGo()
        self.gnugo.clear_board()

    def is_camera_stable(self, gray):
        now = time.time()
        if self.prev_gray is None:
            self.prev_gray = gray
            self.last_motion_time = now
            return True
        diff = cv2.absdiff(self.prev_gray, gray)
        score = np.sum(diff)
        self.prev_gray = gray
        if score > self.stability_threshold:
            self.last_motion_time = now
            return False
        return (now - self.last_motion_time) > self.motion_cooldown

    # ... (rest of code remains unchanged)
