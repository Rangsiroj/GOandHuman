import cv2
import cv2.aruco as aruco
import numpy as np
import time
from board_mapper import get_board_position
from gnugo_text_game import GNUGo

# ... (other functions stay the same)

class VisionSystem:
    def __init__(self, url='http://172.23.32.136:4747/video'):
        self.cap = cv2.VideoCapture(url)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        self.board_state = {}
        self.current_turn = 'black'
        self.last_board_count = 0
        self.frame_count = 0
        self.prev_gray = None
        self.last_motion_time = time.time()
        self.motion_cooldown = 1.0
        self.gnugo = GNUGo()
        self.gnugo.clear_board()
        self.has_warned_motion = False  # <== ADD: state flag for printing once

        if not self.cap.isOpened():
            print("❌ ไม่สามารถเชื่อมต่อกล้องได้")
        else:
            print("✅ เชื่อมต่อกล้องสำเร็จ")

    def is_camera_stable(self, gray, threshold=500000):
        now = time.time()
        if self.prev_gray is None:
            self.prev_gray = gray
            self.last_motion_time = now
            return True
        diff = cv2.absdiff(self.prev_gray, gray)
        score = np.sum(diff)
        self.prev_gray = gray
        if score > threshold:
            self.last_motion_time = now
            return False
        return (now - self.last_motion_time) > self.motion_cooldown

    def run(self):
        # ... (initialization of window and trackbars)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ ไม่สามารถดึงภาพจากกล้องได้")
                break

            self.frame_count += 1
            if self.frame_count % 10 != 0:
                continue

            # ... (process brightness/contrast/thresholds)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if not self.is_camera_stable(gray):
                if not self.has_warned_motion:
                    print("📸 กล้องกำลังขยับ... รอให้หยุดก่อนตรวจจับหมาก")
                    self.has_warned_motion = True
                # ... show window and wait
                continue

            self.has_warned_motion = False  # <== RESET warning state when stable

            # ... (rest of detection logic as-is)

        self.release()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.gnugo.quit()
        print("🛑 ปิดกล้องและ AI เรียบร้อยแล้ว")
