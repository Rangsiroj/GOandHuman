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
        self.motion_cooldown = 1.0
        self.has_warned_motion = False  # ใช้เพื่อป้องกัน log ซ้ำ

        self.gnugo = GNUGo()
        self.gnugo.clear_board()

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
        print("📷 เริ่มคลิกเลือก 4 มุมเพื่อทำ Perspective Transform (ESC เพื่อออก)")

        cv2.namedWindow("Manual Detection")
        cv2.setMouseCallback("Manual Detection", select_point)
        cv2.createTrackbar('Brightness', "Manual Detection", 38, 100, lambda x: None)
        cv2.createTrackbar('Contrast', "Manual Detection", 41, 100, lambda x: None)
        cv2.createTrackbar('White Threshold', "Manual Detection", 240, 255, lambda x: None)
        cv2.createTrackbar('Black Threshold', "Manual Detection", 58, 255, lambda x: None)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ ไม่สามารถดึงภาพจากกล้องได้")
                break

            self.frame_count += 1
            if self.frame_count % 10 != 0:
                continue

            brightness = cv2.getTrackbarPos('Brightness', "Manual Detection") - 50
            contrast = cv2.getTrackbarPos('Contrast', "Manual Detection") / 50
            white_thresh = cv2.getTrackbarPos('White Threshold', "Manual Detection")
            black_thresh = cv2.getTrackbarPos('Black Threshold', "Manual Detection")

            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
            frame_copy = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if not self.is_camera_stable(gray):
                if not self.has_warned_motion:
                    print("📸 กล้องกำลังขยับ... รอให้หยุดก่อนตรวจจับหมาก")
                    self.has_warned_motion = True
                cv2.imshow("Manual Detection", frame_copy)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            self.has_warned_motion = False  # รีเซ็ตเมื่อกล้องนิ่ง

            # ... (ส่วนตรวจจับหมากตามที่มีอยู่เดิม)

            cv2.imshow("Manual Detection", frame_copy)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.release()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.gnugo.quit()
        print("🔕 ปิดกล้องและ AI เรียบร้อยแล้ว")
