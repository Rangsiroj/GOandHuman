
# -----------------------------
# VisionManual: ระบบตรวจจับหมากแบบ Manual
# -----------------------------

import cv2
import numpy as np
import time
from board_mapper_manual import get_board_position  # ฟังก์ชันแปลงพิกัด pixel เป็นตำแหน่งบนกระดาน
from gnugo_text_game import GNUGo  # คลาสสำหรับควบคุม AI โกะ

# manual_pts: เก็บจุดที่ผู้ใช้คลิกเลือก 4 มุมกระดาน
manual_pts = []

# ปรับความสว่างของภาพด้วย CLAHE เพื่อเพิ่ม contrast
def auto_adjust_brightness(gray_image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)

# ฟังก์ชัน callback สำหรับการคลิกเลือกจุดบนภาพ
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(manual_pts) < 4:
        manual_pts.append([x, y])
        print(f"📍 เลือกจุดที่ {len(manual_pts)}: ({x}, {y})")

# แปลงตำแหน่งบนกระดาน (เช่น 'A19') เป็นพิกัด pixel สำหรับแสดงผล
def board_to_pixel(position):
    if len(position) < 2:
        return (0, 0)
    col = ord(position[0].upper()) - ord('A')
    row = 19 - int(position[1:])
    x = int((col / 18) * 500)
    y = int((row / 18) * 500)
    return (x, y)

# คลาสหลักสำหรับระบบ Vision แบบ Manual
class VisionManual:
    def __init__(self, url='http://172.23.36.213:4747/video'):
        # เปิดกล้องจาก URL ที่กำหนด
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            print("❌ ไม่สามารถเชื่อมต่อกล้องได้")
        else:
            print("✅ เชื่อมต่อกล้องสำเร็จ")
        # สถานะกระดาน, สีที่เล่น, จำนวนหมาก, ฯลฯ
        self.board_state = {}
        self.current_turn = 'black'
        self.last_board_count = 0
        self.frame_count = 0
        self.prev_gray = None
        self.last_motion_time = time.time()
        self.motion_cooldown = 1.0

        # สร้างอ็อบเจ็กต์ AI โกะ และเคลียร์กระดาน
        self.gnugo = GNUGo()
        self.gnugo.clear_board()

    # ตรวจสอบว่ากล้องนิ่งหรือไม่ โดยเปรียบเทียบภาพก่อนหน้า
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

    # ฟังก์ชันหลักสำหรับรันระบบตรวจจับหมากแบบ Manual
    def run(self):
        print("📷 เริ่มคลิกเลือก 4 มุมเพื่อทำ Perspective Transform (ESC เพื่อออก)")

        # สร้างหน้าต่างและ trackbar สำหรับปรับ brightness, contrast, threshold
        cv2.namedWindow("Manual Detection")
        cv2.setMouseCallback("Manual Detection", select_point)
        cv2.createTrackbar('Brightness', "Manual Detection", 76, 100, lambda x: None)
        cv2.createTrackbar('Contrast', "Manual Detection", 47, 100, lambda x: None)
        cv2.createTrackbar('White Threshold', "Manual Detection", 252, 255, lambda x: None)
        cv2.createTrackbar('Black Threshold', "Manual Detection", 72, 255, lambda x: None)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ ไม่สามารถดึงภาพจากกล้องได้")
                break

            self.frame_count += 1
            if self.frame_count % 10 != 0:
                continue

            # อ่านค่าจาก trackbar เพื่อปรับภาพ
            brightness = cv2.getTrackbarPos('Brightness', "Manual Detection") - 50
            contrast = cv2.getTrackbarPos('Contrast', "Manual Detection") / 50
            white_thresh = cv2.getTrackbarPos('White Threshold', "Manual Detection")
            black_thresh = cv2.getTrackbarPos('Black Threshold', "Manual Detection")

            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
            frame_copy = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ตรวจสอบว่ากล้องนิ่งหรือไม่
            if not self.is_camera_stable(gray):
                # กล้องขยับ รอให้หยุดก่อนตรวจจับหมาก
                cv2.imshow("Manual Detection", frame_copy)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # ถ้าเลือกครบ 4 จุดแล้ว จะทำ Perspective Transform และตรวจจับหมาก
            if len(manual_pts) == 4:
                try:
                    src_pts = np.float32(manual_pts)
                    dst_pts = np.float32([[0, 0], [500, 0], [500, 500], [0, 500]])
                    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped = cv2.warpPerspective(frame, matrix, (500, 500))

                    enhanced_color = warped.copy()
                    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    enhanced = auto_adjust_brightness(gray)
                    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

                    # สร้าง mask สำหรับหมากดำและหมากขาว
                    BW_black = cv2.threshold(blurred, black_thresh, 255, cv2.THRESH_BINARY_INV)[1]
                    BW_white = cv2.threshold(blurred, white_thresh, 255, cv2.THRESH_BINARY)[1]

                    kernel = np.ones((5, 5), np.uint8)
                    BW_black = cv2.morphologyEx(BW_black, cv2.MORPH_OPEN, kernel)
                    BW_white = cv2.morphologyEx(BW_white, cv2.MORPH_OPEN, kernel)

                    captured_positions = []
                    previous_board_state = self.board_state.copy()

                    # ตรวจจับหมากขาวและดำในภาพ
                    for mask, color in [(BW_white, "white"), (BW_black, "black")]:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        detected_positions = set()

                        for cnt in contours:
                            (x, y), r = cv2.minEnclosingCircle(cnt)
                            area = cv2.contourArea(cnt)
                            perimeter = cv2.arcLength(cnt, True)
                            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

                            # เงื่อนไขสำหรับตรวจจับหมาก
                            if 6 <= r <= 15 and 0.7 <= circularity <= 1.2 and 50 <= area <= 500:
                                board_pos = get_board_position(int(x), int(y))
                                if board_pos:
                                    detected_positions.add(board_pos)

                        previous_positions = {pos for pos, c in self.board_state.items() if c == color}
                        diff = detected_positions - previous_positions

                        # ถ้าเป็นตาของผู้เล่นและมีหมากใหม่ 1 ตำแหน่ง
                        if color == self.current_turn and len(diff) == 1:
                            board_pos = diff.pop()
                            self.board_state[board_pos] = color
                            print(f"✅ {color.upper()} เดินที่ {board_pos}")
                            self.gnugo.play_move(color, board_pos)

                            if color == 'black':
                                # ให้ AI เดินหมากขาว
                                ai_move = self.gnugo.genmove('white')
                                print(f"🤖 AI (WHITE) เดินที่: {ai_move}")
                                self.board_state[ai_move] = 'white'

                                # ตรวจจับตำแหน่งที่ถูกจับกิน
                                captured_positions = [pos for pos in previous_board_state
                                                      if pos not in self.board_state and previous_board_state[pos] != 'white']
                                if captured_positions:
                                    print(f"💥 จับกินที่: {', '.join(captured_positions)}")

                                self.last_board_count = len(self.board_state)
                                time.sleep(0.5)

                            self.current_turn = 'black'

                    # วาดวงกลมแสดงตำแหน่งที่ถูกจับกิน
                    for pos in captured_positions:
                        px, py = board_to_pixel(pos)
                        cv2.circle(enhanced_color, (px, py), 15, (0, 0, 255), 2)

                    # แสดงคะแนนที่ประเมินโดย AI
                    score = self.gnugo.send_command("estimate_score")
                    cv2.putText(enhanced_color, f"Score: {score}", (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    # แสดงผลภาพต่าง ๆ
                    cv2.imshow("Perspective View", enhanced_color)
                    cv2.imshow("Black Stones", BW_black)
                    cv2.imshow("White Stones", BW_white)

                except Exception as e:
                    print(f"⚠️ Transform Error: {e}")
            else:
                # วาดจุดที่เลือกบนภาพและแสดงข้อความแนะนำ
                for pt in manual_pts:
                    cv2.circle(frame_copy, tuple(pt), 5, (0, 255, 255), -1)
                cv2.putText(frame_copy, "คลิกเลือก 4 มุมกระดาน", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Manual Detection", frame_copy)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.release()

    # ปิดกล้องและหน้าต่างทั้งหมดเมื่อจบการทำงาน
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.gnugo.quit()
        print("🔕 ปิดกล้องและ AI เรียบร้อยแล้ว")
