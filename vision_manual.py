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

class VisionManual:
    def __init__(self, url='http://172.23.34.65:4747/video'):
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

        self.gnugo = GNUGo()
        self.gnugo.clear_board()

    def is_camera_stable(self, gray, threshold=1000000):
        if self.prev_gray is None:
            self.prev_gray = gray
            return True
        diff = cv2.absdiff(self.prev_gray, gray)
        score = np.sum(diff)
        self.prev_gray = gray
        return score < threshold

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
                print("📸 กล้องกำลังขยับ... รอให้หยุดก่อนตรวจจับหมาก")
                cv2.imshow("Manual Detection", frame_copy)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

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

                    BW_black = cv2.threshold(blurred, black_thresh, 255, cv2.THRESH_BINARY_INV)[1]
                    BW_white = cv2.threshold(blurred, white_thresh, 255, cv2.THRESH_BINARY)[1]

                    kernel = np.ones((5, 5), np.uint8)
                    BW_black = cv2.morphologyEx(BW_black, cv2.MORPH_OPEN, kernel)
                    BW_white = cv2.morphologyEx(BW_white, cv2.MORPH_OPEN, kernel)

                    cv2.imshow("Perspective View", enhanced_color)
                    cv2.imshow("Black Stones", BW_black)
                    cv2.imshow("White Stones", BW_white)

                    for mask, color in [(BW_white, "white"), (BW_black, "black")]:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            (x, y), r = cv2.minEnclosingCircle(cnt)
                            area = cv2.contourArea(cnt)
                            perimeter = cv2.arcLength(cnt, True)
                            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

                            if 6 <= r <= 15 and 0.7 <= circularity <= 1.2 and 50 <= area <= 500:
                                board_pos = get_board_position(int(x), int(y))
                                if board_pos and board_pos not in self.board_state:
                                    if color == self.current_turn:
                                        self.board_state[board_pos] = color
                                        print(f"✅ {color.upper()} เดินที่ {board_pos}")
                                        self.gnugo.play_move(color, board_pos)

                                        if color == 'black' and len(self.board_state) > self.last_board_count:
                                            ai_move = self.gnugo.genmove('white')
                                            print(f"🤖 AI (WHITE) เดินที่: {ai_move}")
                                            self.board_state[ai_move] = 'white'
                                            self.last_board_count = len(self.board_state)
                                            time.sleep(0.5)

                                        self.current_turn = 'black'
                                elif board_pos in self.board_state:
                                    pass

                except Exception as e:
                    print(f"⚠️ Transform Error: {e}")
            else:
                for pt in manual_pts:
                    cv2.circle(frame_copy, tuple(pt), 5, (0, 255, 255), -1)
                cv2.putText(frame_copy, "คลิกเลือก 4 มุมกระดาน", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Manual Detection", frame_copy)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.release()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.gnugo.quit()
        print("🔕 ปิดกล้องและ AI เรียบร้อยแล้ว")
