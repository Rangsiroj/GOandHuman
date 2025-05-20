import cv2
import cv2.aruco as aruco
import numpy as np
import time
from board_mapper import get_board_position
from gnugo_text_game import GNUGo

def auto_adjust_brightness(gray_image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)

def board_to_pixel(position):
    if len(position) < 2:
        return (0, 0)
    col = ord(position[0].upper()) - ord('A')
    row = 19 - int(position[1:])
    x = int((col / 18) * 500)
    y = int((row / 18) * 500)
    return (x, y)

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
        print("📷 เริ่มแสดงกล้องสดพร้อมตรวจจับ ArUco (กด ESC เพื่อออก)")

        cv2.namedWindow("ArUco Detection")
        cv2.createTrackbar('Brightness', "ArUco Detection", 50, 100, lambda x: None)
        cv2.createTrackbar('Contrast', "ArUco Detection", 50, 100, lambda x: None)
        cv2.createTrackbar('White Threshold', "ArUco Detection", 206, 255, lambda x: None)
        cv2.createTrackbar('Black Threshold', "ArUco Detection", 107, 255, lambda x: None)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ ไม่สามารถดึงภาพจากกล้องได้")
                break

            self.frame_count += 1
            if self.frame_count % 10 != 0:
                continue

            brightness = cv2.getTrackbarPos('Brightness', "ArUco Detection") - 50
            contrast = cv2.getTrackbarPos('Contrast', "ArUco Detection") / 50
            white_thresh = cv2.getTrackbarPos('White Threshold', "ArUco Detection")
            black_thresh = cv2.getTrackbarPos('Black Threshold', "ArUco Detection")

            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
            frame_copy = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if not self.is_camera_stable(gray):
                # print("📸 กล้องกำลังขยับ... รอให้หยุดก่อนตรวจจับหมาก")
                cv2.imshow("ArUco Detection", frame_copy)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

            if ids is not None:
                ids = ids.flatten()
                marker_positions = {}

                for i, marker_id in enumerate(ids):
                    if marker_id in [0, 1, 2, 3]:
                        aruco.drawDetectedMarkers(frame_copy, corners, ids)
                        corner = corners[i][0]
                        top_left = corner[0].astype(int)
                        marker_positions[marker_id] = corner.mean(axis=0)
                        cv2.putText(frame_copy, f"ID {marker_id}", tuple(top_left), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if len(marker_positions) == 4:
                    try:
                        src_pts = np.float32([
                            marker_positions[0],
                            marker_positions[1],
                            marker_positions[2],
                            marker_positions[3],
                        ])
                        dst_pts = np.float32([[0, 0], [500, 0], [500, 500], [0, 500]])
                        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                        warped = cv2.warpPerspective(frame, matrix, (500, 500))

                        stone_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                        enhanced = auto_adjust_brightness(stone_gray)
                        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

                        BW_black = cv2.threshold(blurred, black_thresh, 255, cv2.THRESH_BINARY_INV)[1]
                        BW_white = cv2.threshold(blurred, white_thresh, 255, cv2.THRESH_BINARY)[1]

                        kernel = np.ones((5, 5), np.uint8)
                        BW_black = cv2.morphologyEx(BW_black, cv2.MORPH_OPEN, kernel)
                        BW_white = cv2.morphologyEx(BW_white, cv2.MORPH_OPEN, kernel)

                        captured_positions = []
                        previous_board_state = self.board_state.copy()

                        for mask, color in [(BW_white, "white"), (BW_black, "black")]:
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            detected_positions = set()

                            for cnt in contours:
                                (x, y), r = cv2.minEnclosingCircle(cnt)
                                area = cv2.contourArea(cnt)
                                perimeter = cv2.arcLength(cnt, True)
                                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

                                if 6 <= r <= 15 and 0.7 <= circularity <= 1.2 and 50 <= area <= 500:
                                    board_pos = get_board_position(int(x), int(y))
                                    if board_pos:
                                        detected_positions.add(board_pos)

                            previous_positions = {pos for pos, c in self.board_state.items() if c == color}
                            diff = detected_positions - previous_positions

                            if color == self.current_turn and len(diff) == 1:
                                board_pos = diff.pop()
                                self.board_state[board_pos] = color
                                print(f"✅ {color.upper()} เดินที่ {board_pos}")
                                self.gnugo.play_move(color, board_pos)

                                if color == 'black':
                                    ai_move = self.gnugo.genmove('white')
                                    print(f"🤖 AI (WHITE) เดินที่: {ai_move}")
                                    self.board_state[ai_move] = 'white'

                                    captured_positions = [pos for pos in previous_board_state
                                                          if pos not in self.board_state and previous_board_state[pos] != 'white']
                                    if captured_positions:
                                        print(f"💥 จับกินที่: {', '.join(captured_positions)}")

                                    self.last_board_count = len(self.board_state)
                                    time.sleep(0.5)

                                self.current_turn = 'black'

                        for pos in captured_positions:
                            px, py = board_to_pixel(pos)
                            cv2.circle(warped, (px, py), 15, (0, 0, 255), 2)

                        score = self.gnugo.send_command("estimate_score")
                        cv2.putText(warped, f"Score: {score}", (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                        cv2.imshow("Perspective View", warped)

                    except Exception as e:
                        print(f"⚠️ Transform Error: {e}")

            cv2.imshow("ArUco Detection", frame_copy)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.release()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.gnugo.quit()
        print("🛑 ปิดกล้องและ AI เรียบร้อยแล้ว")