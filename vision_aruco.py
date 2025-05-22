import cv2
import numpy as np
import time
import cv2.aruco as aruco
from board_mapper_aruco import get_board_position
# from board_mapper import draw_board_grid
from gnugo_text_game import GNUGo

def auto_adjust_brightness(gray_image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)

def board_to_pixel(position):
    if len(position) < 2:
        return (0, 0)
    col = ord(position[0].upper()) - ord('A')
    if col >= 8:
        col -= 1
    row = 19 - int(position[1:])
    x = int((col / 18) * 500)
    y = int((row / 18) * 500)
    return (x, y)

class VisionSystem:
    def __init__(self, url='http://10.151.31.60:4747/video'):
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

        self.gnugo = GNUGo()
        self.gnugo.clear_board()

        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()

        # 🟢 สร้างหน้าต่างและ trackbar สำหรับปรับค่าพารามิเตอร์
        cv2.namedWindow("Perspective View")
        cv2.createTrackbar('Brightness', "Perspective View", 38, 100, lambda x: None)
        cv2.createTrackbar('Contrast', "Perspective View", 41, 100, lambda x: None)
        cv2.createTrackbar('White Threshold', "Perspective View", 240, 255, lambda x: None)
        cv2.createTrackbar('Black Threshold', "Perspective View", 63, 255, lambda x: None)

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
        print("📷 เริ่มตรวจจับกระดานด้วย ArUco (ESC เพื่อออก)")
        cv2.namedWindow("Aruco Detection")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ ไม่สามารถดึงภาพจากกล้องได้")
                break

            self.frame_count += 1
            if self.frame_count % 10 != 0:
                continue

            # ดึงค่าจาก trackbar
            brightness = cv2.getTrackbarPos('Brightness', "Perspective View") - 50
            contrast = cv2.getTrackbarPos('Contrast', "Perspective View") / 50
            white_thresh = cv2.getTrackbarPos('White Threshold', "Perspective View")
            black_thresh = cv2.getTrackbarPos('Black Threshold', "Perspective View")

            # ปรับภาพ
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if not self.is_camera_stable(gray):
                cv2.imshow("Aruco Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
            if ids is not None and len(ids) >= 4:
                ids = ids.flatten()
                marker_positions = {}
                for i, marker_id in enumerate(ids):
                    if marker_id in [0, 1, 2, 3]:
                        marker_positions[marker_id] = corners[i][0].mean(axis=0)

                if len(marker_positions) == 4:
                    try:
                        src_pts = np.float32([
                            marker_positions[0],
                            marker_positions[1],
                            marker_positions[2],
                            marker_positions[3]
                        ])
                        dst_pts = np.float32([[0, 0], [500, 0], [500, 500], [0, 500]])
                        # MARGIN = -25  # ตัดขอบ 20 พิกเซลจากทุกด้าน
                        # PIXEL = 500

                        # dst_pts = np.float32([
                        #     [MARGIN, MARGIN],              # Top-left
                        #     [PIXEL - MARGIN, MARGIN],        # Top-right
                        #     [PIXEL - MARGIN, PIXEL - MARGIN],  # Bottom-right
                        #     [MARGIN, PIXEL - MARGIN]         # Bottom-left
                        # ])


                        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                        warped = cv2.warpPerspective(frame, matrix, (500, 500))

                        MARGIN = 20
                        cropped = warped[MARGIN:500 - MARGIN, MARGIN:500 - MARGIN]

                        # ใช้ cropped สำหรับตรวจจับหมาก
                        enhanced_color = cropped.copy()
                        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                        enhanced = auto_adjust_brightness(gray)
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

                            if 8 <= r <= 14 and 0.85 <= circularity <= 1.1 and 80 <= area <= 400:
                                board_pos = get_board_position(int(x), int(y))
                                if board_pos:
                                    detected_positions.add(board_pos)

                                    # 🔍 แสดงตำแหน่งบนภาพ Perspective View เพื่อ debug
                                    cv2.putText(enhanced_color, f"{board_pos}", (int(x), int(y)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


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
                            cv2.circle(enhanced_color, (px, py), 15, (0, 0, 255), 2)

                        score = self.gnugo.send_command("estimate_score")
                        cv2.putText(enhanced_color, f"Score: {score}", (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        # draw_board_grid(enhanced_color)  # เพิ่มบรรทัดนี้ก่อนแสดงภาพ

                        cv2.imshow("Perspective View", enhanced_color)
                        cv2.imshow("Black Stones", BW_black)
                        # cv2.imshow("White Stones", BW_white)

                    except Exception as e:
                        print(f"⚠️ Transform Error: {e}")

            aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.imshow("Aruco Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.release()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.gnugo.quit()
        print("🔕 ปิดกล้องและ AI เรียบร้อยแล้ว")

# เรียกใช้งาน
if __name__ == "__main__":
    system = VisionSystem()
    system.run()