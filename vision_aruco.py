import cv2
import numpy as np
import time
import cv2.aruco as aruco
from board_mapper_aruco import get_board_position
from gnugo_text_game import GNUGo
from game_logic import GameLogic
import os

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

def board_pos_to_xy(pos):
    # pos: เช่น 'A19', 'Q16', ...
    if len(pos) < 2:
        return None
    col_chr = pos[0].upper()
    row_str = pos[1:]
    if not row_str.isdigit():
        return None
    col = ord(col_chr) - ord('A')
    if col_chr >= 'I':
        col -= 1
    row = 19 - int(row_str)
    return (col, row)

class VisionSystem:
    def __init__(self, url='http://10.91.212.186:4747/video'):
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            print("❌ ไม่สามารถเชื่อมต่อกล้องได้")
        else:
            print("✅ เชื่อมต่อกล้องสำเร็จ")

        self.logic = GameLogic(GNUGo())
        self.logic.reset()

        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()

        cv2.namedWindow("Perspective View")
        cv2.createTrackbar('Brightness', "Perspective View", 94, 100, lambda x: None)
        cv2.createTrackbar('Contrast', "Perspective View", 87, 100, lambda x: None)
        cv2.createTrackbar('White Threshold', "Perspective View", 252, 255, lambda x: None)
        cv2.createTrackbar('Black Threshold', "Perspective View", 174, 255, lambda x: None)

        self.warned_illegal_move = False
        self.warned_occupied_positions = set()

    def is_camera_stable(self, gray, threshold=500000):
        now = time.time()
        if not hasattr(self, 'prev_gray'):
            self.prev_gray = gray
            self.last_motion_time = now
            return True
        diff = cv2.absdiff(self.prev_gray, gray)
        score = np.sum(diff)
        self.prev_gray = gray
        if score > threshold:
            self.last_motion_time = now
            return False
        return (now - self.last_motion_time) > 1.0

    def run(self):
        print("📷 เริ่มตรวจจับกระดานด้วย ArUco (ESC เพื่อออก)")
        cv2.namedWindow("Aruco Detection")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ ไม่สามารถดึงภาพจากกล้องได้")
                break
            if not hasattr(self, 'frame_count'):
                self.frame_count = 0
            self.frame_count += 1
            if self.frame_count % 5 != 0:
                continue
            brightness = cv2.getTrackbarPos('Brightness', "Perspective View") - 50
            contrast = cv2.getTrackbarPos('Contrast', "Perspective View") / 50
            white_thresh = cv2.getTrackbarPos('White Threshold', "Perspective View")
            black_thresh = cv2.getTrackbarPos('Black Threshold', "Perspective View")
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
                        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                        warped = cv2.warpPerspective(frame, matrix, (500, 500))
                        MARGIN = 25
                        cropped = warped[MARGIN:500 - MARGIN, MARGIN:500 - MARGIN]
                        enhanced_color = cropped.copy()
                        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                        enhanced = auto_adjust_brightness(gray)
                        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
                        BW_black = cv2.threshold(blurred, black_thresh, 255, cv2.THRESH_BINARY_INV)[1]
                        BW_white = cv2.threshold(blurred, white_thresh, 255, cv2.THRESH_BINARY)[1]
                        kernel = np.ones((5, 5), np.uint8)
                        BW_black = cv2.morphologyEx(BW_black, cv2.MORPH_OPEN, kernel)
                        BW_white = cv2.morphologyEx(BW_white, cv2.MORPH_OPEN, kernel)
                        captured_by_black = []
                        captured_by_white = []
                        previous_board_state = self.logic.logic.board_state.copy() if hasattr(self.logic, 'logic') else self.logic.board_state.copy()
                        for mask, color in [(BW_white, "white"), (BW_black, "black")]:
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            detected_positions = set()
                            for cnt in contours:
                                (x, y), r = cv2.minEnclosingCircle(cnt)
                                area = cv2.contourArea(cnt)
                                perimeter = cv2.arcLength(cnt, True)
                                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                                if 7 <= r <= 14 and 0.80 <= circularity <= 1.15 and 60 <= area <= 450:
                                    board_pos = get_board_position(int(x), int(y))
                                    if board_pos:
                                        detected_positions.add(board_pos)
                                        cv2.putText(enhanced_color, f"{board_pos}", (int(x), int(y)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            previous_positions = {pos for pos, c in self.logic.board_state.items() if c == color}
                            diff = detected_positions - previous_positions
                            if color == self.logic.current_turn and len(diff) == 1:
                                board_pos = diff.pop()
                                if board_pos in self.logic.board_state:
                                    if board_pos not in self.warned_occupied_positions:
                                        print(f"🚫 หมากตำแหน่ง {board_pos} ถูกวางไปแล้ว")
                                        self.warned_occupied_positions.add(board_pos)
                                    continue
                                ok, result = self.logic.play_move(color, board_pos)
                                if not ok:
                                    msg = None
                                    if "occupied" in result.lower():
                                        msg = f"🚫 ตำแหน่ง {board_pos} มีหมากอยู่แล้ว (วางซ้ำ)"
                                    elif "ko" in result.lower():
                                        msg = f"⚠️ ผิดกติกาโคะ (Ko rule) ที่ตำแหน่ง {board_pos}"
                                    elif "suicide" in result.lower():
                                        msg = f"⚠️ เดินที่ {board_pos} แล้วหมากจะถูกกินทันที (Suicide move)"
                                    else:
                                        msg = f"❌ หมากไม่ถูกต้อง ({result})"
                                    if not self.warned_illegal_move:
                                        print(msg)
                                        self.warned_illegal_move = True
                                    continue
                                self.warned_illegal_move = False
                                self.warned_occupied_positions.clear()
                                previous_board_state = self.logic.board_state.copy()
                                xy = board_pos_to_xy(board_pos)
                                if color == 'black':
                                    print(f"=== ตาที่ {self.logic.turn_number} ===")
                                    print(f"✅ BLACK เดินที่ {board_pos} (ตำแหน่ง X,Y = {xy[0]},{xy[1]})")
                                    ai_move, elapsed = self.logic.ai_move()
                                    if ai_move.strip().lower() == 'pass':
                                        print(f"🤖 AI (WHITE) เดินที่: PASS")
                                        print(f"⌚ ใช้เวลา {elapsed:.2f} วินาที")
                                    else:
                                        ai_xy = board_pos_to_xy(ai_move)
                                        print(f"🤖 AI (WHITE) เดินที่: {ai_move} (ตำแหน่ง X,Y = {ai_xy[0]},{ai_xy[1]})")
                                        print(f"⌚ ใช้เวลา {elapsed:.2f} วินาที")
                                else:
                                    print(f"✅ WHITE เดินที่ {board_pos} (ตำแหน่ง X,Y = {xy[0]},{xy[1]})")
                                new_board_state = self.logic.board_state.copy()
                                captured_black = [pos for pos in previous_board_state if pos not in new_board_state and previous_board_state[pos] == 'white']
                                captured_white = [pos for pos in previous_board_state if pos not in new_board_state and previous_board_state[pos] == 'black']
                                captured_by_black.extend(captured_black)
                                captured_by_white.extend(captured_white)
                                if captured_black or captured_white:
                                    if captured_black:
                                        self.logic.captured_count['black'] += len(captured_black)
                                        for pos in captured_black:
                                            print(f"💥 BLACK จับกินที่: {pos} (หมากขาวถูกกิน)")
                                    if captured_white:
                                        self.logic.captured_count['white'] += len(captured_white)
                                        for pos in captured_white:
                                            print(f"💥 WHITE จับกินที่: {pos} (หมากดำถูกกิน)")
                                    capture_message = ""
                                    if captured_by_black:
                                        capture_message += f"BLACK จับกินที่: {', '.join(captured_by_black)} (หมากขาวถูกกิน)\n"
                                    if captured_by_white:
                                        capture_message += f"WHITE จับกินที่: {', '.join(captured_by_white)} (หมากดำถูกกิน)\n"
                                    print("\n===== แจ้งเตือนการจับกินหมาก =====")
                                    print(capture_message.strip())
                                    print(f"Captured - W: {self.logic.captured_count['white']} | B: {self.logic.captured_count['black']}")
                                    print("==============================\n")
                        for pos in captured_by_black + captured_by_white:
                            px, py = board_to_pixel(pos)
                            cv2.circle(enhanced_color, (px, py), 15, (0, 0, 255), 2)
                        score = self.logic.estimate_score()
                        cv2.putText(enhanced_color, f"Score: {score}", (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        cv2.imshow("Perspective View", enhanced_color)
                        cv2.imshow("Black Stones", BW_black)
                    except Exception as e:
                        print(f"⚠️ Transform Error: {e}")
            aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.imshow("Aruco Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('u'), ord('U')):
                print("\n⏪ Undo ล่าสุด (ย้อนกลับ 1 ตา)")
                self.logic.undo()
                print(f"▶️ ตาปัจจุบัน: ตาที่ {self.logic.turn_number}")
                continue
            if key in (ord('r'), ord('R')):
                print("\n🔄 Reset กระดานใหม่!")
                self.logic.reset()
                print("กระดานถูกรีเซ็ตแล้ว\n")
                continue
            if key in (ord('p'), ord('P')):
                print(f"\n⏭️ ผู้เล่น (BLACK) ขอกด PASS ในตาที่ {self.logic.turn_number}")
                ai_move = self.logic.pass_turn()
                print(f"🤖 AI (WHITE) เดินที่: {ai_move}")
                if ai_move.strip().lower() == 'pass':
                    print("\n🏁 เกมจบแล้ว!")
                    score = self.logic.final_score()
                    print(f"📊 ผลคะแนนรวม: {score}")
                    if score.startswith('B+'):
                        print("🏆 ฝ่ายดำ (BLACK) ชนะ!")
                    elif score.startswith('W+'):
                        print("🏆 ฝ่ายขาว (WHITE) ชนะ!")
                    else:
                        print("🤝 ผลเสมอ หรือไม่สามารถคำนวณคะแนนได้")
                    import datetime
                    sgf_dir = "SGF"
                    if not os.path.exists(sgf_dir):
                        os.makedirs(sgf_dir)
                    sgf_filename = f"game_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.sgf"
                    sgf_path = os.path.join(sgf_dir, sgf_filename)
                    self.logic.save_sgf(sgf_path)
                    print(f"📁 บันทึกไฟล์ SGF สำเร็จ: {sgf_path} ")
                    break
                previous_board_state = self.logic.board_state.copy()
                new_board_state = self.logic.board_state.copy()
                captured_black = [pos for pos in previous_board_state if pos not in new_board_state and previous_board_state[pos] == 'white']
                captured_white = [pos for pos in previous_board_state if pos not in new_board_state and previous_board_state[pos] == 'black']
                if captured_black or captured_white:
                    if captured_black:
                        self.logic.captured_count['black'] += len(captured_black)
                        for pos in captured_black:
                            print(f"💥 BLACK จับกินที่: {pos} (หมากขาวถูกกิน)")
                    if captured_white:
                        self.logic.captured_count['white'] += len(captured_white)
                        for pos in captured_white:
                            print(f"💥 WHITE จับกินที่: {pos} (หมากดำถูกกิน)")
                    capture_message = ""
                    if captured_black:
                        capture_message += f"BLACK จับกินที่: {', '.join(captured_black)} (หมากขาวถูกกิน)\n"
                    if captured_white:
                        capture_message += f"WHITE จับกินที่: {', '.join(captured_white)} (หมากดำถูกกิน)\n"
                    print("\n===== แจ้งเตือนการจับกินหมาก =====")
                    print(capture_message.strip())
                    print(f"Captured - W: {self.logic.captured_count['white']} | B: {self.logic.captured_count['black']}")
                    print("==============================\n")
                continue
            if key == 27:
                break
        self.release()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.logic.gnugo.quit()
        print("🔕 ปิดกล้องและ AI เรียบร้อยแล้ว")

if __name__ == "__main__":
    system = VisionSystem()
    system.run()