# -----------------------------
# VisionSystem (YOLO edition): ตรวจจับหมากโกะด้วย ArUco + YOLOv8
# -----------------------------
import os
import time
import cv2
import numpy as np

# --- ArUco compatibility import ---
try:
    import cv2.aruco as aruco
except Exception as e:
    raise ImportError(
        "ไม่พบโมดูล cv2.aruco (ต้องใช้ opencv-contrib-python) "
        "ให้ติดตั้งด้วย: pip install opencv-contrib-python==4.11.0.86"
    ) from e

# --- YOLO / Ultralytics ---
try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError(
        "ไม่พบ ultralytics (YOLOv8). ติดตั้งด้วย: pip install ultralytics"
    ) from e

# ในเครื่องที่ไม่ได้ลง CUDA จะบังคับใช้ CPU อัตโนมัติ
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

from gnugo_text_game import GNUGo
from game_logic import GameLogic


# -----------------------------
# Utilities (ทั่วไป)
# -----------------------------
def board_pos_to_xy(pos):
    """แปลงเช่น 'T19' -> (col,row) index (0..18, 0..18) โดย col A..T (ข้าม I) , row 19..1 -> 0..18"""
    if len(pos) < 2:
        return None
    col_chr = pos[0].upper()
    row_str = pos[1:]
    if not row_str.isdigit():
        return None
    col = ord(col_chr) - ord('A')
    if col_chr >= 'I':
        col -= 1
    # row: 19 -> 0, 1 -> 18
    row = 19 - int(row_str)
    return (col, row)


# -----------------------------
# ArUco helper: รองรับทั้ง API เก่า/ใหม่
# -----------------------------
class ArucoHelper:
    def __init__(self, dict_id=aruco.DICT_4X4_50):
        # dictionary
        if hasattr(aruco, "getPredefinedDictionary"):
            self.dictionary = aruco.getPredefinedDictionary(dict_id)
        else:
            self.dictionary = aruco.Dictionary_get(dict_id)

        # parameters
        if hasattr(aruco, "DetectorParameters"):
            self.parameters = aruco.DetectorParameters()
        else:
            self.parameters = aruco.DetectorParameters_create()

        # detector (ถ้ามี API ใหม่)
        self.detector = None
        if hasattr(aruco, "ArucoDetector"):
            self.detector = aruco.ArucoDetector(self.dictionary, self.parameters)

        # ฟังก์ชัน detectMarkers (API เก่า)
        self.has_detect_function = hasattr(aruco, "detectMarkers")

        if self.detector is None and not self.has_detect_function:
            raise RuntimeError(
                "ไม่พบฟังก์ชัน/คลาสสำหรับตรวจจับ ArUco (ArucoDetector/detectMarkers) "
                "ตรวจสอบเวอร์ชัน opencv-contrib-python"
            )

    def detect(self, gray):
        if self.detector is not None:
            return self.detector.detectMarkers(gray)
        else:
            return aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)


# -----------------------------
# Vision + YOLO System
# -----------------------------
class VisionSystemYOLO:
    def __init__(self,
                 url='http://10.230.241.155:4747/video',   # เปลี่ยนได้เป็น 0 ถ้าใช้เว็บแคม
                 weights_path='models/best.pt',
                 imgsz=640,
                 conf_thres=0.5,
                 iou_thres=0.45):

        # --- Camera ---
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ ไม่สามารถเชื่อมต่อกล้องได้")
        else:
            print("✅ เชื่อมต่อกล้องสำเร็จ")

        # --- Game logic ---
        self.logic = GameLogic(GNUGo())
        self.logic.reset()

        # --- ArUco helper ---
        self.aruco = ArucoHelper(aruco.DICT_4X4_50)

        # --- YOLO weights path ---
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(weights_path):
            weights_path = os.path.join(base_dir, weights_path)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"ไม่พบไฟล์น้ำหนักโมเดล: {weights_path}")

        self.model = YOLO(weights_path)
        self.imgsz = int(imgsz)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)

        # --- UI windows & trackbars ---
        cv2.namedWindow("Aruco Detection")
        cv2.namedWindow("Perspective View")

        # ภาพ/YOLO
        cv2.createTrackbar('Brightness', "Perspective View", 94, 100, lambda x: None)
        cv2.createTrackbar('Contrast',  "Perspective View", 87, 100, lambda x: None)
        cv2.createTrackbar('Conf',      "Perspective View", int(self.conf_thres*100), 100, lambda x: None)
        cv2.createTrackbar('IoU',       "Perspective View", int(self.iou_thres*100), 100, lambda x: None)

        # ปรับขอบ (ครอปจากภาพ 500x500 หลัง warp)
        # เริ่มต้น 25 เท่ากันทุกด้าน แต่เราจูนได้เพื่อให้พื้นที่เล่นตรงกับ 19×19 จริง
        cv2.createTrackbar('Top',    "Perspective View", 25, 60, lambda x: None)
        cv2.createTrackbar('Right',  "Perspective View", 25, 60, lambda x: None)
        cv2.createTrackbar('Bottom', "Perspective View", 25, 60, lambda x: None)
        cv2.createTrackbar('Left',   "Perspective View", 25, 60, lambda x: None)

        # Bias X/Y (px) ไว้ชดเชยกรณี YOLO ให้ศูนย์หมากล้ำเขตช่อง
        cv2.createTrackbar('BiasX',  "Perspective View", 10, 20, lambda x: None)  # -10..+10 (จะแปลงภายหลัง)
        cv2.createTrackbar('BiasY',  "Perspective View", 10, 20, lambda x: None)

        # sticky factor (0..30 -> 0.00..0.30 ของขนาดช่อง) ช่วย "ดูด" ให้ติดขอบสุด
        cv2.createTrackbar('Sticky%', "Perspective View", 12, 30, lambda x: None)

        self.warned_illegal_move = False
        self.warned_occupied_positions = set()

        # motion check
        self.prev_gray = None
        self.last_motion_time = time.time()

    # =========================
    # Mapping helpers (ใหม่)
    # =========================
    @staticmethod
    def _letter_from_col(col_idx):
        # 0..18 -> A..T (ข้าม I)
        letter_code = ord('A') + col_idx
        if col_idx >= 8:
            letter_code += 1  # skip 'I'
        return chr(letter_code)

    def _map_xy_to_pos(self, x, y, W, H, bias_x_px, bias_y_px, sticky_frac):
        """
        x,y: จุดกึ่งกลางหมากในภาพที่ครอปแล้ว (ขนาด W×H)
        bias_x_px, bias_y_px: ชดเชยพิกเซล (+ขวา/+ลง)
        sticky_frac: ส่วนของขนาดช่อง (0..0.3) สำหรับ edge-sticky
        return: 'A19'..'T1' หรือ None
        """
        if W <= 0 or H <= 0:
            return None

        # ชดเชย Bias
        x = float(np.clip(x + bias_x_px, 0, W - 1))
        y = float(np.clip(y + bias_y_px, 0, H - 1))

        cell_w = W / 18.0
        cell_h = H / 18.0

        col = int(round(x / cell_w))
        row = int(round(y / cell_h))

        # sticky edges: ลด threshold ในช่องริมสุด
        # ขวาสุด (col 18)
        if col == 17 and x > W - cell_w * (0.5 + (0.5 - sticky_frac)):
            col = 18
        # ซ้ายสุด (col 0)
        if col == 1 and x < cell_w * (0.5 + (0.5 - sticky_frac)):
            col = 0
        # ล่างสุด (row 18)
        if row == 17 and y > H - cell_h * (0.5 + (0.5 - sticky_frac)):
            row = 18
        # บนสุด (row 0)
        if row == 1 and y < cell_h * (0.5 + (0.5 - sticky_frac)):
            row = 0

        # clamp
        col = int(np.clip(col, 0, 18))
        row = int(np.clip(row, 0, 18))

        # สร้างชื่อช่อง
        col_letter = self._letter_from_col(col)
        pos = f"{col_letter}{19 - row}"
        return pos

    @staticmethod
    def _pos_to_pixel(pos, W, H):
        """แปลงตำแหน่งกระดาน -> พิกเซลในภาพครอป (W×H) เพื่อวาดไฮไลต์"""
        xy = board_pos_to_xy(pos)
        if xy is None:
            return (0, 0)
        col, row = xy
        x = int((col / 18.0) * W)
        y = int((row / 18.0) * H)
        return (x, y)

    # =========================
    # Camera Stability
    # =========================
    def is_camera_stable(self, gray, threshold=500000):
        now = time.time()
        if self.prev_gray is None:
            self.prev_gray = gray
            self.last_motion_time = now
            return True
        diff = cv2.absdiff(self.prev_gray, gray)
        score = float(np.sum(diff))
        self.prev_gray = gray
        if score > threshold:
            self.last_motion_time = now
            return False
        return (now - self.last_motion_time) > 1.0

    # =========================
    # Main loop
    # =========================
    def run(self):
        print("📷 เริ่มตรวจจับกระดานด้วย ArUco + YOLO (ESC เพื่อออก)")

        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ ไม่สามารถดึงภาพจากกล้องได้")
                break

            frame_count += 1
            # โชว์เฟรมบางส่วนเพื่อเฟรมเรตดีขึ้น
            if frame_count % 5 != 0:
                cv2.imshow("Aruco Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # อ่านค่าจาก Trackbar
            brightness = cv2.getTrackbarPos('Brightness', "Perspective View") - 50
            contrast   = cv2.getTrackbarPos('Contrast',  "Perspective View") / 50
            self.conf_thres = max(1, cv2.getTrackbarPos('Conf', "Perspective View")) / 100.0
            self.iou_thres  = max(1, cv2.getTrackbarPos('IoU',  "Perspective View")) / 100.0

            # margins
            m_top    = cv2.getTrackbarPos('Top',    "Perspective View")
            m_right  = cv2.getTrackbarPos('Right',  "Perspective View")
            m_bottom = cv2.getTrackbarPos('Bottom', "Perspective View")
            m_left   = cv2.getTrackbarPos('Left',   "Perspective View")

            # biases (-10..+10)
            biasx_raw = cv2.getTrackbarPos('BiasX', "Perspective View") - 10
            biasy_raw = cv2.getTrackbarPos('BiasY', "Perspective View") - 10

            sticky_pct = cv2.getTrackbarPos('Sticky%', "Perspective View") / 100.0  # 0..0.30

            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ตรวจภาพนิ่งก่อน
            if not self.is_camera_stable(gray):
                cv2.imshow("Aruco Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # ตรวจ ArUco
            try:
                corners, ids, rejected = self.aruco.detect(gray)
            except Exception as e:
                print(f"⚠️ ArUco detect error: {e}")
                corners, ids = [], None

            if corners is None:
                corners = []
            if ids is None:
                ids = []

            # วาด ArUco overlay
            try:
                if len(corners) > 0 and len(ids) > 0:
                    aruco.drawDetectedMarkers(frame, corners, np.array(ids))
            except Exception:
                pass

            # หา 4 มุมด้วย marker id 0,1,2,3
            ids_np = np.array(ids).flatten() if len(ids) > 0 else np.array([])
            if ids_np.size >= 4:
                marker_positions = {}
                for i, marker_id in enumerate(ids_np):
                    if marker_id in [0, 1, 2, 3]:
                        marker_positions[marker_id] = np.mean(corners[i][0], axis=0)

                if len(marker_positions) == 4:
                    try:
                        # perspective warp -> 500x500 เต็มกระดาน (รวมขอบไม้)
                        src_pts = np.float32([
                            marker_positions[0],
                            marker_positions[1],
                            marker_positions[2],
                            marker_positions[3]
                        ])
                        dst_pts = np.float32([[0, 0], [500, 0], [500, 500], [0, 500]])
                        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                        warped = cv2.warpPerspective(frame, matrix, (500, 500))

                        # ครอปด้วย margins ที่จูนได้
                        top, right, bottom, left = m_top, m_right, m_bottom, m_left
                        top    = int(np.clip(top, 0, 200))
                        right  = int(np.clip(right, 0, 200))
                        bottom = int(np.clip(bottom, 0, 200))
                        left   = int(np.clip(left, 0, 200))

                        x1, y1 = left, top
                        x2, y2 = 500 - right, 500 - bottom
                        if x2 - x1 < 100 or y2 - y1 < 100:
                            # กันพลาดหากตั้ง margin มากเกินไป
                            x1, y1, x2, y2 = 25, 25, 475, 475

                        cropped = warped[y1:y2, x1:x2].copy()
                        show_img = cropped.copy()
                        H, W = show_img.shape[:2]

                        # ===== YOLO inference =====
                        results = self.model.predict(
                            source=cropped,
                            imgsz=self.imgsz,
                            conf=self.conf_thres,
                            iou=self.iou_thres,
                            verbose=False,
                            device=DEVICE
                        )

                        detected_positions_by_color = {"black": set(), "white": set()}

                        if len(results) > 0:
                            r = results[0]
                            for b in r.boxes:
                                cls_id = int(b.cls.item())
                                conf   = float(b.conf.item())
                                x1b, y1b, x2b, y2b = map(int, b.xyxy[0].tolist())
                                cx = int((x1b + x2b) / 2)
                                cy = int((y1b + y2b) / 2)

                                color = "black" if cls_id == 0 else "white"
                                board_pos = self._map_xy_to_pos(
                                    cx, cy, W, H,
                                    bias_x_px=biasx_raw,
                                    bias_y_px=biasy_raw,
                                    sticky_frac=min(max(sticky_pct, 0.0), 0.30)
                                )
                                if board_pos:
                                    detected_positions_by_color[color].add(board_pos)
                                    # debug draw
                                    cv2.rectangle(show_img, (x1b, y1b), (x2b, y2b), (0, 255, 255), 1)
                                    cv2.circle(show_img, (cx, cy), 3, (0, 255, 0), -1)
                                    cv2.putText(
                                        show_img,
                                        f"{color} {conf:.2f} {board_pos}",
                                        (x1b, max(15, y1b - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.45, (0, 255, 255), 1
                                    )

                        # --- อัปเดตลอจิกเกมตามหมากที่ YOLO เห็น ---
                        previous_board_state = self.logic.board_state.copy()
                        captured_by_black, captured_by_white = [], []

                        for color in ("white", "black"):
                            if color != self.logic.current_turn:
                                continue

                            previous_positions = {pos for pos, c in self.logic.board_state.items() if c == color}
                            diff = detected_positions_by_color[color] - previous_positions
                            if len(diff) != 1:
                                continue

                            board_pos = diff.pop()
                            if board_pos in self.logic.board_state:
                                if board_pos not in self.warned_occupied_positions:
                                    print(f"🚫 หมากตำแหน่ง {board_pos} ถูกวางไปแล้ว")
                                    self.warned_occupied_positions.add(board_pos)
                                continue

                            ok, result = self.logic.play_move(color, board_pos)
                            if not ok:
                                if not self.warned_illegal_move:
                                    print(f"❌ เดินไม่ถูกต้อง ({result}) ที่ {board_pos}")
                                    self.warned_illegal_move = True
                                continue

                            self.warned_illegal_move = False
                            self.warned_occupied_positions.clear()
                            xy = board_pos_to_xy(board_pos)

                            if color == 'black':
                                print(f"=== ตาที่ {self.logic.turn_number} ===")
                                print(f"✅ BLACK เดินที่ {board_pos} (X,Y={xy[0]},{xy[1]})")

                                # ตรวจจับการจับกินหลังดำเดิน
                                new_board_state_after_black = self.logic.board_state.copy()
                                captured_white_by_black = [
                                    pos for pos in previous_board_state
                                    if pos not in new_board_state_after_black
                                    and previous_board_state[pos] == 'white'
                                ]
                                if captured_white_by_black:
                                    self.logic.captured_count['black'] += len(captured_white_by_black)
                                    for pos in captured_white_by_black:
                                        print(f"💥 BLACK จับกินที่: {pos} (หมากขาวถูกกิน)")
                                    print(
                                        f"Captured - W: {self.logic.captured_count['white']} "
                                        f"| B: {self.logic.captured_count['black']}"
                                    )

                                # ให้ AI (WHITE) เดิน
                                ai_move, elapsed = self.logic.ai_move()
                                if ai_move.strip().lower() == 'pass':
                                    print(f"🤖 AI (WHITE) เดิน: PASS ({elapsed:.2f}s)")
                                else:
                                    ai_xy = board_pos_to_xy(ai_move)
                                    print(
                                        f"🤖 AI (WHITE) เดินที่: {ai_move} (X,Y={ai_xy[0]},{ai_xy[1]})  ⏱ {elapsed:.2f}s"
                                    )
                            else:
                                print(f"✅ WHITE เดินที่ {board_pos} (X,Y={xy[0]},{xy[1]})")

                        # ไฮไลต์จุดที่ถูกจับกิน (ถ้ามี) ด้วยพิกเซลของภาพที่ครอปจริง
                        new_board_state = self.logic.board_state.copy()
                        captured_black = [
                            pos for pos in previous_board_state
                            if pos not in new_board_state and previous_board_state[pos] == 'white'
                        ]
                        captured_white = [
                            pos for pos in previous_board_state
                            if pos not in new_board_state and previous_board_state[pos] == 'black'
                        ]
                        captured_by_black.extend(captured_black)
                        captured_by_white.extend(captured_white)
                        for pos in captured_by_black + captured_by_white:
                            px, py = self._pos_to_pixel(pos, W, H)
                            cv2.circle(show_img, (px, py), 15, (0, 0, 255), 2)

                        # สกอร์ประมาณการ
                        score = self.logic.estimate_score()
                        cv2.putText(show_img, f"Score: {score}", (10, H - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                        # แสดงภาพกระดานที่ครอปแล้ว
                        cv2.imshow("Perspective View", show_img)

                    except Exception as e:
                        print(f"⚠️ Transform/YOLO Error: {e}")

            # โชว์ภาพ ArUco
            cv2.imshow("Aruco Detection", frame)

            # คีย์ลัด: U=Undo, R=Reset, P=Pass, ESC=Exit
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('u'), ord('U')):
                print("\n⏪ Undo ล่าสุด")
                self.logic.undo()
                print(f"▶️ ตาปัจจุบัน: ตาที่ {self.logic.turn_number}")
                continue

            if key in (ord('r'), ord('R')):
                print("\n🔄 Reset กระดานใหม่!")
                self.logic.reset()
                print("กระดานถูกรีเซ็ตแล้ว\n")
                continue

            if key in (ord('p'), ord('P')):
                print(f"\n⏭️ ผู้เล่น (BLACK) PASS ตาที่ {self.logic.turn_number}")
                ai_move = self.logic.pass_turn()
                print(f"🤖 AI (WHITE) เดินที่: {ai_move}")
                if ai_move.strip().lower() == 'pass':
                    print("\n🏁 เกมจบแล้ว!")
                    score = self.logic.final_score()
                    print(f"📊 ผลคะแนนรวม: {score}")
                    import datetime
                    sgf_dir = "SGF"
                    os.makedirs(sgf_dir, exist_ok=True)
                    sgf_filename = f"game_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.sgf"
                    sgf_path = os.path.join(sgf_dir, sgf_filename)
                    self.logic.save_sgf(sgf_path)
                    print(f"📁 บันทึก SGF: {sgf_path}")
                    break
                continue
            if key in (ord('S'), ord('s')):
                import datetime
                sgf_dir = "SGF"
                os.makedirs(sgf_dir, exist_ok=True)
                sgf_filename = f"game_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.sgf"
                sgf_path = os.path.join(sgf_dir, sgf_filename)
                self.logic.save_sgf(sgf_path)
                print(f"\n💾 บันทึก SGF: {sgf_path}")
                continue

            if key == 27:  # ESC
                break

        self.release()

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            self.logic.gnugo.quit()
        except Exception:
            pass
        print("🔕 ปิดกล้องและ AI เรียบร้อยแล้ว")


if __name__ == "__main__":
    print("=" * 60)
    print("🎯 Go Vision System with YOLOv8")
    print("=" * 60)
    print("Controls: ESC ออก | U Undo | R Reset | P Pass")
    print("Trackbars: Brightness/Contrast, Conf/IoU, Top/Right/Bottom/Left, BiasX/BiasY, Sticky%")
    print("=" * 60)
    system = VisionSystemYOLO()
    system.run()
