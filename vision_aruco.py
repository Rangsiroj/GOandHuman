# -----------------------------
# VisionSystem: ระบบตรวจจับหมากโกะด้วย ArUco Marker + Adaptive Otsu Thresholding
# -----------------------------

import cv2
import numpy as np
import time
import cv2.aruco as aruco
import os
from board_mapper_aruco import get_board_position  # ฟังก์ชันแปลงพิกัด pixel เป็นตำแหน่งบนกระดาน
from gnugo_text_game import GNUGo  # คลาสสำหรับควบคุม AI โกะ
from game_logic import GameLogic  # ตรรกะเกมโกะ

# ปรับความสว่างของภาพด้วย CLAHE เพื่อเพิ่ม contrast
def auto_adjust_brightness(gray_image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)


# Adaptive Otsu Thresholding แบบแบ่ง zones
def adaptive_otsu_thresholding(gray_image, zones=(8, 8), use_adaptive=True):
    """
    แบ่งภาพเป็น zones และใช้ Otsu thresholding แต่ละ zone แยกกัน โดยไม่ต้องส่งตัวแปรหลายตัว
    ทำการคิวรีค่า threshold ในแต่ละโซนอัตโนมัติ

    Args:
        gray_image: ภาพขาวดำที่ต้องการประมวลผล
        zones: tuple (rows, cols) จำนวน zones ที่จะแบ่ง เช่น (8,8) = 32 zones
        use_adaptive: True ใช้ adaptive, False ใช้ global Otsu
    
    Returns:
        binary_white: Binary image สำหรับหมากขาว
        binary_black: Binary image สำหรับหมากดำ  
        threshold_map: แผนที่แสดง threshold values แต่ละ zone
    """
    h, w = gray_image.shape
    zone_h = h // zones[0]
    zone_w = w // zones[1]
    
    # สร้างภาพผลลัพธ์
    binary_white = np.zeros_like(gray_image)
    binary_black = np.zeros_like(gray_image)
    threshold_map = np.zeros_like(gray_image, dtype=np.float32)
    
    if not use_adaptive:
        # Global Otsu thresholding (วิธีเดิม)
        thresh_val, binary_white = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binary_black = cv2.threshold(gray_image, thresh_val, 255, cv2.THRESH_BINARY_INV)
        threshold_map.fill(thresh_val)
        return binary_white, binary_black, threshold_map
    
    # Adaptive Otsu thresholding แต่ละ zone
    for i in range(zones[0]):
        for j in range(zones[1]):
            # คำนวณพิกัด zone
            y1 = i * zone_h
            y2 = min((i + 1) * zone_h, h)
            x1 = j * zone_w
            x2 = min((j + 1) * zone_w, w)
            
            # ตัดเอา zone ปัจจุบัน
            zone = gray_image[y1:y2, x1:x2]
            
            if zone.size == 0:
                continue
                
            # คำนวณ histogram เพื่อตรวจสอบว่ามีข้อมูลเพียงพอหรือไม่
            hist = cv2.calcHist([zone], [0], None, [256], [0, 256])
            non_zero_bins = np.count_nonzero(hist)
            
            # ถ้ามีความแปรปรวนเพียงพอ ใช้ Otsu
            if non_zero_bins > 2 and np.std(zone) > 10:
                thresh_val, zone_white = cv2.threshold(zone, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, zone_black = cv2.threshold(zone, thresh_val, 255, cv2.THRESH_BINARY_INV)
            else:
                # ถ้าความแปรปรวนต่ำเกินไป ใช้ค่าเฉลี่ยแทน
                mean_val = np.mean(zone)
                thresh_val = mean_val
                _, zone_white = cv2.threshold(zone, thresh_val, 255, cv2.THRESH_BINARY)
                _, zone_black = cv2.threshold(zone, thresh_val, 255, cv2.THRESH_BINARY_INV)
            
            # ใส่ผลลัพธ์กลับในภาพหลัก
            binary_white[y1:y2, x1:x2] = zone_white
            binary_black[y1:y2, x1:x2] = zone_black
            threshold_map[y1:y2, x1:x2] = thresh_val
    
    return binary_white, binary_black, threshold_map

# ฟังก์ชันสำหรับแสดง threshold map เป็นสี
def visualize_threshold_map(threshold_map, zones=(8, 8)):
    """แสดง threshold map เป็นสีสำหรับแต่ละ zone"""
    # Normalize threshold values to 0-255 range
    norm_map = cv2.normalize(threshold_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # แปลงเป็น colormap
    colored_map = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    
    # วาดเส้นแบ่ง zones
    h, w = threshold_map.shape
    zone_h = h // zones[0]
    zone_w = w // zones[1]
    
    # วาดเส้นแนวนอน
    for i in range(1, zones[0]):
        y = i * zone_h
        cv2.line(colored_map, (0, y), (w, y), (255, 255, 255), 1)
    
    # วาดเส้นแนวตั้ง
    for j in range(1, zones[1]):
        x = j * zone_w
        cv2.line(colored_map, (x, 0), (x, h), (255, 255, 255), 1)
    
    # แสดงค่า threshold ในแต่ละ zone
    for i in range(zones[0]):
        for j in range(zones[1]):
            y1 = i * zone_h
            y2 = min((i + 1) * zone_h, h)
            x1 = j * zone_w
            x2 = min((j + 1) * zone_w, w)
            
            # หาค่าเฉลี่ยของ threshold ใน zone นี้
            zone_thresh = threshold_map[y1:y2, x1:x2]
            avg_thresh = np.mean(zone_thresh)
            
            # แสดงค่าที่กึ่งกลาง zone
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.putText(colored_map, f"{int(avg_thresh)}", 
                       (center_x - 15, center_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return colored_map

# แปลงตำแหน่งบนกระดาน (เช่น 'A19') เป็นพิกัด pixel สำหรับแสดงผล
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


# แปลงตำแหน่งบนกระดาน (เช่น 'A19') เป็นพิกัดในรูปแบบ (col, row)
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


# คลาสหลักสำหรับระบบ Vision ที่ใช้ ArUco ในการตรวจจับกระดานและหมาก
class VisionSystem:
    def __init__(self, url='http://10.79.185.102:4747/video'):
        # เปิดกล้องจาก URL ที่กำหนด
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            print("❌ ไม่สามารถเชื่อมต่อกล้องได้")
        else:
            print("✅ เชื่อมต่อกล้องสำเร็จ")

        # สร้างอ็อบเจ็กต์สำหรับ logic ของเกมและรีเซ็ตกระดาน
        self.logic = GameLogic(GNUGo())
        self.logic.reset()

        # กำหนด dictionary และ parameter สำหรับ ArUco marker
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()

        # สร้าง trackbar สำหรับปรับ brightness, contrast, threshold และ adaptive settings
        cv2.namedWindow("Perspective View")
        cv2.createTrackbar('Brightness', "Perspective View", 94, 100, lambda x: None)
        cv2.createTrackbar('Contrast', "Perspective View", 87, 100, lambda x: None)
        cv2.createTrackbar('White Threshold', "Perspective View", 252, 255, lambda x: None)
        cv2.createTrackbar('Black Threshold', "Perspective View", 174, 255, lambda x: None)
        
        # เพิ่ม trackbar สำหรับ Adaptive Otsu
        cv2.createTrackbar('Use Adaptive', "Perspective View", 1, 1, lambda x: None)  # 0=Global, 1=Adaptive
        cv2.createTrackbar('Zone Rows', "Perspective View", 8, 8, lambda x: None)     # จำนวนแถว zones
        cv2.createTrackbar('Zone Cols', "Perspective View", 8, 8, lambda x: None)     # จำนวนคอลัมน์ zones

        self.warned_illegal_move = False
        self.warned_occupied_positions = set()


    # ตรวจสอบว่ากล้องนิ่งหรือไม่ โดยเปรียบเทียบภาพก่อนหน้า
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


    # ฟังก์ชันหลักสำหรับรันระบบตรวจจับ ArUco และประมวลผลภาพ
    def run(self):
        print("📷 เริ่มตรวจจับกระดานด้วย ArUco + Adaptive Otsu (ESC เพื่อออก)")
        cv2.namedWindow("Aruco Detection")
        cv2.namedWindow("Threshold Map")  # หน้าต่างใหม่สำหรับแสดง threshold map
        
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
            
            # อ่านค่าจาก trackbar เพื่อปรับภาพและ adaptive settings
            brightness = cv2.getTrackbarPos('Brightness', "Perspective View") - 50
            contrast = cv2.getTrackbarPos('Contrast', "Perspective View") / 50
            white_thresh = cv2.getTrackbarPos('White Threshold', "Perspective View")
            black_thresh = cv2.getTrackbarPos('Black Threshold', "Perspective View")
            
            # อ่านค่า adaptive settings
            use_adaptive = bool(cv2.getTrackbarPos('Use Adaptive', "Perspective View"))
            zone_rows = max(1, cv2.getTrackbarPos('Zone Rows', "Perspective View"))
            zone_cols = max(1, cv2.getTrackbarPos('Zone Cols', "Perspective View"))
            
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # ตรวจสอบว่ากล้องนิ่งหรือไม่
            if not self.is_camera_stable(gray):
                cv2.imshow("Aruco Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
            
            # ตรวจจับ ArUco marker ในภาพ
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
            
            # ตรวจสอบว่าพบ marker ครบ 4 จุดหรือไม่
            if ids is not None and len(ids) >= 4:
                ids = ids.flatten()
                marker_positions = {}
                for i, marker_id in enumerate(ids):
                    if marker_id in [0, 1, 2, 3]:
                        marker_positions[marker_id] = corners[i][0].mean(axis=0)
                
                if len(marker_positions) == 4:
                    try:
                        # Perspective Transform เพื่อปรับมุมมองภาพกระดาน
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
                        
                        # ใช้ Adaptive Otsu Thresholding แทนวิธีเดิม
                        if use_adaptive:
                            BW_white, BW_black, threshold_map = adaptive_otsu_thresholding(
                                blurred, zones=(zone_rows, zone_cols), use_adaptive=True
                            )
                            # แสดง threshold map
                            colored_threshold_map = visualize_threshold_map(threshold_map, (zone_rows, zone_cols))
                            cv2.imshow("Threshold Map", colored_threshold_map)
                        else:
                            # ใช้วิธีเดิม (manual threshold)
                            BW_black = cv2.threshold(blurred, black_thresh, 255, cv2.THRESH_BINARY_INV)[1]
                            BW_white = cv2.threshold(blurred, white_thresh, 255, cv2.THRESH_BINARY)[1]
                            # สร้าง threshold map แบบ uniform
                            threshold_map = np.full_like(blurred, (white_thresh + black_thresh) / 2, dtype=np.float32)
                            cv2.imshow("Threshold Map", cv2.applyColorMap((threshold_map/255*255).astype(np.uint8), cv2.COLORMAP_JET))
                        
                        # ปรับปรุงด้วย morphological operations
                        kernel = np.ones((5, 5), np.uint8)
                        BW_black = cv2.morphologyEx(BW_black, cv2.MORPH_OPEN, kernel)
                        BW_white = cv2.morphologyEx(BW_white, cv2.MORPH_OPEN, kernel)
                        
                        captured_by_black = []
                        captured_by_white = []
                        
                        # เก็บสถานะกระดานก่อนเดินหมาก
                        previous_board_state = self.logic.logic.board_state.copy() if hasattr(self, 'logic') and hasattr(self.logic, 'logic') else self.logic.board_state.copy()
                        
                        # ตรวจจับหมากขาวและดำในภาพ
                        for mask, color in [(BW_white, "white"), (BW_black, "black")]:
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            detected_positions = set()
                            
                            for cnt in contours:
                                (x, y), r = cv2.minEnclosingCircle(cnt)
                                area = cv2.contourArea(cnt)
                                perimeter = cv2.arcLength(cnt, True)
                                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                                
                                # เงื่อนไขสำหรับตรวจจับหมาก (อาจต้องปรับตามผลของ adaptive thresholding)
                                if 7 <= r <= 14 and 0.80 <= circularity <= 1.15 and 60 <= area <= 450:
                                    board_pos = get_board_position(int(x), int(y))
                                    if board_pos:
                                        detected_positions.add(board_pos)
                                        # แสดงตำแหน่งที่ตรวจจับได้
                                        cv2.putText(enhanced_color, f"{board_pos}", (int(x), int(y)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                                        # แสดงค่า threshold ในตำแหน่งนั้น
                                        local_thresh = threshold_map[max(0, int(y)-5):min(threshold_map.shape[0], int(y)+5),
                                                                   max(0, int(x)-5):min(threshold_map.shape[1], int(x)+5)]
                                        avg_thresh = np.mean(local_thresh) if local_thresh.size > 0 else 0
                                        cv2.putText(enhanced_color, f"T:{int(avg_thresh)}", 
                                                   (int(x), int(y)-15),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                            
                            previous_positions = {pos for pos, c in self.logic.board_state.items() if c == color}
                            diff = detected_positions - previous_positions
                            
                            # ถ้าเป็นตาของผู้เล่นและมีหมากใหม่ 1 ตำแหน่ง
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
                                xy = board_pos_to_xy(board_pos)
                                
                                if color == 'black':
                                    print(f"=== ตาที่ {self.logic.turn_number} ===")
                                    print(f"✅ BLACK เดินที่ {board_pos} (ตำแหน่ง X,Y = {xy[0]},{xy[1]})")
                                    
                                    # แจ้งเตือนจับกินหมากขาวหลังหมากดำเดิน
                                    new_board_state_after_black = self.logic.board_state.copy()
                                    captured_white_by_black = [pos for pos in previous_board_state 
                                                               if pos not in new_board_state_after_black 
                                                               and previous_board_state[pos] == 'white']
                                    if captured_white_by_black:
                                        self.logic.captured_count['black'] += len(captured_white_by_black)
                                        for pos in captured_white_by_black:
                                            print(f"💥 BLACK จับกินที่: {pos} (หมากขาวถูกกิน)")
                                        capture_message = f"BLACK จับกินที่: {', '.join(captured_white_by_black)} (หมากขาวถูกกิน)"
                                        print("\n===== แจ้งเตือนการจับกินหมาก =====")
                                        print(capture_message)
                                        print(f"Captured - W: {self.logic.captured_count['white']} | B: {self.logic.captured_count['black']}")
                                        print("==============================\n")
                                    self.last_captured_white_by_black = set(captured_white_by_black)
                                    
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
                                captured_black = [pos for pos in previous_board_state 
                                                 if pos not in new_board_state 
                                                 and previous_board_state[pos] == 'white']
                                captured_white = [pos for pos in previous_board_state 
                                                 if pos not in new_board_state 
                                                 and previous_board_state[pos] == 'black']
                                
                                # ตรวจสอบไม่ให้แจ้งเตือนซ้ำตำแหน่งที่หมากขาวถูกจับกินโดยหมากดำ
                                if hasattr(self, 'last_captured_white_by_black'):
                                    captured_black = [pos for pos in captured_black 
                                                     if pos not in self.last_captured_white_by_black]
                                
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
                                
                                self.last_captured_white_by_black = set()
                        
                        # วาดวงกลมแสดงตำแหน่งที่ถูกจับกิน
                        for pos in captured_by_black + captured_by_white:
                            px, py = board_to_pixel(pos)
                            cv2.circle(enhanced_color, (px, py), 15, (0, 0, 255), 2)
                        
                        # แสดงคะแนนที่ประเมินโดย AI
                        score = self.logic.estimate_score()
                        cv2.putText(enhanced_color, f"Score: {score}", (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
                        # แสดงสถานะ Adaptive Otsu
                        adaptive_text = f"Adaptive: {'ON' if use_adaptive else 'OFF'}"
                        if use_adaptive:
                            adaptive_text += f" ({zone_rows}x{zone_cols})"
                        cv2.putText(enhanced_color, adaptive_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # แสดงผลภาพต่าง ๆ
                        cv2.imshow("Perspective View", enhanced_color)
                        cv2.imshow("Black Stones", BW_black)
                        cv2.imshow("White Stones", BW_white)
                        
                    except Exception as e:
                        print(f"⚠️ Transform Error: {e}")
            
            # วาด marker ที่ตรวจจับได้บนภาพ
            aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.imshow("Aruco Detection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # กด u/U เพื่อ Undo, r/R เพื่อ Reset, p/P เพื่อ Pass, ESC เพื่อออก
            # เพิ่มปุ่ม a/A เพื่อเปลี่ยนโหมด Adaptive
            if key in (ord('a'), ord('A')):
                current_adaptive = cv2.getTrackbarPos('Use Adaptive', "Perspective View")
                cv2.setTrackbarPos('Use Adaptive', "Perspective View", 1 - current_adaptive)
                mode_text = "Adaptive Otsu" if (1 - current_adaptive) else "Manual Threshold"
                print(f"🔄 เปลี่ยนเป็นโหมด: {mode_text}")
                continue
            
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
                captured_black = [pos for pos in previous_board_state 
                                 if pos not in new_board_state 
                                 and previous_board_state[pos] == 'white']
                captured_white = [pos for pos in previous_board_state 
                                 if pos not in new_board_state 
                                 and previous_board_state[pos] == 'black']
                
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
                    print(f"Captured - W: {self.logic.captured_count['white']} | B: {self.logic.captured_count['black']}\n")
                    print("==============================\n")
                continue
            
            if key == 27:  # ESC key
                break
        
        self.release()


    # ปิดกล้องและหน้าต่างทั้งหมดเมื่อจบการทำงาน
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.logic.gnugo.quit()
        print("🔕 ปิดกล้องและ AI เรียบร้อยแล้ว")


# รันโปรแกรมหลักเมื่อถูกเรียกโดยตรง
if __name__ == "__main__":
    print("=" * 60)
    print("🎯 Go Vision System with Adaptive Otsu Thresholding")
    print("=" * 60)
    print("🔧 Controls:")
    print("   ESC - ออกจากโปรแกรม")
    print("   U   - Undo (ย้อนกลับ)")
    print("   R   - Reset กระดาน")
    print("   P   - Pass (ข้ามตา)")
    print("   A   - เปลี่ยนโหมด Adaptive/Manual")
    print("")
    print("📊 Trackbar Controls:")
    print("   Brightness, Contrast - ปรับภาพพื้นฐาน")
    print("   White/Black Threshold - สำหรับโหมด Manual")
    print("   Use Adaptive - เปิด/ปิด Adaptive Otsu")
    print("   Zone Rows/Cols - จำนวน zones สำหรับ Adaptive")
    print("=" * 60)
    
    system = VisionSystem()
    system.run()
