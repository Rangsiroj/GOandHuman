import cv2
import cv2.aruco as aruco
import numpy as np
from board_mapper import get_board_position

def auto_adjust_brightness(gray_image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)
    return clahe_image

class VisionSystem:
    def __init__(self, url='http://10.105.55.249:4747/video'):
        self.cap = cv2.VideoCapture(url)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        self.board_state = {}
        self.current_turn = 'black'

        if not self.cap.isOpened():
            print("❌ ไม่สามารถเชื่อมต่อกล้องได้")
        else:
            print("✅ เชื่อมต่อกล้องสำเร็จ")

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

            brightness = cv2.getTrackbarPos('Brightness', "ArUco Detection") - 50
            contrast = cv2.getTrackbarPos('Contrast', "ArUco Detection") / 50
            white_thresh = cv2.getTrackbarPos('White Threshold', "ArUco Detection")
            black_thresh = cv2.getTrackbarPos('Black Threshold', "ArUco Detection")

            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
            frame_copy = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

                        width, height = 500, 500
                        dst_pts = np.float32([
                            [0, 0],
                            [width, 0],
                            [width, height],
                            [0, height]
                        ])

                        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                        warped = cv2.warpPerspective(frame, matrix, (width, height))
<<<<<<< HEAD:vision.py
cv2.imshow("Perspective View", warped)

                        # ตรวจจับหมาก
stone_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(stone_gray, (5, 5), 0)
circles = cv2.HoughCircles(
    blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
    param1=50, param2=30, minRadius=10, maxRadius=30
    )

count_black = 0
count_white = 0

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cx, cy, r = i
        roi = stone_gray[cy - 5:cy + 5, cx - 5:cx + 5]
        if roi.size == 0:
            continue
            brightness = np.mean(roi)
            if brightness > 127:
                count_white += 1
            else:
                count_black += 1

                        # ✅ แสดงผลเฉพาะเมื่อจำนวนหมากเปลี่ยน
            if (count_black != prev_count_black) or (count_white != prev_count_white):
                print("✅ ตรวจพบหมากใหม่")
                print(f"จำนวนหมากดำ: {count_black}")
                print(f"จำนวนหมากขาว: {count_white}")
                prev_count_black = count_black
                prev_count_white = count_white
=======

stone_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
enhanced = auto_adjust_brightness(stone_gray)

blurred_enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)

BW_black = cv2.threshold(blurred_enhanced, black_thresh, 255, cv2.THRESH_BINARY_INV)[1]
BW_white = cv2.threshold(blurred_enhanced, white_thresh, 255, cv2.THRESH_BINARY)[1]

kernel = np.ones((5, 5), np.uint8)
BW_black = cv2.morphologyEx(BW_black, cv2.MORPH_OPEN, kernel)
BW_white = cv2.morphologyEx(BW_white, cv2.MORPH_OPEN, kernel)

cv2.imshow("Perspective View", enhanced)
cv2.imshow("Black Stones", BW_black)
cv2.imshow("White Stones", BW_white)

for mask, color in [(BW_white, "white"), (BW_black, "black")]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        if r >= 5 and circularity > 0.5:
            board_pos = get_board_position(int(x), int(y))
            if board_pos and board_pos not in self.board_state:
                if color == self.current_turn:
                                            self.board_state[board_pos] = color
                                            print(f"✅ {color.upper()} เดินที่ {board_pos}")
                                            self.current_turn = 'white' if self.current_turn == 'black' else 'black'
                else:
                                            print(f"⛔️ ยังไม่ถึงตาของ {color}")
            elif board_pos in self.board_state:
                                        print(f"⚠️ หมากที่ตำแหน่ง {board_pos} มีอยู่แล้ว")
>>>>>>> 573b899a59cc378f253aa50b6995dec7cda34758:vision_aruco.py

except Exception as e:
print(f"⚠️ Transform Error: {e}")

cv2.imshow("ArUco Detection", frame_copy)
if cv2.waitKey(1) & 0xFF == 27:
break

self.release()



def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("🛑 ปิดกล้องเรียบร้อยแล้ว")