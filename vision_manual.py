import cv2
import numpy as np
from board_mapper import get_board_position

# สำหรับเก็บจุดคลิก
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

            brightness = cv2.getTrackbarPos('Brightness', "Manual Detection") - 50
            contrast = cv2.getTrackbarPos('Contrast', "Manual Detection") / 50
            white_thresh = cv2.getTrackbarPos('White Threshold', "Manual Detection")
            black_thresh = cv2.getTrackbarPos('Black Threshold', "Manual Detection")

            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
            frame_copy = frame.copy()

            if len(manual_pts) == 4:
                try:
                    src_pts = np.float32(manual_pts)
                    dst_pts = np.float32([[0, 0], [500, 0], [500, 500], [0, 500]])
                    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped = cv2.warpPerspective(frame, matrix, (500, 500))

                    enhanced_color = warped.copy()
                    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    enhanced = auto_adjust_brightness(gray)

                    BW_black = cv2.threshold(enhanced, black_thresh, 255, cv2.THRESH_BINARY_INV)[1]
                    BW_white = cv2.threshold(enhanced, white_thresh, 255, cv2.THRESH_BINARY)[1]

                    kernel = np.ones((5, 5), np.uint8)
                    BW_black = cv2.morphologyEx(BW_black, cv2.MORPH_OPEN, kernel)
                    BW_white = cv2.morphologyEx(BW_white, cv2.MORPH_OPEN, kernel)

                    # แสดง label ช่องกระดาน 19x19 บนภาพ
                    # cell_size = 500 // 19
                    # for row in range(19):
                    #     for col in range(19):
                    #         x = col * cell_size + cell_size // 2
                    #         y = row * cell_size + cell_size // 2

                    #         col_letter = chr(ord('A') + col)
                    #         if col_letter >= 'I':
                    #             col_letter = chr(ord(col_letter) + 1)

                    #         label = f"{col_letter}{19 - row}"
                    #         cv2.putText(enhanced_color, label, (x - 10, y + 5),
                    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

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
        print("🔕 ปิดกล้องเรียบร้อยแล้ว")