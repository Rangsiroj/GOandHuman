import cv2
import cv2.aruco as aruco
import numpy as np

def auto_adjust_brightness(gray_image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_image)
    return enhanced

class VisionSystem:
    def __init__(self, url='http://172.23.33.72:4747/video'):  # เปลี่ยน IP DroidCam ตรงนี้
        self.cap = cv2.VideoCapture(2)

        # ใช้ Dictionary สำหรับ ArUco 4x4 (ID สูงสุด ~50)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()

        if not self.cap.isOpened():
            print("❌ ไม่สามารถเชื่อมต่อกล้องได้")
        else:
            print("✅ เชื่อมต่อกล้องสำเร็จ")

    def run(self):
        print("📷 เริ่มแสดงกล้องสดพร้อมตรวจจับ ArUco (กด ESC เพื่อออก)")
        prev_count_black = 0
        prev_count_white = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ ไม่สามารถดึงภาพจากกล้องได้")
                break

            frame_copy = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

            warped = None

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

                        # แปลงภาพเป็น grayscale และปรับแสงด้วย CLAHE
                        stone_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                        enhanced = auto_adjust_brightness(stone_gray)
                        cv2.imshow("Perspective View", enhanced)

                        # ตรวจจับหมาก
                        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
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
                                roi = enhanced[cy - 5:cy + 5, cx - 5:cx + 5]
                                if roi.size == 0:
                                    continue
                                brightness = np.mean(roi)
                                if brightness > 127:
                                    count_white += 1
                                else:
                                    count_black += 1

                        if (count_black != prev_count_black) or (count_white != prev_count_white):
                            print("✅ ตรวจพบหมากใหม่")
                            print(f"จำนวนหมากดำ: {count_black}")
                            print(f"จำนวนหมากขาว: {count_white}")
                            prev_count_black = count_black
                            prev_count_white = count_white

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