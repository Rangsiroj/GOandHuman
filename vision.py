import cv2
import cv2.aruco as aruco
import numpy as np

class VisionSystem:
    def __init__(self, url='http://10.153.244.243:4747/video'):  # เปลี่ยน IP DroidCam ตรงนี้
        self.cap = cv2.VideoCapture(url)

        # ใช้ Dictionary สำหรับ ArUco 4x4 (ID สูงสุด ~50)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()

        if not self.cap.isOpened():
            print("❌ ไม่สามารถเชื่อมต่อกล้องได้")
        else:
            print("✅ เชื่อมต่อกล้องสำเร็จ")

    def run(self):
        print("📷 เริ่มแสดงกล้องสดพร้อมตรวจจับ ArUco (กด ESC เพื่อออก)")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ ไม่สามารถดึงภาพจากกล้องได้")
                break

            frame_copy = frame.copy()  # ✅ ใช้ตัวนี้สำหรับวาด marker
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
                            marker_positions[0],  # Top-left
                            marker_positions[1],  # Top-right
                            marker_positions[2],  # Bottom-right
                            marker_positions[3],  # Bottom-left
                        ])

                        width, height = 500, 500
                        dst_pts = np.float32([
                            [0, 0],
                            [width, 0],
                            [width, height],
                            [0, height]
                        ])

                        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

                        # ✅ ใช้ frame ดั้งเดิมที่ไม่มี marker ถูกวาดทับ
                        warped = cv2.warpPerspective(frame, matrix, (width, height))
                        cv2.imshow("Perspective View", warped)

                    except Exception as e:
                        print(f"⚠️ Transform Error: {e}")

            # ✅ แสดงกล้องพร้อม marker
            cv2.imshow("ArUco Detection", frame_copy)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.release()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("🛑 ปิดกล้องเรียบร้อยแล้ว")