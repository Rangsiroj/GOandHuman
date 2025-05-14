import cv2
import cv2.aruco as aruco

class VisionSystem:
    def __init__(self, url='http://172.23.33.72:4747/video'):  # เปลี่ยน IP DroidCam ตรงนี้
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

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ตรวจจับ ArUco
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id in [0, 1, 2, 3]:
                        aruco.drawDetectedMarkers(frame, corners, ids)

                        # ใส่ข้อความ ID บนมุมซ้ายบนของ marker
                        corner = corners[i][0]
                        top_left = corner[0].astype(int)
                        cv2.putText(frame, f"ID {marker_id}", tuple(top_left), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("ArUco Detection", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC = ออก
                break

        self.release()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("🛑 ปิดกล้องเรียบร้อยแล้ว")