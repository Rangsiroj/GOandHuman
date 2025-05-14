import cv2

class VisionSystem:
    def __init__(self, url='http://10.153.244.243:4747/video'):  # เปลี่ยน IP ตามของคุณ
        self.cap = cv2.VideoCapture(url)

        if not self.cap.isOpened():
            print("❌ ไม่สามารถเชื่อมต่อกล้องได้")
        else:
            print("✅ เชื่อมต่อกล้องสำเร็จ")

    def run(self):
        print("📷 เริ่มแสดงกล้องสด (กด ESC เพื่อปิด)")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ ไม่สามารถดึงภาพจากกล้องได้")
                break

            cv2.imshow("DroidCam Live Feed", frame)

            # ออกจากโปรแกรมเมื่อกด ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.release()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("🛑 ปิดกล้องเรียบร้อยแล้ว")