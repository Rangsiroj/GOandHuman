
# -----------------------------
# GNUGo: คลาสสำหรับควบคุม GNUGo (AI หมากล้อม) ผ่าน GTP protocol
# -----------------------------

import subprocess

class GNUGo:
    def __init__(self, gnugo_path='GNU_GO\gnugo.exe'):
        # สร้าง process สำหรับรัน GNUGo ด้วยโหมด GTP
        self.process = subprocess.Popen(
            [gnugo_path, '--mode', 'gtp'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    # ส่งคำสั่งไปยัง GNUGo และรับผลลัพธ์กลับมา
    def send_command(self, command):
        try:
            self.process.stdin.write(command + '\n')
            self.process.stdin.flush()
            response_lines = []
            while True:
                line = self.process.stdout.readline()
                if line.strip() == '':
                    break
                response_lines.append(line.strip())
            return '\n'.join(response_lines).replace('= ', '').strip()
        except Exception as e:
            return f"[ERROR] {e}"

    # สั่งให้ GNUGo เดินหมากที่ตำแหน่งที่กำหนด
    def play_move(self, color, move):
        return self.send_command(f"play {color} {move}")

    # ให้ GNUGo คำนวณและเดินหมากอัตโนมัติ
    def genmove(self, color):
        return self.send_command(f"genmove {color}")

    # แสดงกระดานปัจจุบัน
    def show_board(self):
        return self.send_command("showboard")

    # เคลียร์กระดานใหม่
    def clear_board(self):
        return self.send_command("clear_board")

    # คืนคะแนนสุดท้ายเมื่อจบเกม
    def final_score(self):
        return self.send_command("final_score")

    # แสดงพื้นที่ที่ครอบครอง
    def territory(self):
        return self.send_command("territory")

    # คืนคะแนนปัจจุบัน
    def score(self):
        return self.send_command("score")

    # บันทึกเกมเป็นไฟล์ SGF
    def savesgf(self, filepath):
        return self.send_command(f"savesgf {filepath}")

    # ปิด process GNUGo
    def quit(self):
        self.process.terminate()
        print("👋 ออกจากเกมเรียบร้อยแล้ว")


# -----------------------------
# main: ฟังก์ชันสำหรับเล่นเกมหมากล้อมกับ AI ผ่าน command line
# -----------------------------
def main():
    print("🎮 เริ่มเกมหมากล้อมกับ AI (สีดำคือคุณ, สีขาวคือ AI)")
    print("พิมพ์ตำแหน่ง เช่น D4, K10 (หรือ 'quit' เพื่อออก)\n")

    gnugo = GNUGo()
    gnugo.clear_board()

    while True:
        print("\n🧾 กระดานปัจจุบัน:")
        print(gnugo.show_board())

        move = input("👤 คุณ (ดำ) เดินที่: ").strip().upper()

        if move == 'QUIT':
            break

        # ตรวจสอบความถูกต้องของการเดินหมาก
        valid = gnugo.play_move("black", move)
        if "illegal move" in valid.lower() or valid == '':
            print("❌ หมากไม่ถูกต้อง หรือเดินไม่ได้ ลองใหม่")
            continue

        # ให้ AI เดินหมากขาว
        ai_move = gnugo.genmove("white")
        print(f"🤖 AI (ขาว) เดินที่: {ai_move}")

    gnugo.quit()


# รันโปรแกรมหลักเมื่อถูกเรียกโดยตรง
if __name__ == "__main__":
    main()