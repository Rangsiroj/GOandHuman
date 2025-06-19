import subprocess

class GNUGo:
    def __init__(self, gnugo_path='GNU_GO\gnugo.exe'):
        self.process = subprocess.Popen(
            [gnugo_path, '--mode', 'gtp'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

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

    def play_move(self, color, move):
        return self.send_command(f"play {color} {move}")

    def genmove(self, color):
        return self.send_command(f"genmove {color}")

    def show_board(self):
        return self.send_command("showboard")

    def clear_board(self):
        return self.send_command("clear_board")

    def final_score(self):
        return self.send_command("final_score")

    def territory(self):
        return self.send_command("territory")

    def score(self):
        return self.send_command("score")

    def savesgf(self, filepath):
        return self.send_command(f"savesgf {filepath}")

    def quit(self):
        self.process.terminate()
        print("👋 ออกจากเกมเรียบร้อยแล้ว")


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

        valid = gnugo.play_move("black", move)
        if "illegal move" in valid.lower() or valid == '':
            print("❌ หมากไม่ถูกต้อง หรือเดินไม่ได้ ลองใหม่")
            continue

        ai_move = gnugo.genmove("white")
        print(f"🤖 AI (ขาว) เดินที่: {ai_move}")

    gnugo.quit()


if __name__ == "__main__":
    main()