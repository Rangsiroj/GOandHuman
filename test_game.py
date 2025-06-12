from gnugo_text_game import GNUGo

class GoGameSimulator:
    def __init__(self):
        self.gnugo = GNUGo()
        self.reset_board()

    def reset_board(self):
        self.gnugo.clear_board()
        self.board_state = {}
        self.captured_history = set()
        self.move_history = []
        self.current_turn = 'black'
        self.turn_count = 1
        print("🔄 กระดานถูกรีเซตเรียบร้อยแล้ว\n")

    def play(self):
        print("🎮 เริ่มเกมหมากล้อม (ดำ=ผู้เล่น, ขาว=AI)")
        print("พิมพ์ตำแหน่ง เช่น D4, K10 หรือพิมพ์ 'PASS', 'UNDO', 'RESET', 'HELP', 'QUIT'\n")

        try:
            while True:
                print(f"\n🎲 ตาที่ {self.turn_count}")
                print("🧾 กระดานปัจจุบัน:")
                print(self.gnugo.show_board())

                if self.current_turn == 'black':
                    move = input("👤 คุณ (ดำ) เดินที่: ").strip().upper()

                    if move == 'QUIT':
                        break

                    if move == 'HELP':
                        self.show_help()
                        continue

                    if move == 'RESET':
                        self.reset_board()
                        continue

                    if move == 'PASS':
                        print("⏭️ คุณข้ามตา")
                        ai_move = self.gnugo.genmove("white")
                        print(f"🤖 AI (ขาว) เดินที่: {ai_move}")
                        if ai_move.upper() == 'PASS':
                            print("🏁 เกมจบลง (ทั้งสองฝ่ายข้ามตา)")
                            break
                        self.board_state[ai_move] = 'white'
                        self.move_history.append(('white', ai_move))
                        continue

                    if move == 'UNDO':
                        self.undo_last_move()
                        continue

                    if move in self.board_state and move not in self.captured_history:
                        print("🚫 หมากนี้ถูกวางไปแล้ว (ลงซ้ำ)")
                        continue

                    valid = self.gnugo.play_move("black", move)
                    if "illegal move" in valid.lower() or valid == '':
                        print("❌ หมากไม่ถูกต้อง หรือเดินไม่ได้")
                        continue

                    self.board_state[move] = 'black'
                    self.move_history.append(('black', move))
                    self.current_turn = 'white'
                    self.turn_count += 1

                elif self.current_turn == 'white':
                    ai_move = self.gnugo.genmove("white")
                    print(f"🤖 AI (ขาว) เดินที่: {ai_move}")
                    if ai_move.lower() != 'pass':
                        self.board_state[ai_move] = 'white'
                        self.move_history.append(('white', ai_move))
                    self.current_turn = 'black'

        except KeyboardInterrupt:
            print("\n⛔ คุณยกเลิกเกมด้วย Ctrl+C")

        finally:
            self.gnugo.quit()
            print("👋 จบเกมแล้ว ขอบคุณที่เล่น!")

    def undo_last_move(self):
        if len(self.move_history) < 2:
            print("⚠️ ไม่สามารถ Undo ได้ (ยังไม่มีหมากทั้งสองฝ่าย)")
            return

        self.gnugo.send_command("undo")
        color2, move2 = self.move_history.pop()
        if move2 in self.board_state:
            del self.board_state[move2]

        self.gnugo.send_command("undo")
        color1, move1 = self.move_history.pop()
        if move1 in self.board_state:
            del self.board_state[move1]

        self.turn_count -= 1
        self.current_turn = 'black'
        print(f"↩️ ย้อนกลับ 1 ตา (ลบหมาก {move1} และ {move2})")

    def show_help(self):
        print("\n📘 คำสั่งที่รองรับ:")
        print("- D4, K10, ... : เดินหมากดำ")
        print("- PASS        : ข้ามตา")
        print("- UNDO        : ย้อนกลับ 1 ตา (ผู้เล่น + AI)")
        print("- RESET       : เริ่มเกมใหม่")
        print("- HELP        : แสดงคำสั่งทั้งหมด")
        print("- QUIT        : ออกจากเกม\n")

if __name__ == "__main__":
    GoGameSimulator().play()