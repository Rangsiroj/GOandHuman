class GoGameLogic:
    def __init__(self, gnugo):
        self.gnugo = gnugo
        self.reset()

    def reset(self):
        self.gnugo.clear_board()
        self.board_state = {}
        self.captured_history = set()
        self.move_history = []
        self.current_turn = 'black'

    def is_duplicate(self, pos):
        return pos in self.board_state and pos not in self.captured_history

    def play_move(self, color, pos):
        result = self.gnugo.play_move(color, pos)
        if "illegal move" in result.lower():
            return False, result
        self.board_state[pos] = color
        self.move_history.append((color, pos))
        self.current_turn = 'white' if color == 'black' else 'black'
        return True, result

    def ai_move(self):
        ai_pos = self.gnugo.genmove('white')
        if ai_pos.lower() != 'pass':
            self.board_state[ai_pos] = 'white'
            self.move_history.append(('white', ai_pos))
        self.current_turn = 'black'
        return ai_pos

    def undo(self):
        if len(self.move_history) < 2:
            return False, "ไม่สามารถ Undo ได้ (ยังไม่มีหมากทั้งสองฝ่าย)"

        self.gnugo.send_command("undo")
        _, ai_move = self.move_history.pop()
        if ai_move in self.board_state:
            del self.board_state[ai_move]

        self.gnugo.send_command("undo")
        _, player_move = self.move_history.pop()
        if player_move in self.board_state:
            del self.board_state[player_move]

        self.current_turn = 'black'
        return True, f"↩️ ย้อนกลับ 1 ตา (ลบหมาก {player_move} และ {ai_move})"

    def pass_turn(self):
        current_color = self.current_turn
        self.gnugo.send_command(f"play {current_color} pass")
        self.move_history.append((current_color, 'pass'))
        self.current_turn = 'white' if current_color == 'black' else 'black'
        return True, f"⏭️ {current_color.upper()} ข้ามตา"

    def estimate_score(self):
        return self.gnugo.send_command("estimate_score")

    def get_current_turn(self):
        return self.current_turn

    def get_board_state(self):
        return self.board_state.copy()

    def get_move_count(self):
        return len(self.move_history) // 2  # นับรอบ (ดำ+ขาว = 1 รอบ)