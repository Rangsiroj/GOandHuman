class GameLogic:
    def __init__(self, gnugo):
        self.gnugo = gnugo
        self.reset()

    def reset(self):
        self.gnugo.clear_board()
        self.board_state = {}
        self.captured_count = {"black": 0, "white": 0}
        self.move_history = []
        self.current_turn = 'black'
        self.turn_number = 1
        self.undo_pending = False

    def play_move(self, color, pos):
        result = self.gnugo.play_move(color, pos)
        if "illegal move" in result.lower():
            return False, result
        self.sync_board_state_from_gnugo()
        self.move_history.append((color[0].upper(), pos))
        self.current_turn = 'white' if color == 'black' else 'black'
        return True, result

    def ai_move(self):
        import time
        start_time = time.time()
        ai_pos = self.gnugo.genmove('white')
        elapsed = time.time() - start_time
        self.sync_board_state_from_gnugo()
        self.move_history.append(('W', ai_pos))
        self.current_turn = 'black'
        self.turn_number += 1
        return ai_pos, elapsed

    def pass_turn(self):
        result = self.gnugo.play_move('black', 'pass')
        self.move_history.append(('B', ''))
        ai_move = self.gnugo.genmove('white')
        if ai_move.strip().lower() == 'pass':
            self.move_history.append(('W', ''))
        else:
            self.move_history.append(('W', ai_move))
        self.sync_board_state_from_gnugo()
        self.turn_number += 1
        self.current_turn = 'black'
        return ai_move

    def undo(self):
        self.gnugo.send_command('undo')
        self.gnugo.send_command('undo')
        if len(self.move_history) >= 2:
            self.move_history.pop()
            self.move_history.pop()
        self.sync_board_state_from_gnugo()
        self.turn_number = max(1, self.turn_number - 1)
        self.current_turn = 'black'
        self.undo_pending = True

    def sync_board_state_from_gnugo(self):
        board_str = self.gnugo.send_command('showboard')
        new_state = {}
        for line in board_str.splitlines():
            if line.strip() and line[0].isdigit():
                parts = line.strip().split()
                row_num = int(parts[0])
                for col_idx, cell in enumerate(parts[1:]):
                    if cell in ['X', 'O']:
                        col_chr = chr(ord('A') + col_idx)
                        if col_chr >= 'I':
                            col_chr = chr(ord(col_chr) + 1)
                        pos = f"{col_chr}{row_num}"
                        color = 'black' if cell == 'X' else 'white'
                        new_state[pos] = color
        self.board_state = new_state

    def estimate_score(self):
        return self.gnugo.send_command("estimate_score")

    def final_score(self):
        return self.gnugo.final_score()

    def save_sgf(self, filepath):
        def to_sgf_coord(move):
            if not move:
                return ''
            if len(move) < 2 or not move[0].isalpha() or not move[1:].isdigit():
                return ''
            col = move[0].lower()
            row = move[1:]
            col_num = ord(col) - ord('a')
            if col_num >= 8:
                col_num -= 1
            sgf_col = chr(ord('a') + col_num)
            sgf_row = chr(ord('a') + 19 - int(row))
            return f"{sgf_col}{sgf_row}"
        sgf_moves = ''
        for color, move in self.move_history:
            if color == 'B':
                sgf_moves += f";B[{to_sgf_coord(move)}]"
            elif color == 'W':
                sgf_moves += f";W[{to_sgf_coord(move)}]"
        sgf_content = f"(;GM[1]FF[4]SZ[19]{sgf_moves})\n"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(sgf_content)
