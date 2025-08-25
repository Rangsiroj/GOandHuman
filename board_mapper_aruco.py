def get_board_position(x, y):
    grid_size = 450 / 18  # 19 ช่องมี 18 ระยะ
    col = round(x / grid_size)
    row = round(y / grid_size)
    if 0 <= col <= 18 and 0 <= row <= 18:
        col_letter = chr(ord('A') + col + (1 if col >= 8 else 0))  # ข้าม I
        return f"{col_letter}{19 - row}"
    return None