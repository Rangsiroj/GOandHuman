# def get_board_position(cx, cy, grid_size=18, image_size=500):
#     # """
#     # แปลงพิกัดภาพ (cx, cy) เป็นตำแหน่งกระดานหมากล้อม เช่น D4, E5

#     # Parameters:
#     #     - cx, cy: พิกัดศูนย์กลางของหมากในภาพหลัง Perspective Transform
#     #     - grid_size: จำนวนช่อง (ค่า default = 19)
#     #     - image_size: ขนาดภาพ (หลัง Perspective = 500x500 โดย default)

#     # Returns: 
#     #     - ตำแหน่งกระดานในรูปแบบ A1 ถึง T19 (ยกเว้น I) หรือ None ถ้าอยู่นอกกรอบ
#     # """
#     cell_size = image_size // grid_size
#     col = cx // cell_size
#     row = cy // cell_size

#     if col >= grid_size or row >= grid_size:
#         return None

#     # ข้ามตัวอักษร 'I' ตามมาตรฐานหมากล้อม
#     col_letter = chr(ord('A') + col)
#     if col_letter >= 'I':
#         col_letter = chr(ord(col_letter) + 1)

#     row_number = grid_size - row
#     return f"{col_letter}{row_number}"

def get_board_position(x, y):
    grid_size = 450 / 18  # 19 ช่องมี 18 ระยะ
    col = round(x / grid_size)
    row = round(y / grid_size)
    if 0 <= col <= 18 and 0 <= row <= 18:
        col_letter = chr(ord('A') + col + (1 if col >= 8 else 0))  # ข้าม I
        return f"{col_letter}{19 - row}"
    return None