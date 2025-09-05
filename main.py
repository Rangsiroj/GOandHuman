from vision_aruco import VisionSystem
from vision_aruco_YOLO import VisionSystemYOLO
from vision_manual import VisionManual

if __name__ == "__main__":
    # camera = VisionSystem()
    camera = VisionSystemYOLO(weights_path=r"C:\Users\acer\OneDrive\ProjectFinal\GoandHuman\models\best.pt")
    # camera = VisionManual()
    camera.run()