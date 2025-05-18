from vision_aruco import VisionSystem
from vision_manual import VisionManual

if __name__ == "__main__":
    # camera = VisionSystem()
    camera = VisionManual()
    camera.run()