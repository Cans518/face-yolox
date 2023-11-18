import cv2
import sys
from retinaface import Retinaface

if __name__ == "__main__":
    retinaface = Retinaface()
    img = sys.argv[1]
    output_dir = img.split(".")[0] + "_out." + img.split(".")[1]
    image = cv2.imread(img)
    if image is None:
        print('Open Error! Try again!')
    else:
        retinaface.detect_image(image)
        print('\nDone')