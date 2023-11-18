from PIL import Image
from yolo import YOLO
import numpy as np
from retinaface import Retinaface
import cv2
import moving as m

if __name__ == "__main__":
    yolo = YOLO()
    retinaface = Retinaface()
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
    persenboxes = yolo.detect_image(image)
    info = []
    for i in range(len(persenboxes)):
        imgs = image.crop((persenboxes[i]))
        name,box = retinaface.detect_image(imgs)
        x,y = persenboxes[i][0],persenboxes[i][1]
        box = box[0]+x,box[1]+y,box[2]+x,box[3]+y
        # imgs转化为cv2的形式
        imgs = np.array(imgs)
        action_name = m.moving(img,persenboxes[i])
        name = name + ' - ' + action_name
        info.append([name,box])
    # 重新打开图片
    image = cv2.imread(img)
    # 利用info进行绘制
    for i in range(len(info)):
        name,box = info[i]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 10)
        # cv2输出
        image = retinaface.cv2ImgAddText(image, name, box[0] + 20, box[1] - 40 ,(255, 255, 0),40)
    # 保存绘制好的图片
    cv2.imwrite(f'{img}_result.jpeg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 60])