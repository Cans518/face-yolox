# -*- coding: utf-8 -*-
from PIL import Image
from yolo import YOLO
from moving import Moving
import numpy as np
from retinaface import Retinaface
import cv2
import sys
import base64


class Person(object):
    def __init__(self):
        self.yolo = YOLO()
        self.retinaface = Retinaface()
        self.moving = Moving()

    def detect_image(self,image):
        # 转换为PIL格式
        s_image = Image.fromarray(image)
        persenboxes = self.yolo.detect_image(s_image)
        name_date = []
        info = []
        for i in range(len(persenboxes)):
            imgs = s_image.crop((persenboxes[i]))
            name,box = self.retinaface.detect_image(imgs)
            x,y = persenboxes[i][0],persenboxes[i][1]
            box = box[0]+x,box[1]+y,box[2]+x,box[3]+y
            # imgs转化为cv2的形式
            imgs = np.array(imgs)
            action_name = self.moving.moving(imgs)
            name_date.append({"resultName": name, "resultAction": action_name})
            name = name + ' - ' + action_name
            info.append([name,box])
        # 利用info进行绘制
        for i in range(len(info)):
            name,box = info[i]
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 10)
            # cv2输出
            image = self.retinaface.cv2ImgAddText(image, name, box[0] + 20, box[1] - 40 ,(255, 255, 0),40)
        # 转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 将图片保存为.jpeg格式
        cv2.imwrite('output.jpeg', image)
        # 保存绘制好的图片为jpeg
        jpeg_data = cv2.imencode(".jpg", image)[1]
        # 将图片转化为base64
        base64_data = base64.b64encode(jpeg_data)
        # 将图片转化为字符串
        base64_data = base64_data.decode('utf-8')
        base64_data = 'data:image/jpeg;base64,' + base64_data
        return base64_data,name_date