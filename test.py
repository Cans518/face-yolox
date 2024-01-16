# -*- coding: utf-8 -*-
from Person import Person
import cv2
import json

if __name__ == '__main__':
    person = Person()
    img = input('Input image filename:')
    try:
        image = cv2.imread(img)
    except:
        print('Open Error! Try again!')
    http_image, http_name = person.detect_image(image)
    json_data = {
        "eventId": "7180423a",
        "resultImg": http_image,
        "info": http_name
    }
    json_data = json.dumps(json_data, ensure_ascii=False)
    # 写入json文件
    with open('result.json', 'w',encoding='utf-8') as f:
        f.write(json_data)
        

