from paho.mqtt import client as mqtt
from Person import Person
import requests
import uuid
import cv2
import json

person = Person()
http_data = "null"
http_url = "http://192.168.1.100:18099/rtb/result"
headers = {
    "Content-Type": "application/json",
}


def capture_camera_frame(rtsp_url):
    # 截取摄像头的帧
    cap = cv2.VideoCapture(rtsp_url,cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("无法打开摄像头")
        return None
    ret, frame = cap.read()
    cap.release()
    return frame

def on_connect(client, userdata, flags, rc):
    """一旦连接成功, 回调此方法"""
    rc_status = ["连接成功", "协议版本不正确", "客户端标识符无效", "服务器不可用", "用户名或密码不正确", "未经授权"]
    print("连接结果：", rc_status[rc])
    if rc == 0:
        print("成功连接到MQTT服务器")
        # 在连接成功后订阅主题
        print("订阅主题...")
        client.subscribe("nssol/rtb/sensor1", 2)  # 订阅第一个主题
        print("成功订阅主题：nssol/rtb/sensor1 ")
        print("等待接收消息...")
    else:
        print("连接失败")

def on_message(client, userdata, msg):
    try:
        print(f"接收到消息 - 主题: {msg.topic}，内容: {msg.payload.decode('utf-8')}")
        process_mqtt_message(msg.topic, msg.payload)
    except Exception as e:
        print(f"处理消息时发生错误: {e}")

def process_mqtt_message(topic, payload):
    try:
        data = json.loads(payload)
        device_id = data.get("device_id")
        if topic == "nssol/rtb/sensor1":
            rtsp_url = "rtsp://camera65:nssol2023@nat.reimu.site:39665/ch1/main/av_stream"
            # 使用cv2读取摄像头数据
            frame = capture_camera_frame(rtsp_url)
            if frame is not None:
                http_image,http_name = person.detect_image(frame)
                json_data = {
                "teamName": "myteam",
                "eventId": "7180423a",
                "resultImg": http_image,
                "info": http_name
                }
                http_data = json.dumps(json_data, ensure_ascii=False)
                response = requests.post(http_url, headers=headers, data=http_data)
                if response.status_code == 200:
                    print("成功发送数据")
                else:
                    print("发送数据失败")

    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
    except Exception as e:
        print(f"处理消息时发生错误: {e}")

def mqtt_connect():
    """连接MQTT服务器并订阅主题"""
    print("开始运行MQTT订阅程序")
    print("正在连接MQTT服务器...")
    mqttClient = mqtt.Client(str(uuid.uuid4()))
    mqttClient.on_connect = on_connect  # 返回连接状态的回调函数
    mqttClient.on_message = on_message  # 返回订阅消息回调函数
    MQTTHOST = "broker.emqx.io"  # MQTT服务器地址
    MQTTPORT = 1883  # MQTT端口
    # mqttClient.username_pw_set("username", "password")  # mqtt服务器账号密码
    mqttClient.connect(MQTTHOST, MQTTPORT, 60)
    mqttClient.loop_forever()  # 保持连接状态

if __name__ == "__main__":
    mqtt_connect()
