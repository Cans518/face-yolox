import cv2
import mediapipe as mp
import math

class Moving(object):

    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
    
    def moving(self,image):
        self.holistic = self.mp_holistic.Holistic(static_image_mode=False)
        # 以box中数据裁剪图像
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)

        action_name = "Unknown"

        # 判断举起双手过头顶动作
        if (results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_WRIST].y < results.pose_landmarks.landmark[
            self.mp_holistic.PoseLandmark.NOSE].y) and \
                (results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_WRIST].y < results.pose_landmarks.landmark[
                    self.mp_holistic.PoseLandmark.NOSE].y):
            if action_name == "Unknown":
                action_name = "Hands up"
            else:
                action_name = action_name + " and Hands up"

        # 优化下蹲的识别
        if results.pose_landmarks:
            left_hip = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_HIP]
            left_knee = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_KNEE]
            left_ankle = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_ANKLE]
            right_hip = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_HIP]
            right_knee = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_KNEE]
            right_ankle = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_ANKLE]

            # 计算左腿夹角
            left_thigh_dot_product = (left_knee.x - left_hip.x) * (left_knee.x - left_ankle.x) + (
                left_knee.y - left_hip.y) * (left_knee.y - left_ankle.y)
            left_thigh_norm_product = (math.sqrt((left_knee.x - left_hip.x) ** 2 + (left_knee.y - left_hip.y) ** 2)) * (
                math.sqrt((left_knee.x - left_ankle.x) ** 2 + (left_knee.y - left_ankle.y) ** 2))
            if left_thigh_norm_product != 0:  # 避免除零错误
                left_thigh_angle = math.degrees(
                    math.acos(min(max(left_thigh_dot_product / left_thigh_norm_product, -1), 1)))

            # 计算右腿夹角
            right_thigh_dot_product = (right_knee.x - right_hip.x) * (right_knee.x - right_ankle.x) + (
                    right_knee.y - right_hip.y) * (right_knee.y - right_ankle.y)
            right_thigh_norm_product = (math.sqrt(
                (right_knee.x - right_hip.x) ** 2 + (right_knee.y - right_hip.y) ** 2)) * (math.sqrt(
                (right_knee.x - right_ankle.x) ** 2 + (right_knee.y - right_ankle.y) ** 2))
            if right_thigh_norm_product != 0:  # 避免除零错误
                right_thigh_angle = math.degrees(
                    math.acos(min(max(right_thigh_dot_product / right_thigh_norm_product, -1), 1)))

            if left_thigh_angle < 90 and right_thigh_angle < 90:
                if action_name == "Unknown":
                    action_name = "Squat"
                else:
                    action_name = action_name + " and Squat"

        # 判断胸前单手比剪刀手动作
        right_scissor_hand = False
        left_scissor_hand = False

        # 判断右手剪刀手动作
        if results.right_hand_landmarks:
            right_hand = results.right_hand_landmarks
            right_thumb_tip = right_hand.landmark[self.mp_holistic.HandLandmark.THUMB_TIP]
            right_pinky_tip = right_hand.landmark[self.mp_holistic.HandLandmark.PINKY_TIP]
            right_index_finger_tip = right_hand.landmark[self.mp_holistic.HandLandmark.INDEX_FINGER_TIP]
            right_middle_finger_tip = right_hand.landmark[self.mp_holistic.HandLandmark.MIDDLE_FINGER_TIP]
            right_index_finger_dip = right_hand.landmark[self.mp_holistic.HandLandmark.INDEX_FINGER_DIP]
            right_middle_finger_dip = right_hand.landmark[self.mp_holistic.HandLandmark.MIDDLE_FINGER_DIP]
            right_index_finger_pip = right_hand.landmark[self.mp_holistic.HandLandmark.INDEX_FINGER_PIP]
            right_middle_finger_pip = right_hand.landmark[self.mp_holistic.HandLandmark.MIDDLE_FINGER_PIP]
            right_index_finger_mcp = right_hand.landmark[self.mp_holistic.HandLandmark.INDEX_FINGER_MCP]
            right_middle_finger_mcp = right_hand.landmark[self.mp_holistic.HandLandmark.MIDDLE_FINGER_MCP]

            right_thumb_pinky_distance = ((right_thumb_tip.x - right_pinky_tip.x) ** 2 + (
                    right_thumb_tip.y - right_pinky_tip.y) ** 2) ** 0.5
            if right_thumb_pinky_distance < 1:  # 可根据实际情况调整阈值
                if (right_index_finger_mcp.y < right_hand.landmark[self.mp_holistic.HandLandmark.WRIST].y) and \
                (right_middle_finger_mcp.y < right_hand.landmark[self.mp_holistic.HandLandmark.WRIST].y)and \
                        abs(right_index_finger_dip.y - right_index_finger_pip.y - right_index_finger_mcp.y) < 1 and \
                        abs(right_middle_finger_dip.y - right_middle_finger_pip.y - right_middle_finger_mcp.y) < 1:
                    right_scissor_hand = True

        # 判断左手剪刀手动作
        if results.left_hand_landmarks:
            left_hand = results.left_hand_landmarks
            left_thumb_tip = left_hand.landmark[self.mp_holistic.HandLandmark.THUMB_TIP]
            left_pinky_tip = left_hand.landmark[self.mp_holistic.HandLandmark.PINKY_TIP]
            left_index_finger_tip = left_hand.landmark[self.mp_holistic.HandLandmark.INDEX_FINGER_TIP]
            left_middle_finger_tip = left_hand.landmark[self.mp_holistic.HandLandmark.MIDDLE_FINGER_TIP]
            left_index_finger_dip = left_hand.landmark[self.mp_holistic.HandLandmark.INDEX_FINGER_DIP]
            left_middle_finger_dip = left_hand.landmark[self.mp_holistic.HandLandmark.MIDDLE_FINGER_DIP]
            left_index_finger_pip = left_hand.landmark[self.mp_holistic.HandLandmark.INDEX_FINGER_PIP]
            left_middle_finger_pip = left_hand.landmark[self.mp_holistic.HandLandmark.MIDDLE_FINGER_PIP]
            left_index_finger_mcp = left_hand.landmark[self.mp_holistic.HandLandmark.INDEX_FINGER_MCP]
            left_middle_finger_mcp = left_hand.landmark[self.mp_holistic.HandLandmark.MIDDLE_FINGER_MCP]

            left_thumb_pinky_distance = ((left_thumb_tip.x - left_pinky_tip.x) ** 2 + (
                    left_thumb_tip.y - left_pinky_tip.y) ** 2) ** 0.5
            if left_thumb_pinky_distance < 0.1:  # 可根据实际情况调整阈值
                if (left_index_finger_mcp.y < left_hand.landmark[self.mp_holistic.HandLandmark.WRIST].y) and \
                (left_middle_finger_mcp.y < left_hand.landmark[self.mp_holistic.HandLandmark.WRIST].y) and \
                        abs(left_index_finger_dip.y - left_index_finger_pip.y - left_index_finger_mcp.y) < 1 and \
                        abs(left_middle_finger_dip.y - left_middle_finger_pip.y - left_middle_finger_mcp.y) < 1:
                    left_scissor_hand = True

        if right_scissor_hand and not left_scissor_hand:
            if action_name == "Unknown":
                action_name = "right Scissor hand"
            else:
                action_name = action_name + " and right Scissor hand"
        elif left_scissor_hand and not right_scissor_hand:
            if action_name == "Unknown":
                action_name = "left Scissor hand"
            else:
                action_name = action_name + " and left Scissor hand"
    
        return action_name






