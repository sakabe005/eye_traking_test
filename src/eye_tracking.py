import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import csv


# Webカメラのセットアップ
cap = cv2.VideoCapture(0)

with open('iris_landmarks.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # CSVのヘッダーを定義
    writer.writerow(['Frame', 'Left_X', 'Left_Y', 'Right_X', 'Right_Y'])

    # Mediapipeの初期化
    mp_face_mesh = mp.solutions.face_mesh
    #hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # 虹彩ランドマークのIDリスト
    left_iris_ids = [474, 475, 476, 477]
    right_iris_ids = [469, 470, 471, 472]
    frame_count = 0
    
    iris = {
        'left': {'x': [], 'y': []},
        'right': {'x': [], 'y': []}
    }
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
    
        #img = cv2.flip(img, 1)  # 映像を左右反転
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Mediapipeで手のランドマークを推論
        results = face_mesh.process(img_rgb)
    
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = img.shape
                
                left_iris_X = 0
                left_iris_Y = 0
                right_iris_X = 0
                right_iris_Y = 0
                for id, lm in enumerate(face_landmarks.landmark):
                    if id in left_iris_ids:
                        cx, cy =int(lm.x * w), int(lm.y * h)
                        left_iris_X += cx
                        left_iris_Y += cy
                        
                    elif id in right_iris_ids:
                        cx, cy =int(lm.x * w), int(lm.y * h)
                        right_iris_X += cx
                        right_iris_Y += cy
                        
        writer.writerow([frame_count, left_iris_X / 4, left_iris_Y / 4, right_iris_X / 4, right_iris_Y / 4])
        iris['left']['x'].append(left_iris_X / 4)
        iris['left']['y'].append(left_iris_Y / 4)

        iris['right']['x'].append(right_iris_X / 4)
        iris['right']['y'].append(right_iris_Y / 4)

        frame_count += 1

        # 映像を表示
        cv2.imshow("Image", img)

        # 「c」キーで終了
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.destroyWindow('Image')
            break
plt.plot(iris['left']['x'], iris['left']['y'], linestyle="solid", color=(0, 0, 1), label="左目")
plt.plot(iris['right']['x'], iris['right']['y'], linestyle="solid", color=(0, 1, 0), label="右目")
plt.show()       
