import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import math

IMG_SIZE = 100         
OFFSET = 40            
MODEL_PATH = 'moj_model_migowy.h5' 
class_names = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] 

print("Ładowanie modelu...")
model = tf.keras.models.load_model(MODEL_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success: break

    imgOutput = img.copy()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(imgOutput, hand_lms, mp_hands.HAND_CONNECTIONS)

            h, w, c = img.shape
            
            x_vals = [lm.x for lm in hand_lms.landmark]
            y_vals = [lm.y for lm in hand_lms.landmark]

            x_min, x_max = int(min(x_vals) * w), int(max(x_vals) * w)
            y_min, y_max = int(min(y_vals) * h), int(max(y_vals) * h)

            x1 = max(0, x_min - OFFSET)
            y1 = max(0, y_min - OFFSET)
            x2 = min(w, x_max + OFFSET)
            y2 = min(h, y_max + OFFSET)

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                
                imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)

                imgResize = cv2.resize(imgGray, (IMG_SIZE, IMG_SIZE))
                
                imgFinal = np.expand_dims(imgResize, axis=0) 
                imgFinal = np.expand_dims(imgFinal, axis=-1) 

                prediction = model.predict(imgFinal, verbose=0)
                index = np.argmax(prediction) 
                label = class_names[index]
                confidence = prediction[0][index]

                cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (255, 0, 255), 4)
                cv2.putText(imgOutput, f'{label} {int(confidence*100)}%', 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

                cv2.imshow("Widok AI", imgResize)

    cv2.imshow("Kamera", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()