import tensorflow.keras
import numpy as np
import cv2
import time

model = tensorflow.keras.models.load_model('model/keras_model.h5')

cap = cv2.VideoCapture(1)
size = (224, 224)

video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (720, 720))

classes = ['Scissors', 'Rock', 'Paper', 'None']

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    h, w, _ = img.shape
    print(img.shape)  # 720 X 1280
    # print('img shape : ', img.shape[0])
    
    cx = h // 2
    cy = w // 2
    # img 720 X 1280
    # img = img[:, 100:100+img.shape[0]]
    img = img[:, 0:720]
    print('img', img.shape)
    
    # left_margin
    x1_left_margein = 170    # x : height
    y1_left_margin = 170     # y : width
    
    y1 = y1_left_margin
    y2 = img.shape[0] - y1_left_margin    
    x1 = x1_left_margein
    x2 = img.shape[1] - x1_left_margein    
    
    img = cv2.flip(img, 1)

    img_input = cv2.resize(img, size)
    # print('max', np.max(img_input[0]))
    # print('min', np.min(img_input[0]))
    
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    img_input = (img_input.astype(np.float32) / 127.0) - 1
    # print('max', np.max(img_input[0]))
    # print('min', np.min(img_input[0]))
    
    
    img_input = np.expand_dims(img_input, axis=0)
    # print('img_input : ', img_input.shape)

    prediction = model.predict(img_input)
    idx = np.argmax(prediction)
    threshold = np.max(prediction)
    threshold = round(threshold, 2)
    # print(threshold)
    
    # rectangle  
    rectangleColor              = (0, 255, 0)
    rectangleThickness          = 2
    cv2.rectangle(img, (x1,y1), (x2,y2), rectangleColor, rectangleThickness)

    cv2.putText(img, text=classes[idx], org=(30, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)
    cv2.putText(img, text=str(round(threshold*100, 2))+"%", org=(170, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                color=(255, 255, 255), thickness=2)

    cv2.imshow('result', img)
    # video.write(img)
    
    time.sleep(1 / 30)
    if cv2.waitKey(1) == ord('q'):
        break
