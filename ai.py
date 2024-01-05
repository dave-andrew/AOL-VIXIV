import cv2
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import time


def create_emotion_model(input_shape=(48, 48, 1), num_classes=3, learning_rate=0.001):

    if os.path.exists('emotion_model'):
        model = tf.keras.models.load_model('emotion_model')
        print("Model emosi berhasil dimuat.")
    else:
        model = keras.Sequential([
            layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
    
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
    
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
    return model

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_emotion(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_img = np.expand_dims(face_img, axis=-1)
    face_img = np.expand_dims(face_img, axis=0)
    face_img = face_img / 255.0

    emotions = emotion_model.predict(face_img)
    emotion_label = np.argmax(emotions)
    emotion_mapping = {0: 'Happy', 1: 'Netral', 2: 'Sad'}
    detected_emotion = emotion_mapping[emotion_label]
    print(emotions, detected_emotion)

    return detected_emotion

storage = './assets/train/'
persons = os.listdir(storage)

face_list = []
emotion_face_list = []
name_list = []
emotion_list = []

for id, person in enumerate(persons):
    for e_id, emotion in enumerate(os.listdir(f'{storage}/{person}')):
        for img_path in os.listdir(f'{storage}/{person}/{emotion}'):
            path = f'{storage}/{person}/{emotion}/{img_path}'
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            faces = face_cascade.detectMultiScale(gray, 1.1, 3)
            emotion_list.append(e_id)
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
            if person != 'other':
                face_list.append(face_img)
                name_list.append(id)
                emotion_face_list.append(face_img)
                print(f'{path} done person')
            else:
                if not os.path.exists('emotion_model'):
                    emotion_face_list.append(face_img)
                    print(f'{path} done emotion')
                else:
                    break

face_detector = cv2.face.LBPHFaceRecognizer_create()
face_detector.train(face_list, np.array(name_list))
print(len(face_list), len(name_list), len(emotion_list))

emotion_model = create_emotion_model()

if not os.path.exists('emotion_model'):
    img_list = []
    for img in emotion_face_list:
        img = cv2.resize(img, (48, 48))
        img = np.expand_dims(img, -1)
        img = img / 255.0
        img_list.append(img)

    combined_data = list(zip(img_list, emotion_list))

    train_data, test_data = train_test_split(combined_data, test_size=0.2, random_state=42)

    train_img_array, train_emotion_array = zip(*train_data)
    train_img_array = np.array(train_img_array)
    train_emotion_array = np.array(train_emotion_array)

    test_img_array, test_emotion_array = zip(*test_data)
    test_img_array = np.array(test_img_array)
    test_emotion_array = np.array(test_emotion_array)

    emotion_model.fit(train_img_array, train_emotion_array, epochs=100, validation_split=0.2, batch_size=512)
    test_loss, test_accuracy = emotion_model.evaluate(test_img_array, test_emotion_array)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.2f}")

    emotion_model.save('emotion_model')
    print("Model emosi berhasil disimpan.")


cap = cv2.VideoCapture(0)
last_emotion_time = time.time()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        label, confidence = face_detector.predict(face_img)
        same = int(100 * (1 - (confidence) / 300))

        current_time = time.time()
        if current_time - last_emotion_time >= 5:
            emotion = detect_emotion(face_img)
            last_emotion_time = current_time 
        
        text = f'{persons[label]} - {emotion} - {same}%'
        
        if confidence < 70:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Unknown - {emotion}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('p'):
        cap.release()
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()