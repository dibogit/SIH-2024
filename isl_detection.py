import cv2
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import numpy as np
import pandas as pd
import string
import time
import os
from gtts import gTTS

# Load the saved model from file
model = keras.models.load_model("model.h5")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet += list(string.ascii_uppercase)

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    temp_landmark_list = [n / max_value for n in temp_landmark_list]
    return temp_landmark_list

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    os.system("start output.mp3")  # For Windows
    # os.system("afplay output.mp3")  # For macOS
    # os.system("xdg-open output.mp3")  # For Linux

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  
    recognized_text = ""  # To store the recognized characters
    last_detection_time = time.time()  # To store the time of the last detection
    max_chars_per_line = 30  # Maximum characters per line in the text box
    pause_duration = 2  # Pause duration for recognizing space between words
    speech_pause_duration = 5  # Pause duration to wait before converting text to speech

    while cap.isOpened():
        success, image = cap.read()
        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        debug_image = copy.deepcopy(image)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                df = pd.DataFrame(pre_processed_landmark_list).transpose()

                # Predict the sign language
                predictions = model.predict(df, verbose=0)
                predicted_classes = np.argmax(predictions, axis=1)
                label = alphabet[predicted_classes[0]]
                
                # Check for pause between signs
                current_time = time.time()
                if current_time - last_detection_time > pause_duration:
                    recognized_text += " "  # Add a space to separate words
                
                last_detection_time = current_time  # Update the last detection time
                
                # Add the recognized character to the text string
                recognized_text += label
                print(f"Recognized Text: {recognized_text}")
                
                # Display the recognized letter
                cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        
        # Draw a rectangle for the text box
        cv2.rectangle(image, (30, 400), (610, 470), (255, 255, 255), -1)
        
        # Handle text wrapping in the box
        y = 450  # Starting y position for the text
        lines = [recognized_text[i:i+max_chars_per_line] for i in range(0, len(recognized_text), max_chars_per_line)]
        for i, line in enumerate(lines):
            cv2.putText(image, line, (40, y + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Output image
        cv2.imshow('Indian sign language detector', image)
        
        # Convert the recognized text to speech after a pause
        if recognized_text and (current_time - last_detection_time > speech_pause_duration):
            text_to_speech(recognized_text)
            recognized_text = ""  # Clear the recognized text after speaking

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
