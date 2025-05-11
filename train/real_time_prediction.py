# okay detection 4/10, 8:24

import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import pyttsx3
import difflib

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Dictionary of common words for autocorrection
COMMON_WORDS = [
    'hello', 'world', 'thank', 'you', 'please', 'sorry', 'help', 'name', 'how', 'are',
    'good', 'morning', 'afternoon', 'evening', 'night', 'bye', 'yes', 'no', 'what',
    'where', 'when', 'why', 'who', 'which', 'this', 'that', 'these', 'those'
]

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Buffer to store letters
letter_buffer = deque(maxlen=20)  # Store up to 20 letters
last_letter = None
letter_counter = 0
letter_threshold = 10  # Number of frames to wait before adding a new letter
last_spoken_letter = None

def autocorrect_word(word):
    if not word:
        return word
    # Find the closest match from common words
    matches = difflib.get_close_matches(word.lower(), COMMON_WORDS, n=1, cutoff=0.6)
    return matches[0] if matches else word

def speak_letter(letter):
    global last_spoken_letter
    if letter != last_spoken_letter:
        engine.say(letter)
        engine.runAndWait()
        last_spoken_letter = letter

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    # Add a semi-transparent overlay for the word display
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # Only use the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw landmarks for the first hand only
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Process landmarks for the first hand only
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y

            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        try:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Add letter to buffer if it's different from the last one and we've waited enough frames
            if predicted_character != last_letter:
                letter_counter = 0
                last_letter = predicted_character
            else:
                letter_counter += 1
                if letter_counter >= letter_threshold:
                    letter_buffer.append(predicted_character)
                    speak_letter(predicted_character)  # Speak the new letter
                    letter_counter = 0
                    last_letter = None  # Reset last_letter to allow repeating letters

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            continue

    # Get current word and its autocorrected version
    current_word = ''.join(letter_buffer)
    corrected_word = autocorrect_word(current_word)

    # Display the current word and its correction
    cv2.putText(frame, f"Word: {current_word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if corrected_word != current_word:
        cv2.putText(frame, f"Suggested: {corrected_word}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display instructions
    cv2.putText(frame, "Press 'c' to clear word | 'b' for backspace | 'q' to quit | 's' to speak word", (10, H - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('ASL Fingerspelling', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        letter_buffer.clear()
        last_letter = None
        letter_counter = 0
        last_spoken_letter = None
    elif key == ord('b'):  # Backspace
        if letter_buffer:
            letter_buffer.pop()
            last_letter = None
            letter_counter = 0
    elif key == ord('s'):
        if current_word:
            engine.say(f"The word is {current_word}")
            engine.runAndWait()

cap.release()
cv2.destroyAllWindows()
