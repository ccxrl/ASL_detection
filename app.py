from flask import Flask, render_template, Response, request, jsonify
import cv2
import pickle
import mediapipe as mp
import numpy as np
from collections import deque
import pyttsx3
import difflib
import nltk
from nltk.corpus import words
import threading
import time
import os
import base64

# Initialize Flask app
app = Flask(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# Global variables
COMMON_WORDS = words.words()
letter_buffer = deque(maxlen=20)
last_letter = None
letter_counter = 0
letter_threshold = 10
last_spoken_letter = None
current_word = ""
corrected_word = ""
predicted_character = ""
camera_active = False
speak_enabled = True

# Initialize text-to-speech engine
# Will create separate engine instances per thread for better reliability
engine = None  # We'll create engine instances in the speak_text function

# Load the model
try:
    model_path = os.path.join('ASL_detection', 'ASL_detection', 'model.p')
    if not os.path.exists(model_path):
        model_path = 'model.p'  # Fallback to current directory
    
    model_dict = pickle.load(open(model_path, 'rb'))
    model = model_dict['model']
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

def autocorrect_word(word):
    if not word:
        return word
    # Find the closest match from common words
    matches = difflib.get_close_matches(word.lower(), COMMON_WORDS, n=1, cutoff=0.6)
    return matches[0] if matches else word

def speak_text(text):
    """Thread-safe function to speak text"""
    if speak_enabled:
        # Create a new thread for TTS to avoid blocking
        # Use a direct approach to ensure the engine works reliably
        def tts_thread():
            try:
                # Create a local engine instance for this thread to avoid conflicts
                local_engine = pyttsx3.init()
                local_engine.setProperty('rate', 150)
                local_engine.say(text)
                local_engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {str(e)}")
        
        thread = threading.Thread(target=tts_thread)
        thread.daemon = True  # Set as daemon so it doesn't block program exit
        thread.start()

def process_frame(frame):
    global letter_buffer, last_letter, letter_counter, last_spoken_letter, current_word, corrected_word, predicted_character
    
    data_aux = []
    x_ = []
    y_ = []
    
    H, W, _ = frame.shape
    
    # Add overlay for text area
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        # Process landmarks
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
            if model is not None:
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
                        
                        # Speak every detected letter if speech is enabled
                        if speak_enabled:
                            speak_text(predicted_character)
                            last_spoken_letter = predicted_character
                            
                        letter_counter = 0
                        last_letter = None
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 122, 204), 4)  # Blue rectangle
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 122, 204), 3, cv2.LINE_AA)
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
    
    # Get current word and its autocorrected version
    current_word = ''.join(letter_buffer)
    corrected_word = autocorrect_word(current_word) if current_word else ""
    
    # Display the current word and its correction
    cv2.putText(frame, f"Word: {current_word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if corrected_word != current_word and corrected_word:
        cv2.putText(frame, f"Suggested: {corrected_word}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

def generate_frames():
    global camera_active
    
    camera = cv2.VideoCapture(0)
    camera_active = True
    
    try:
        while camera_active:
            success, frame = camera.read()
            if not success:
                break
            
            # Process the frame
            processed_frame = process_frame(frame)
            
            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.03)  # Limit frame rate
    finally:
        camera.release()
        camera_active = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/clear_word', methods=['POST'])
def clear_word():
    global letter_buffer, last_letter, letter_counter, last_spoken_letter
    letter_buffer.clear()
    last_letter = None
    letter_counter = 0
    last_spoken_letter = None
    return jsonify({"status": "success", "word": ""})

@app.route('/backspace', methods=['POST'])
def backspace():
    global letter_buffer, last_letter, letter_counter
    if letter_buffer:
        letter_buffer.pop()
        last_letter = None
        letter_counter = 0
    return jsonify({"status": "success", "word": ''.join(letter_buffer)})

@app.route('/speak_word', methods=['POST'])
def speak_word():
    word = ''.join(letter_buffer)
    if word:
        # Use the corrected word if available
        to_speak = autocorrect_word(word) if word else word
        if to_speak:
            # Using a more natural phrase
            speak_text(f"The word is {to_speak}")
    return jsonify({"status": "success"})

@app.route('/toggle_speech', methods=['POST'])
def toggle_speech():
    global speak_enabled
    speak_enabled = not speak_enabled
    return jsonify({"status": "success", "speak_enabled": speak_enabled})

@app.route('/get_current_data')
def get_current_data():
    return jsonify({
        "word": current_word,
        "corrected_word": corrected_word,
        "current_letter": predicted_character
    })

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active
    camera_active = False
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)