import numpy as np
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# GestureRecognizer-Objekt erstellen basieren auf heruntergeladenem Modell
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Initialize MediaPipe drawing utilities for landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def detect_gestures_in_halves(frame, recognizer):
    """Process frame and detect gestures in left and right halves independently using GestureRecognizer"""
    h, w = frame.shape[:2]
    mid_x = w // 2
    
    # Initialize results for both halves
    left_gesture = "None"
    left_confidence = "N/A"
    right_gesture = "None"
    right_confidence = "N/A"
    
    # Draw vertical line to split screen
    cv2.line(frame, (mid_x, 0), (mid_x, h), (0, 255, 0), 2)
    
    # Process LEFT half
    left_half = frame[:, :mid_x].copy()
    rgb_left = cv2.cvtColor(left_half, cv2.COLOR_BGR2RGB)
    mp_image_left = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_left)
    left_result = recognizer.recognize(mp_image_left)
    
    # Process detected gestures in left half
    if left_result.gestures:
        gesture_list = left_result.gestures[0]
        hand_landmarks = left_result.hand_landmarks[0]
        
        # Draw hand landmarks manually on left half
        for landmark in hand_landmarks:
            x = int(landmark.x * mid_x)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Draw connections between landmarks
        connections = mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_x = int(hand_landmarks[start_idx].x * mid_x)
            start_y = int(hand_landmarks[start_idx].y * h)
            end_x = int(hand_landmarks[end_idx].x * mid_x)
            end_y = int(hand_landmarks[end_idx].y * h)
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)
        
        # Get the top gesture and its confidence
        if gesture_list:
            top_gesture = gesture_list[0]
            left_gesture = top_gesture.category_name
            left_confidence = f"{top_gesture.score:.2%}"
    
    # Process RIGHT half
    right_half = frame[:, mid_x:].copy()
    rgb_right = cv2.cvtColor(right_half, cv2.COLOR_BGR2RGB)
    mp_image_right = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_right)
    right_result = recognizer.recognize(mp_image_right)
    
    # Process detected gestures in right half
    if right_result.gestures:
        gesture_list = right_result.gestures[0]
        hand_landmarks = right_result.hand_landmarks[0]
        
        # Draw hand landmarks manually on right half (offset by mid_x)
        for landmark in hand_landmarks:
            x = int(landmark.x * mid_x) + mid_x
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Draw connections between landmarks
        connections = mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_x = int(hand_landmarks[start_idx].x * mid_x) + mid_x
            start_y = int(hand_landmarks[start_idx].y * h)
            end_x = int(hand_landmarks[end_idx].x * mid_x) + mid_x
            end_y = int(hand_landmarks[end_idx].y * h)
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)
        
        # Get the top gesture and its confidence
        if gesture_list:
            top_gesture = gesture_list[0]
            right_gesture = top_gesture.category_name
            right_confidence = f"{top_gesture.score:.2%}"
    
    # Add text labels for each half
    cv2.putText(frame, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Gesture: {left_gesture}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Confidence: {left_confidence}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.putText(frame, "RIGHT", (mid_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Gesture: {right_gesture}", (mid_x + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Confidence: {right_confidence}", (mid_x + 10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame, left_gesture, right_gesture

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Process frame and detect gestures in both halves
    frame, left_gesture, right_gesture = detect_gestures_in_halves(frame, recognizer)
    
    # Display the frame
    cv2.imshow('Dual Gesture Recognition', frame)
    
    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()