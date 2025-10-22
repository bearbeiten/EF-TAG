import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

class GestureDetectorApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Hand Gesture Detector")
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        # Create GUI elements
        self.video_label = tk.Label(window)
        self.video_label.pack(pady=10)
        
        self.gesture_label = tk.Label(window, text="Gesture: None", font=("Arial", 16))
        self.gesture_label.pack(pady=5)
        
        self.confidence_label = tk.Label(window, text="Confidence: N/A", font=("Arial", 14))
        self.confidence_label.pack(pady=5)
        
        self.quit_button = tk.Button(window, text="Quit", command=self.quit_app, font=("Arial", 12))
        self.quit_button.pack(pady=10)
        
        # Start video loop
        self.update_frame()
    
    def detect_gesture(self, hand_landmarks):
        """Detect basic gestures based on finger positions"""
        # Get landmark positions
        landmarks = hand_landmarks.landmark
        
        # Count extended fingers
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        finger_pips = [6, 10, 14, 18]
        thumb_tip = 4
        thumb_ip = 3
        wrist = 0
        
        fingers_up = 0
        fingers_status = []  # Track which fingers are up
        
        # Check thumb (different logic for thumb)
        thumb_up = landmarks[thumb_tip].x < landmarks[thumb_ip].x
        if thumb_up:
            fingers_up += 1
        fingers_status.append(thumb_up)
        
        # Check other fingers
        for tip, pip in zip(finger_tips, finger_pips):
            finger_extended = landmarks[tip].y < landmarks[pip].y
            if finger_extended:
                fingers_up += 1
            fingers_status.append(finger_extended)
        
        # Check for thumbs up: thumb up, all other fingers down, thumb above wrist
        if fingers_status[0] and not any(fingers_status[1:]) and landmarks[thumb_tip].y < landmarks[wrist].y:
            return "Thumbs Up", 0.92
        
        # Check for thumbs down: thumb down, all other fingers closed, thumb below palm
        if not fingers_status[0] and not any(fingers_status[1:]) and landmarks[thumb_tip].y > landmarks[wrist].y:
            return "Thumbs Down", 0.92
        
        # Check for middle finger: only middle finger up (index 2 in fingers_status)
        if not fingers_status[0] and not fingers_status[1] and fingers_status[2] and not fingers_status[3] and not fingers_status[4]:
            return "Middle Finger", 0.88
        
        # Determine gesture
        if fingers_up == 0:
            return "Fist", 0.9
        elif fingers_up == 1:
            return "One Finger", 0.85
        elif fingers_up == 2:
            return "Peace/Two", 0.85
        elif fingers_up == 3:
            return "Three Fingers", 0.85
        elif fingers_up == 4:
            return "Four Fingers", 0.85
        elif fingers_up == 5:
            return "Open Hand", 0.9
        else:
            return "Unknown", 0.5
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = self.hands.process(rgb_frame)
            
            gesture_text = "None"
            confidence_text = "N/A"
            
            # Draw hand landmarks and detect gestures
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        rgb_frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Detect gesture
                    gesture, confidence = self.detect_gesture(hand_landmarks)
                    gesture_text = gesture
                    confidence_text = f"{confidence:.2%}"
            
            # Update labels
            self.gesture_label.config(text=f"Gesture: {gesture_text}")
            self.confidence_label.config(text=f"Confidence: {confidence_text}")
            
            # Convert to PhotoImage for tkinter
            img = Image.fromarray(rgb_frame)
            img = img.resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        # Schedule next update
        self.window.after(10, self.update_frame)
    
    def quit_app(self):
        self.cap.release()
        self.hands.close()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureDetectorApp(root)
    root.mainloop()
