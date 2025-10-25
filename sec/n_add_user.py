import cv2
import numpy as np
import sqlite3
from cryptography.fernet import Fernet
import pickle
import os
from datetime import datetime
from insightface.app import FaceAnalysis
import time

# Load encryption key
KEY_FILE = "secret.key"
if not os.path.exists(KEY_FILE):
    print(
        "[ERROR] Encryption key not found. Please run the main script first to generate it."
    )
    exit(1)

with open(KEY_FILE, "rb") as f:
    key = f.read()
print("[info] loaded encryption key")

cipher = Fernet(key)

# Initialize face recognition model
print("[info] initializing insightface FaceAnalysis...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
print("[info] model loaded successfully!")


# Initialize database
def init_db():
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()
    conn.close()


# Store encrypted embedding
def store_face(name, embedding):
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()

    # Serialize and encrypt embedding
    embedding_bytes = pickle.dumps(embedding)
    encrypted_embedding = cipher.encrypt(embedding_bytes)

    cursor.execute(
        "INSERT INTO faces (name, embedding, created_at) VALUES (?, ?, ?)",
        (name, encrypted_embedding, datetime.now()),
    )
    conn.commit()
    conn.close()
    print(f"[SUCCESS] Stored face for '{name}'")


def main():
    # Initialize database
    init_db()

    # Get user name
    print("\n" + "=" * 50)
    print("USER REGISTRATION SYSTEM")
    print("=" * 50)
    user_name = input("\nEnter user name (or press Enter to skip): ").strip()

    if not user_name:
        print("[INFO] No name entered. Exiting registration.")
        return

    print(f"\n[INFO] Starting registration for: {user_name}")
    print("[INFO] Position your face in front of the camera")
    print("[INFO] Press '1' to capture and save your face")
    print("[INFO] Press 'q' to quit\n")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    face_detected = False
    current_embedding = None
    detection_cooldown = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break

            display_frame = frame.copy()

            # Process face detection every few frames
            if detection_cooldown <= 0:
                # Detect and get face embeddings
                faces = app.get(frame)

                if len(faces) > 0:
                    face_detected = True
                    face = faces[0]  # Use first detected face
                    current_embedding = face.embedding

                    # Draw bounding box
                    bbox = face.bbox.astype(int)
                    cv2.rectangle(
                        display_frame,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        (0, 255, 0),
                        3,
                    )

                    # Draw face detected message
                    cv2.putText(
                        display_frame,
                        f"Face Detected - {user_name}",
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                else:
                    face_detected = False
                    current_embedding = None

                detection_cooldown = 5  # Process every 5 frames
            else:
                detection_cooldown -= 1

            # Display instructions
            status_y = 30
            if face_detected:
                cv2.putText(
                    display_frame,
                    "Press '1' to SAVE this face",
                    (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                status_y += 35
                cv2.putText(
                    display_frame,
                    "Face ready for registration!",
                    (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    display_frame,
                    "No face detected - position yourself",
                    (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            status_y += 35
            cv2.putText(
                display_frame,
                "Press 'q' to QUIT",
                (10, status_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(f"User Registration - {user_name}", display_frame)

            key = cv2.waitKey(1) & 0xFF

            # Press '1' to save the face
            if key == ord("1"):
                if face_detected and current_embedding is not None:
                    store_face(user_name, current_embedding)
                    print(f"\n[SUCCESS] User '{user_name}' registered successfully!")
                    print("[INFO] You can now use the main recognition system")
                    time.sleep(2)  # Give user time to see the success message
                    break
                else:
                    print(
                        "[WARNING] No face detected! Position your face in front of the camera."
                    )

            # Press 'q' to quit
            if key == ord("q"):
                print("[INFO] Registration cancelled")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
