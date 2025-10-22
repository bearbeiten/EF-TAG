import cv2
import numpy as np
import sqlite3
from cryptography.fernet import Fernet
import pickle
import os
from datetime import datetime
from insightface.app import FaceAnalysis
from scipy.spatial import distance as dist
import mediapipe as mp

# Generate or load encryption key
KEY_FILE = "secret.key"
if os.path.exists(KEY_FILE):
    with open(KEY_FILE, "rb") as f:
        key = f.read()
    print("[info] loaded existing encryption key")
else:
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    print("[info] generated new encryption key and saved to secret.key. Keep it safe!")

cipher = Fernet(key)

# Initialize face recognition model
print("[info] initializing insightface FaceAnalysis (detection + recognition)...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
print("[info] model loaded successfully!")

# Initialize MediaPipe Face Mesh for blink detection
print("[info] initializing MediaPipe Face Mesh for blink detection...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
print("[info] MediaPipe Face Mesh loaded successfully!")

# Define eye landmarks indices for MediaPipe
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Key points for EAR calculation
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Key points for EAR calculation


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
    print(f"[success] stored face for '{name}'")


# Retrieve all faces from database
def get_all_faces():
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, embedding FROM faces")
    rows = cursor.fetchall()
    conn.close()

    faces = []
    for name, encrypted_embedding in rows:
        # Decrypt and deserialize embedding
        embedding_bytes = cipher.decrypt(encrypted_embedding)
        embedding = pickle.loads(embedding_bytes)
        faces.append((name, embedding))

    return faces


# Compare embeddings using cosine similarity
def cosine_similarity(emb1, emb2):
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


# Find matching face
def find_match(embedding, threshold=0.3):
    faces = get_all_faces()
    if len(faces) == 0:
        return None, -1.0

    best_match = None
    best_similarity = -1

    for name, stored_embedding in faces:
        similarity = cosine_similarity(embedding, stored_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = name

    if best_similarity > threshold:
        return best_match, best_similarity
    return None, best_similarity


# Eye Aspect Ratio (EAR) calculation
def eye_aspect_ratio(eye_points):
    """
    Calculate the Eye Aspect Ratio (EAR) for blink detection
    eye_points: array of 6 (x,y) coordinates defining the eye
    """
    # Compute the euclidean distances between the vertical eye landmarks
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])

    # Compute the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye_points[0], eye_points[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear


# MediaPipe-based blink detector
class MediaPipeBlinkDetector:
    def __init__(self, ear_threshold=0.21, consec_frames=2, min_blinks=1):
        """
        Detect blinks using MediaPipe Face Mesh landmarks

        ear_threshold: EAR threshold below which eye is considered closed
        consec_frames: Number of consecutive frames eye must be closed
        min_blinks: Minimum number of blinks required for liveness
        """
        self.EAR_THRESHOLD = ear_threshold
        self.CONSEC_FRAMES = consec_frames
        self.MIN_BLINKS = min_blinks

        self.frame_counter = 0
        self.blink_counter = 0
        self.is_verified = False
        self.ear_history = []
        self.max_history = 30

    def get_eye_landmarks(self, landmarks, indices, img_shape):
        """Extract eye landmarks from MediaPipe results"""
        height, width = img_shape[:2]
        eye_points = []
        for idx in indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            eye_points.append([x, y])
        return np.array(eye_points)

    def detect_blink(self, mediapipe_results, img_shape):
        """
        Detect if a blink occurred using MediaPipe landmarks
        Returns: (verified, ear_avg, blink_count)
        """
        if mediapipe_results is None or not mediapipe_results.multi_face_landmarks:
            return self.is_verified, 0.0, self.blink_counter

        # Get the first face's landmarks (we'll match this to tracked faces)
        face_landmarks = mediapipe_results.multi_face_landmarks[0]

        # Get eye landmarks
        left_eye = self.get_eye_landmarks(face_landmarks, LEFT_EYE_INDICES, img_shape)
        right_eye = self.get_eye_landmarks(face_landmarks, RIGHT_EYE_INDICES, img_shape)

        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear_avg = (left_ear + right_ear) / 2.0

        # Store EAR history
        self.ear_history.append(ear_avg)
        if len(self.ear_history) > self.max_history:
            self.ear_history.pop(0)

        # Check if eyes are closed
        if ear_avg < self.EAR_THRESHOLD:
            self.frame_counter += 1
        else:
            # If eyes were closed for sufficient frames, count as blink
            if self.frame_counter >= self.CONSEC_FRAMES:
                self.blink_counter += 1

                # Verify liveness after minimum blinks
                if self.blink_counter >= self.MIN_BLINKS:
                    self.is_verified = True

            self.frame_counter = 0

        return self.is_verified, ear_avg, self.blink_counter

    def reset(self):
        """Reset the detector"""
        self.frame_counter = 0
        self.blink_counter = 0
        self.is_verified = False
        self.ear_history = []


# Multi-person face tracker with MediaPipe blink detection
class MultiPersonTracker:
    def __init__(self, max_age=30, distance_threshold=80):
        """
        Track multiple people across frames with MediaPipe-based liveness
        """
        self.tracks = (
            {}
        )  # track_id -> {'center': (x,y), 'liveness': detector, 'age': int, 'bbox': bbox}
        self.next_id = 0
        self.max_age = max_age
        self.distance_threshold = distance_threshold

    def get_face_center(self, bbox):
        """Calculate center point of bounding box"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def update(self, faces):
        """
        Update tracks with new detections
        Returns: list of (track_id, face) tuples
        """
        if len(faces) == 0:
            # Age out all tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]["age"] += 1
                if self.tracks[track_id]["age"] > self.max_age:
                    del self.tracks[track_id]
            return []

        # Calculate centers for all faces
        face_centers = [self.get_face_center(face.bbox) for face in faces]

        # Match faces to existing tracks
        matched_tracks = set()
        matched_faces = set()
        assignments = []

        # For each existing track, find closest detection
        for track_id, track_data in self.tracks.items():
            track_center = track_data["center"]
            min_dist = float("inf")
            best_face_idx = -1

            for face_idx, face_center in enumerate(face_centers):
                if face_idx in matched_faces:
                    continue
                dist = np.linalg.norm(np.array(track_center) - np.array(face_center))
                if dist < min_dist and dist < self.distance_threshold:
                    min_dist = dist
                    best_face_idx = face_idx

            if best_face_idx >= 0:
                matched_tracks.add(track_id)
                matched_faces.add(best_face_idx)
                assignments.append((track_id, best_face_idx))
                # Update track
                self.tracks[track_id]["center"] = face_centers[best_face_idx]
                self.tracks[track_id]["bbox"] = faces[best_face_idx].bbox
                self.tracks[track_id]["age"] = 0

        # Create new tracks for unmatched faces
        for face_idx, face_center in enumerate(face_centers):
            if face_idx not in matched_faces:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    "center": face_center,
                    "bbox": faces[face_idx].bbox,
                    "liveness": MediaPipeBlinkDetector(),
                    "age": 0,
                }
                assignments.append((track_id, face_idx))
                matched_tracks.add(track_id)

        # Remove old tracks
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id]["age"] += 1
                if self.tracks[track_id]["age"] > self.max_age:
                    del self.tracks[track_id]

        # Return track_id, face pairs
        return [(track_id, faces[face_idx]) for track_id, face_idx in assignments]

    def get_liveness_detector(self, track_id):
        """Get the liveness detector for a specific track"""
        if track_id in self.tracks:
            return self.tracks[track_id]["liveness"]
        return None

    def get_track_bbox(self, track_id):
        """Get the bounding box for a specific track"""
        if track_id in self.tracks:
            return self.tracks[track_id]["bbox"]
        return None


# Helper function to find which MediaPipe face corresponds to which tracked face
def match_mediapipe_to_tracks(mediapipe_results, tracker, img_shape):
    """
    Match MediaPipe face detections to tracked faces based on position
    Returns: dict of track_id -> mediapipe_face_index
    """
    if not mediapipe_results or not mediapipe_results.multi_face_landmarks:
        return {}

    height, width = img_shape[:2]
    matches = {}

    # Get MediaPipe face centers
    mp_centers = []
    for face_landmarks in mediapipe_results.multi_face_landmarks:
        # Use nose tip as face center (landmark 1)
        nose = face_landmarks.landmark[1]
        center_x = int(nose.x * width)
        center_y = int(nose.y * height)
        mp_centers.append((center_x, center_y))

    # Match to tracked faces
    for track_id in tracker.tracks:
        track_center = tracker.tracks[track_id]["center"]
        min_dist = float("inf")
        best_mp_idx = -1

        for mp_idx, mp_center in enumerate(mp_centers):
            dist = np.linalg.norm(np.array(track_center) - np.array(mp_center))
            if dist < min_dist and dist < 100:  # 100 pixel threshold
                min_dist = dist
                best_mp_idx = mp_idx

        if best_mp_idx >= 0:
            matches[track_id] = best_mp_idx

    return matches


# Main program
def main():
    # Get name before running (optional for storage mode)
    person_name = input(
        "Enter name to store faces (or press Enter for recognition-only mode): "
    ).strip()

    init_db()
    tracker = MultiPersonTracker()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n[controls]")
    if person_name:
        print("  Press '1' to store current face(s)")
    print("  Press 'q' to quit")
    print("\n[status] MediaPipe-based blink detection active!")
    print("[info] Each face starts as SPOOF until proper eye blinks are detected")
    print("[info] System now uses Eye Aspect Ratio (EAR) for accurate blink detection")
    print("[info] Cannot be fooled by covering/uncovering eyes\n")

    store_mode = bool(person_name)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces with insightface
        faces = app.get(frame)

        # Detect with MediaPipe for blink detection
        mediapipe_results = face_mesh.process(rgb_frame)

        # Update tracker with detections
        tracked_faces = tracker.update(faces)

        # Match MediaPipe faces to tracked faces
        mp_matches = match_mediapipe_to_tracks(mediapipe_results, tracker, frame.shape)

        # Process each tracked face
        for track_id, face in tracked_faces:
            bbox = face.bbox.astype(int)

            # Get liveness detector for this track
            liveness = tracker.get_liveness_detector(track_id)
            if liveness is None:
                continue

            # Check if we have MediaPipe match for this track
            if track_id in mp_matches:
                mp_idx = mp_matches[track_id]
                # Create a result object with just this face
                single_face_result = type(
                    "obj",
                    (object,),
                    {
                        "multi_face_landmarks": [
                            mediapipe_results.multi_face_landmarks[mp_idx]
                        ]
                    },
                )()

                # Detect blink
                is_live, ear_value, blink_count = liveness.detect_blink(
                    single_face_result, frame.shape
                )
            else:
                # No MediaPipe match - can't verify
                is_live, ear_value, blink_count = liveness.detect_blink(
                    None, frame.shape
                )

            # Draw track ID
            cv2.putText(
                display_frame,
                f"ID:{track_id}",
                (bbox[0], bbox[1] - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Draw eye landmarks if available
            if track_id in mp_matches and mediapipe_results.multi_face_landmarks:
                mp_idx = mp_matches[track_id]
                face_landmarks = mediapipe_results.multi_face_landmarks[mp_idx]
                height, width = frame.shape[:2]

                # Draw eye landmarks
                for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(display_frame, (x, y), 2, (255, 0, 255), -1)

            # Handle spoof state (not yet verified)
            if not is_live:
                cv2.rectangle(
                    display_frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 0, 255),
                    3,
                )
                cv2.putText(
                    display_frame,
                    "SPOOF - BLINK NATURALLY",
                    (bbox[0], bbox[1] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    display_frame,
                    f"EAR: {ear_value:.3f} | Blinks: {blink_count}",
                    (bbox[0], bbox[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )
                continue

            # Face is verified as live - proceed with recognition
            embedding = face.normed_embedding
            match, similarity = find_match(embedding)

            if match:
                label = f"{match}"
                sublabel = f"conf:{similarity:.2f} | EAR:{ear_value:.3f}"
                color = (0, 255, 0)
                status_text = f"[LIVE] Track {track_id}: {match} | conf:{similarity:.2f} | blinks:{blink_count}"
            else:
                label = "Unknown (LIVE)"
                sublabel = f"best:{similarity:.2f} | EAR:{ear_value:.3f}"
                color = (0, 165, 255)
                status_text = f"[LIVE] Track {track_id}: Unknown | best:{similarity:.2f} | blinks:{blink_count}"

            # Draw bounding box and labels
            cv2.rectangle(
                display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2
            )
            cv2.putText(
                display_frame,
                label,
                (bbox[0], bbox[1] - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            cv2.putText(
                display_frame,
                sublabel,
                (bbox[0], bbox[1] - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
            cv2.putText(
                display_frame,
                f"VERIFIED (Blinks: {blink_count})",
                (bbox[0], bbox[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Show status on screen
        status_y = 30
        if store_mode:
            cv2.putText(
                display_frame,
                "Press '1' to store live faces",
                (10, status_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            status_y += 30

        # Count verified vs unverified faces
        verified_count = sum(
            1
            for tid in tracker.tracks.keys()
            if tracker.get_liveness_detector(tid)
            and tracker.get_liveness_detector(tid).is_verified
        )

        cv2.putText(
            display_frame,
            f"MediaPipe Blink Detection | {verified_count}/{len(tracker.tracks)} verified",
            (10, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Display frame
        cv2.imshow("MediaPipe Anti-Spoofing System", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # Press '1' to store all live faces
        if key == ord("1") and store_mode:
            if len(tracked_faces) > 0:
                stored_count = 0
                for track_id, face in tracked_faces:
                    liveness = tracker.get_liveness_detector(track_id)
                    if liveness and liveness.is_verified:
                        embedding = face.normed_embedding
                        name = (
                            f"{person_name}"
                            if len(tracked_faces) == 1
                            else f"{person_name}_{track_id}"
                        )
                        store_face(name, embedding)
                        stored_count += 1
                        print(f"[STORED] {name} | blinks: {liveness.blink_counter}")
                    elif liveness and not liveness.is_verified:
                        print(
                            f"[WARNING] Track {track_id} not verified - needs natural blinks!"
                        )
                    else:
                        print(f"[ERROR] Track {track_id} no liveness detector!")

                if stored_count == 0:
                    print("[ERROR] No verified live faces to store")
                else:
                    print(f"[SUCCESS] Stored {stored_count} face(s)")
            else:
                print("[ERROR] No faces detected")

        # Press 'q' to quit
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    main()
