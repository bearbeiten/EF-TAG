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


# Find matching face - returns match info and whether person exists in DB
def find_match(embedding, threshold=0.3):
    faces = get_all_faces()
    if len(faces) == 0:
        return None, -1.0, False  # No match, similarity, not in DB

    best_match = None
    best_similarity = -1

    for name, stored_embedding in faces:
        similarity = cosine_similarity(embedding, stored_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = name

    if best_similarity > threshold:
        return best_match, best_similarity, True  # Match found, in DB
    return None, best_similarity, False  # No match, not in DB


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
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, faces):
        """
        Update tracks with new face detections
        Returns: list of (track_id, face) tuples for active tracks
        """
        # Mark all existing tracks as unseen
        for track_id in self.tracks:
            self.tracks[track_id]["age"] += 1

        # Match detected faces to existing tracks
        assigned_tracks = set()
        updated_tracks = []

        for face in faces:
            bbox = face.bbox.astype(int)
            center = self.get_face_center(bbox)

            # Find closest unassigned track
            best_track = None
            best_dist = float("inf")

            for track_id, track_data in self.tracks.items():
                if track_id in assigned_tracks:
                    continue

                # Calculate distance between centers
                track_center = track_data["center"]
                dist_val = np.sqrt(
                    (center[0] - track_center[0]) ** 2
                    + (center[1] - track_center[1]) ** 2
                )

                if dist_val < best_dist and dist_val < self.distance_threshold:
                    best_dist = dist_val
                    best_track = track_id

            # Update existing track or create new
            if best_track is not None:
                self.tracks[best_track]["center"] = center
                self.tracks[best_track]["bbox"] = bbox
                self.tracks[best_track]["age"] = 0
                assigned_tracks.add(best_track)
                updated_tracks.append((best_track, face))
            else:
                # Create new track
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {
                    "center": center,
                    "bbox": bbox,
                    "liveness": MediaPipeBlinkDetector(),
                    "age": 0,
                }
                updated_tracks.append((new_id, face))
                print(f"[new track] ID {new_id} created")

        # Remove old tracks
        tracks_to_remove = [
            tid for tid, tdata in self.tracks.items() if tdata["age"] > self.max_age
        ]
        for tid in tracks_to_remove:
            print(f"[track lost] ID {tid} removed")
            del self.tracks[tid]

        return updated_tracks

    def get_liveness_detector(self, track_id):
        """Get liveness detector for a specific track"""
        if track_id in self.tracks:
            return self.tracks[track_id]["liveness"]
        return None


def match_mediapipe_to_tracks(mediapipe_results, tracker, frame_shape):
    """
    Match MediaPipe detected faces to tracked faces based on center proximity
    Returns: dict mapping track_id -> mediapipe_face_index
    """
    if (
        mediapipe_results is None
        or not mediapipe_results.multi_face_landmarks
        or not tracker.tracks
    ):
        return {}

    height, width = frame_shape[:2]
    matches = {}

    # Get centers of MediaPipe faces
    mp_centers = []
    for face_landmarks in mediapipe_results.multi_face_landmarks:
        # Use nose tip (landmark 1) as center
        nose = face_landmarks.landmark[1]
        center = (int(nose.x * width), int(nose.y * height))
        mp_centers.append(center)

    # Match each track to closest MediaPipe face
    for track_id, track_data in tracker.tracks.items():
        track_center = track_data["center"]
        best_mp_idx = None
        best_dist = float("inf")

        for mp_idx, mp_center in enumerate(mp_centers):
            if mp_idx in matches.values():
                continue  # Already assigned

            dist_val = np.sqrt(
                (track_center[0] - mp_center[0]) ** 2
                + (track_center[1] - mp_center[1]) ** 2
            )

            if dist_val < best_dist and dist_val < 100:  # threshold
                best_dist = dist_val
                best_mp_idx = mp_idx

        if best_mp_idx is not None:
            matches[track_id] = best_mp_idx

    return matches


def main():
    init_db()

    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-person face recognition with MediaPipe anti-spoofing"
    )
    parser.add_argument("--store", type=str, help="Name of person to store (optional)")
    args = parser.parse_args()

    person_name = args.store

    cap = cv2.VideoCapture(0)
    tracker = MultiPersonTracker()

    print("\n" + "=" * 60)
    print("MEDIAPIPE ANTI-SPOOFING FACE RECOGNITION")
    print("=" * 60)
    print("  Press '1' to store detected LIVE faces (if --store specified)")
    print("  Press 'q' to quit")
    print("\n[status] MediaPipe-based blink detection active!")
    print("[info] Unknown for faces not in database")
    print("[info] SPOOF only for known faces that haven't blinked yet")
    print("[info] System uses Eye Aspect Ratio (EAR) for accurate blink detection\n")

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

            # FIRST: Check if person is in database
            embedding = face.normed_embedding
            match, similarity, in_database = find_match(embedding)

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

            # NEW LOGIC: Handle based on DB presence and liveness
            if not in_database:
                # Person NOT in database -> Mark as Unknown (regardless of liveness)
                cv2.rectangle(
                    display_frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 165, 255),
                    3,
                )
                cv2.putText(
                    display_frame,
                    "Unknown",
                    (bbox[0], bbox[1] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )
                cv2.putText(
                    display_frame,
                    f"Not in DB | best:{similarity:.2f}",
                    (bbox[0], bbox[1] - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 165, 255),
                    1,
                )
                cv2.putText(
                    display_frame,
                    f"EAR: {ear_value:.3f} | Blinks: {blink_count}",
                    (bbox[0], bbox[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 165, 255),
                    1,
                )
                continue

            # Person IS in database
            if not is_live:
                # Known person but not verified live -> Mark as SPOOF
                cv2.rectangle(
                    display_frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 0, 255),
                    3,
                )
                cv2.putText(
                    display_frame,
                    f"{match} - SPOOF",
                    (bbox[0], bbox[1] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    display_frame,
                    "BLINK NATURALLY TO VERIFY",
                    (bbox[0], bbox[1] - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    display_frame,
                    f"EAR: {ear_value:.3f} | Blinks: {blink_count}",
                    (bbox[0], bbox[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )
                continue

            # Known person AND verified live -> Show as verified
            label = f"{match}"
            sublabel = f"conf:{similarity:.2f} | EAR:{ear_value:.3f}"
            color = (0, 255, 0)

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
