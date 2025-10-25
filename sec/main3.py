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
import threading
import queue
import time

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
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]


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
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear


# MediaPipe-based blink detector
class MediaPipeBlinkDetector:
    def __init__(self, ear_threshold=0.21, consec_frames=2, min_blinks=1):
        self.EAR_THRESHOLD = ear_threshold
        self.CONSEC_FRAMES = consec_frames
        self.MIN_BLINKS = min_blinks

        self.frame_counter = 0
        self.blink_counter = 0
        self.is_verified = False
        self.ear_history = []
        self.max_history = 30

    def get_eye_landmarks(self, landmarks, indices, img_shape):
        height, width = img_shape[:2]
        eye_points = []
        for idx in indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            eye_points.append([x, y])
        return np.array(eye_points)

    def detect_blink(self, mediapipe_results, img_shape):
        if mediapipe_results is None or not mediapipe_results.multi_face_landmarks:
            return self.is_verified, 0.0, self.blink_counter

        face_landmarks = mediapipe_results.multi_face_landmarks[0]
        left_eye = self.get_eye_landmarks(face_landmarks, LEFT_EYE_INDICES, img_shape)
        right_eye = self.get_eye_landmarks(face_landmarks, RIGHT_EYE_INDICES, img_shape)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear_avg = (left_ear + right_ear) / 2.0

        self.ear_history.append(ear_avg)
        if len(self.ear_history) > self.max_history:
            self.ear_history.pop(0)

        if ear_avg < self.EAR_THRESHOLD:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.CONSEC_FRAMES:
                self.blink_counter += 1
                if self.blink_counter >= self.MIN_BLINKS:
                    self.is_verified = True
            self.frame_counter = 0

        return self.is_verified, ear_avg, self.blink_counter

    def reset(self):
        self.frame_counter = 0
        self.blink_counter = 0
        self.is_verified = False
        self.ear_history = []


# Async InsightFace processor
class AsyncInsightFaceProcessor:
    def __init__(self, app, interval=30):
        self.app = app
        self.interval = interval
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None
        self.frame_count = 0
        self.last_process_time = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _process_loop(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
                faces = self.app.get(frame)

                # Clear output queue and put new result
                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        break

                self.output_queue.put(faces)
                self.last_process_time = time.time()
            except queue.Empty:
                continue

    def submit_frame(self, frame):
        self.frame_count += 1
        if self.frame_count % self.interval == 0:
            # Clear input queue and submit new frame
            while not self.input_queue.empty():
                try:
                    self.input_queue.get_nowait()
                except queue.Empty:
                    break

            try:
                self.input_queue.put_nowait(frame.copy())
                return True
            except queue.Full:
                return False
        return False

    def get_result(self):
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None


# Multi-person face tracker with continuous tracking
class MultiPersonTracker:
    def __init__(self, max_age=30, distance_threshold=80):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
        self.distance_threshold = distance_threshold

    def get_face_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def get_distance(self, center1, center2):
        return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    def update_with_mediapipe(self, mediapipe_results, img_shape):
        """
        Update track positions using MediaPipe detections (every frame)
        This keeps tracking smooth even without InsightFace
        """
        if mediapipe_results is None or not mediapipe_results.multi_face_landmarks:
            # Age all tracks
            dead_tracks = []
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]["age"] += 1
                if self.tracks[track_id]["age"] > self.max_age:
                    dead_tracks.append(track_id)

            for track_id in dead_tracks:
                del self.tracks[track_id]
            return

        height, width = img_shape[:2]

        # Get MediaPipe face centers and bounding boxes
        mp_faces = []
        for face_landmarks in mediapipe_results.multi_face_landmarks:
            # Calculate bounding box from landmarks
            x_coords = [lm.x * width for lm in face_landmarks.landmark]
            y_coords = [lm.y * height for lm in face_landmarks.landmark]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Add some padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width, x_max + padding)
            y_max = min(height, y_max + padding)

            bbox = np.array([x_min, y_min, x_max, y_max])
            center = self.get_face_center(bbox)
            mp_faces.append({"bbox": bbox, "center": center})

        # Match MediaPipe faces to existing tracks
        matched_tracks = set()
        matched_mp = set()

        for track_id, track_info in list(self.tracks.items()):
            best_mp_idx = None
            best_distance = float("inf")

            for mp_idx, mp_face in enumerate(mp_faces):
                if mp_idx in matched_mp:
                    continue

                distance = self.get_distance(track_info["center"], mp_face["center"])
                if distance < self.distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_mp_idx = mp_idx

            if best_mp_idx is not None:
                # Update track position with MediaPipe detection
                self.tracks[track_id]["center"] = mp_faces[best_mp_idx]["center"]
                self.tracks[track_id]["bbox"] = mp_faces[best_mp_idx]["bbox"]
                self.tracks[track_id]["age"] = 0
                matched_tracks.add(track_id)
                matched_mp.add(best_mp_idx)

        # Age unmatched tracks
        dead_tracks = []
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id]["age"] += 1
                if self.tracks[track_id]["age"] > self.max_age:
                    dead_tracks.append(track_id)

        for track_id in dead_tracks:
            del self.tracks[track_id]

        # Create new tracks for unmatched MediaPipe faces
        for mp_idx, mp_face in enumerate(mp_faces):
            if mp_idx not in matched_mp:
                self.tracks[self.next_id] = {
                    "center": mp_face["center"],
                    "bbox": mp_face["bbox"],
                    "liveness": MediaPipeBlinkDetector(),
                    "face": None,
                    "embedding": None,
                    "match": None,
                    "similarity": 0.0,
                    "age": 0,
                }
                self.next_id += 1

    def update_with_insightface(self, faces):
        """
        Update track embeddings with InsightFace detections (periodic)
        """
        if len(faces) == 0:
            return

        # Match InsightFace detections to existing tracks
        for face in faces:
            face_center = self.get_face_center(face.bbox)
            best_track = None
            best_distance = float("inf")

            for track_id, track_info in self.tracks.items():
                distance = self.get_distance(face_center, track_info["center"])
                if distance < self.distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_track = track_id

            if best_track is not None:
                # Update track with InsightFace data
                self.tracks[best_track]["face"] = face
                self.tracks[best_track]["embedding"] = face.normed_embedding
                self.tracks[best_track]["bbox"] = face.bbox

                # Update match in database
                match, similarity = find_match(face.normed_embedding)
                self.tracks[best_track]["match"] = match
                self.tracks[best_track]["similarity"] = similarity

    def get_liveness_detector(self, track_id):
        if track_id in self.tracks:
            return self.tracks[track_id]["liveness"]
        return None


def match_mediapipe_to_tracks(mediapipe_results, tracker, img_shape):
    if (
        mediapipe_results is None
        or not mediapipe_results.multi_face_landmarks
        or len(tracker.tracks) == 0
    ):
        return {}

    height, width = img_shape[:2]
    matches = {}

    mp_centers = []
    for face_landmarks in mediapipe_results.multi_face_landmarks:
        nose = face_landmarks.landmark[1]
        mp_centers.append((nose.x * width, nose.y * height))

    for track_id, track_info in tracker.tracks.items():
        track_center = track_info["center"]
        best_mp_idx = None
        best_distance = float("inf")

        for mp_idx, mp_center in enumerate(mp_centers):
            distance = tracker.get_distance(track_center, mp_center)
            if distance < best_distance:
                best_distance = distance
                best_mp_idx = mp_idx

        if best_distance < 100:
            matches[track_id] = best_mp_idx

    return matches


def main():
    init_db()

    import sys

    if len(sys.argv) > 1:
        store_mode = True
        person_name = sys.argv[1]
        print(f"[mode] STORE mode - will save faces as '{person_name}'")
    else:
        store_mode = False
        print("[mode] RECOGNITION mode")

    cap = cv2.VideoCapture(0)
    tracker = MultiPersonTracker()

    # Start async InsightFace processor (runs every 30 frames)
    insight_processor = AsyncInsightFaceProcessor(app, interval=30)
    insight_processor.start()

    print("[info] starting video stream...")
    print("[info] Press 'q' to quit")
    if store_mode:
        print("[info] Press '1' to store verified live faces")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = frame.copy()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe (every frame for smooth tracking)
            mediapipe_results = face_mesh.process(rgb_frame)

            # Update tracker with MediaPipe (keeps tracking smooth)
            tracker.update_with_mediapipe(mediapipe_results, frame.shape)

            # Submit frame to async InsightFace processor
            submitted = insight_processor.submit_frame(frame)

            # Check for InsightFace results
            insight_faces = insight_processor.get_result()
            if insight_faces is not None:
                tracker.update_with_insightface(insight_faces)

            # Match MediaPipe faces to tracked faces
            mp_matches = match_mediapipe_to_tracks(
                mediapipe_results, tracker, frame.shape
            )

            # Process each tracked face
            for track_id, track_info in tracker.tracks.items():
                bbox = track_info["bbox"].astype(int)
                liveness = tracker.get_liveness_detector(track_id)

                if liveness is None:
                    continue

                # Detect blink with MediaPipe
                if track_id in mp_matches:
                    mp_idx = mp_matches[track_id]
                    single_face_result = type(
                        "obj",
                        (object,),
                        {
                            "multi_face_landmarks": [
                                mediapipe_results.multi_face_landmarks[mp_idx]
                            ]
                        },
                    )()
                    is_live, ear_value, blink_count = liveness.detect_blink(
                        single_face_result, frame.shape
                    )
                else:
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

                # Draw eye landmarks
                if track_id in mp_matches and mediapipe_results.multi_face_landmarks:
                    mp_idx = mp_matches[track_id]
                    face_landmarks = mediapipe_results.multi_face_landmarks[mp_idx]
                    height, width = frame.shape[:2]

                    for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        cv2.circle(display_frame, (x, y), 2, (255, 0, 255), -1)

                # Get match from cache
                match = track_info.get("match", None)
                similarity = track_info.get("similarity", 0.0)

                # If person NOT in DB, mark as Unknown
                if not match:
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
                        (bbox[0], bbox[1] - 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"best:{similarity:.2f} | EAR: {ear_value:.3f} | Blinks: {blink_count}",
                        (bbox[0], bbox[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 165, 255),
                        1,
                    )
                    continue

                # Person IS in DB - check liveness
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

                # Face is verified as live AND in DB
                label = f"{match}"
                sublabel = f"conf:{similarity:.2f} | EAR:{ear_value:.3f}"
                color = (0, 255, 0)

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

            # Show status
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

            verified_count = sum(
                1
                for tid in tracker.tracks.keys()
                if tracker.get_liveness_detector(tid)
                and tracker.get_liveness_detector(tid).is_verified
            )

            # Show processing status
            processing_status = "PROCESSING" if submitted else "async running"
            cv2.putText(
                display_frame,
                f"Tracking: MediaPipe | Recognition: {processing_status}",
                (10, status_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            status_y += 30

            cv2.putText(
                display_frame,
                f"Blink Detection | {verified_count}/{len(tracker.tracks)} verified",
                (10, status_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Async Anti-Spoofing System", display_frame)

            key = cv2.waitKey(1) & 0xFF

            # Press '1' to store all live faces
            if key == ord("1") and store_mode:
                if len(tracker.tracks) > 0:
                    stored_count = 0
                    for track_id, track_info in tracker.tracks.items():
                        liveness = tracker.get_liveness_detector(track_id)
                        if (
                            liveness
                            and liveness.is_verified
                            and track_info.get("embedding") is not None
                        ):
                            embedding = track_info["embedding"]
                            name = (
                                f"{person_name}"
                                if len(tracker.tracks) == 1
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
                            print(f"[ERROR] Track {track_id} no embedding yet!")

                    if stored_count == 0:
                        print("[ERROR] No verified live faces with embeddings to store")
                    else:
                        print(f"[SUCCESS] Stored {stored_count} face(s)")
                else:
                    print("[ERROR] No faces detected")

            if key == ord("q"):
                break

    finally:
        insight_processor.stop()
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()


if __name__ == "__main__":
    main()
